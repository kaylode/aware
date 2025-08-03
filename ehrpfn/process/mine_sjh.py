import os
import sys
import polars as pl
from tqdm import tqdm
from ehrpfn.utils.vocabulary import Vocabulary
sys.path.append("/home/mpham/workspace/source/tabpfn-ehr")

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from theseus import seed_everything
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

seed_everything(42)


def process_activity_data(df):
    """Process activity data with code transformations and filtering."""

    activity_df = df.with_columns(
        pl.when(
            pl.col("code").str.starts_with("TRANSFER//") |
            pl.col("code").str.starts_with("CPE_SCREEN//") |
            pl.col("code").str.starts_with("DISCHARGE_LOCATION")
        )
        .then("code")
        .otherwise(pl.col("code").str.split("//").list.get(0))
        .alias("code_prefix")
    )

    activity_df = activity_df.filter(
        (
            pl.col("code_prefix").str.starts_with("ICD") |
            pl.col("code_prefix").str.starts_with("DISCHARGE_LOCATION") |
            pl.col("code_prefix").str.starts_with("CPE_SCREEN") |
            pl.col("code_prefix").str.starts_with("TRANSFER")
        ) & (pl.col("code_prefix").is_in(["TRANSFER//UNKNOWN", 'DISCHARGE_LOCATION//OTHER']).not_())
    )

    activity_df = activity_df.select([
        "subject_id",
        "admission_id",
        "time",
        "code_prefix",
    ]).group_by([
        "subject_id",
        "admission_id",
        "code_prefix"
    ]).agg([
        pl.col("time").min().alias("time")
    ]).sort(["subject_id", "admission_id", "time"])

    return activity_df



def get_unique_pathway_tokens(process_df, pathway_columns=None):
    """
    Extract all unique tokens from pathway columns.
    
    Args:
        process_df: Polars DataFrame containing pathway data
        pathway_columns: List of column names to extract tokens from. 
                        If None, uses all columns containing 'pathway'
    
    Returns:
        List of unique tokens sorted alphabetically
    """
    if pathway_columns is None:
        # Get all columns that contain 'pathway'
        pathway_columns = [col for col in process_df.columns if 'pathway' in col.lower()]
    
    # Collect all tokens from all pathway columns
    all_tokens = set()
    
    for col in pathway_columns:
        # Get all non-null values from the column
        pathway_values = process_df.select(pl.col(col)).drop_nulls().to_series().to_list()
        
        for pathway in pathway_values:
            if pathway:  # Check if not empty string
                # Split by both comma and semicolon to handle different separators
                tokens = pathway.replace(';', ',').split(',')
                # Clean tokens and add to set
                for token in tokens:
                    cleaned_token = token.strip()
                    if cleaned_token:  # Only add non-empty tokens
                        all_tokens.add(cleaned_token)
    
    return sorted(list(all_tokens))


def encode_pathway_columns(process_df, vocab, pathway_columns=None):
    """
    Encode pathway columns using vocabulary, converting tokens to IDs.
    
    Args:
        process_df: Polars DataFrame containing pathway data
        vocab: Vocabulary object with word-to-id mapping
        pathway_columns: List of column names to encode. If None, uses all columns containing 'pathway'
    
    Returns:
        DataFrame with additional encoded columns
    """
    if pathway_columns is None:
        pathway_columns = [col for col in process_df.columns if 'pathway' in col.lower()]
    
    df_encoded = process_df.clone()
    
    for col in pathway_columns:
        # Create encoded column name
        encoded_col = f"{col}_encoded"
        
        # Apply encoding using map_elements
        df_encoded = df_encoded.with_columns(
            pl.col(col)
            .map_elements(
                lambda pathway: encode_pathway(pathway, vocab) if pathway else None,
                return_dtype=pl.String
            )
            .alias(encoded_col)
        )
    
    return df_encoded

def encode_pathway(pathway, vocab):
    """
    Encode a single pathway string to space-separated token IDs.
    Replaces ';' with '[SEP]' and ',' with spaces.
    
    Args:
        pathway: String containing comma/semicolon separated tokens
        vocab: Vocabulary object
    
    Returns:
        String of space-separated token IDs with '[SEP]' between admissions
    """
    if not pathway:
        return None
    
    # Split by semicolon first to handle admission boundaries
    admission_parts = pathway.split(';')
    encoded_admissions = []
    
    for admission in admission_parts:
        if admission.strip():  # Only process non-empty admissions
            # Split by comma within each admission
            tokens = admission.split(',')
            # Clean tokens: strip whitespace and remove any existing spaces
            cleaned_tokens = [token.strip().replace(' ', '') for token in tokens if token.strip()]
            
            # Convert tokens to IDs
            token_ids = [str(vocab.stoi.get(token, vocab.stoi.get('[UNK]', 0))) for token in cleaned_tokens]
            
            # Join tokens within admission with space
            encoded_admission = ' '.join(token_ids)
            encoded_admissions.append(encoded_admission)
    
    # Join admissions with [SEP]
    return f' {vocab.stoi.get('[SEP]', 0)} '.join(encoded_admissions)

def mine_using_vocabulary(activity_df, DATASET_PATH, SPLIT):
    # Create process pathways with backward-looking features
    process_df = activity_df.group_by(["subject_id", "admission_id"]).agg(
        pl.col("code_prefix").str.join(",").alias("pathway")
    )

    process_df = process_df.sort(["subject_id", "admission_id"])

    # Add backward-looking pathway columns
    process_df = process_df.with_columns([
        # Get previous pathway (lag 1)
        pl.col("pathway").shift(1).over("subject_id").alias("prev_pathway_1"),
        # Get pathway from 2 steps back (lag 2)
        pl.col("pathway").shift(2).over("subject_id").alias("prev_pathway_2")
    ]).with_columns([
        # Concat current with 1 previous pathway
        pl.when(pl.col("prev_pathway_1").is_null())
        .then(pl.col("pathway"))
        .otherwise(pl.col("prev_pathway_1") + pl.lit(",") + pl.col("pathway"))
        .alias("pathway_with_prev_1"),
        
        # Concat current with 2 previous pathways
        pl.when(pl.col("prev_pathway_2").is_null() | pl.col("prev_pathway_1").is_null())
        .then(
            pl.when(pl.col("prev_pathway_1").is_null())
            .then(pl.col("pathway"))
            .otherwise(pl.col("prev_pathway_1") + pl.lit(";") + pl.col("pathway"))
        )
        .otherwise(
            pl.col("prev_pathway_2") + pl.lit(";") + 
            pl.col("prev_pathway_1") + pl.lit(";") + 
            pl.col("pathway")
        )
        .alias("pathway_with_prev_2")
    ])

    # Extract unique tokens and create vocabulary
    unique_tokens = get_unique_pathway_tokens(process_df)
    print(f"Found {len(unique_tokens)} unique tokens:")
    print(unique_tokens[:10])  # Show first 10 tokens
    SPECIAL_TOKENS = ['[UNK]', '[PAD]', '[SEP]']


    if SPLIT == 'train':
        vocab = Vocabulary()
        vocab.add_words(SPECIAL_TOKENS)
        vocab.add_words(unique_tokens)
        vocab.dump(f"{DATASET_PATH}/process")
    else:
        vocab = Vocabulary.from_path(f"{DATASET_PATH}/process")

    # Apply encoding to the process_df
    encoded_process_df = encode_pathway_columns(process_df, vocab)
    encoded_process_df.write_csv(f"{DATASET_PATH}/process/process_pathways_encoded_{SPLIT}.csv")


def convert_to_events(input_path, ROOT_DIR, DATASET_PATH, SPLIT):
    """Main processing function."""
    # Configuration


    # Load and filter main dataframe
    df = pl.read_parquet(input_path)
    df = df.select([
        "subject_id",
        'admission_id',
        "patientid",
        "time",
        "code",
        "numeric_value",
        "text_value"
    ])


    episode_id_df = df.select([
        "subject_id",
        "admission_id",
        "patientid"
    ]).unique()

    # vocab = Vocabulary.from_path(f"{DATASET_PATH}/process")

    # Load and clean attribute data
    # attribute_df = pl.read_parquet(f"{DATASET_PATH}/2_numpify/{SPLIT}")
    # attribute_df = attribute_df.select([
    #     "subject_id", "admission_id",
    #     "AGE", "RACE", "LOS", "MARITAL", 
    #     "INSURANCE", "GENDER", "VITAL//ED_ACUITY"
    # ]).unique()
    # attribute_df = clean_attribute_data(attribute_df)

    # Load labels
    # all_label_df = load_labels(ROOT_DIR, SPLIT)

    # # Process activity data
    # procedure_codes = get_procedure_codes()
    activity_df = process_activity_data(df)

    # # Join with attributes and labels
    # activity_df = activity_df.join(
    #     attribute_df,
    #     on=["subject_id", "admission_id"],
    #     how="left"
    # )

    # activity_df = activity_df.join(
    #     all_label_df,
    #     on=["subject_id", "admission_id"],
    #     how="left"
    # )
    # # Save results
    # activity_df.write_csv(f"{DATASET_PATH}/process/event_logs_{SPLIT}.csv")

    return activity_df, episode_id_df

def mine_using_cluster(activity_df, n_clusters=20, model:KMeans=None, vectorizer:TfidfVectorizer=None, plot_path=None):

    # Create process pathways with backward-looking features
    process_df = activity_df.group_by(["subject_id", "admission_id"]).agg(
        pl.col("code_prefix").str.join(",").alias("pathway")
    )

    process_df = process_df.sort(["subject_id", "admission_id"])

    # Add backward-looking pathway columns
    process_df = process_df.with_columns([
        # Get previous pathway (lag 1)
        pl.col("pathway").shift(1).over("subject_id").alias("prev_pathway_1"),
        # Get pathway from 2 steps back (lag 2)
        pl.col("pathway").shift(2).over("subject_id").alias("prev_pathway_2")
    ]).with_columns([
        # Concat current with 1 previous pathway
        pl.when(pl.col("prev_pathway_1").is_null())
        .then(pl.col("pathway"))
        .otherwise(pl.col("prev_pathway_1") + pl.lit(",") + pl.col("pathway"))
        .alias("pathway_with_prev_1"),
        
        # Concat current with 2 previous pathways
        pl.when(pl.col("prev_pathway_2").is_null() | pl.col("prev_pathway_1").is_null())
        .then(
            pl.when(pl.col("prev_pathway_1").is_null())
            .then(pl.col("pathway"))
            .otherwise(pl.col("prev_pathway_1") + pl.lit(";") + pl.col("pathway"))
        )
        .otherwise(
            pl.col("prev_pathway_2") + pl.lit(";") + 
            pl.col("prev_pathway_1") + pl.lit(";") + 
            pl.col("pathway")
        )
        .alias("pathway_with_prev_2")
    ])

    # Create new column for combined IDs of subject_id and admission_id
    process_df = process_df.with_columns(
        (pl.col("subject_id").cast(pl.String) + "_" + pl.col("admission_id").cast(pl.String)).alias("case_id")
    )
    variants = process_df.select("pathway").to_series().to_list()
    case_ids = process_df.select("case_id").to_series().to_list()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','))
        X = vectorizer.fit_transform(variants)  # shape: [n_traces, n_activities]
    else:
        X = vectorizer.transform(variants)

    if model is None:
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
    else:
        labels = model.predict(X)

    cluster_to_cases = defaultdict(list)
    for idx, label in zip(case_ids, labels):
        cluster_to_cases[label].append(idx)

    cluster_df = {
        "cluster_id": [],
        "admission_id": [],
        "subject_id": [],
    }
    for cluster_id, cases in cluster_to_cases.items():
        for case in cases:
            cluster_df["subject_id"].append(case.split('_')[0])
            cluster_df["admission_id"].append(case.split('_')[1])
            cluster_df["cluster_id"].append(cluster_id)
    cluster_df = pl.DataFrame(cluster_df)

    print("Cluster distribution:")
    cluster_counts = cluster_df['cluster_id'].value_counts().sort('cluster_id')
    print(cluster_counts)

    # Create distribution plots if plot_path is provided
    if plot_path:
        # Convert to pandas for easier plotting
        cluster_counts_pd = cluster_counts.to_pandas()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cases per Cluster', 'Distribution of Cluster Sizes', 
                          'Top 10 Clusters Distribution', 'Cluster Size Statistics'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "box"}]]
        )
        
        # 1. Bar plot of cluster counts
        fig.add_trace(
            go.Bar(
                x=cluster_counts_pd['cluster_id'].astype(str),
                y=cluster_counts_pd['count'],
                name='Cases per Cluster',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Histogram of cluster sizes
        fig.add_trace(
            go.Histogram(
                x=cluster_counts_pd['count'],
                nbinsx=min(20, len(cluster_counts_pd)),
                name='Cluster Size Distribution',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Pie chart of top clusters (show top 10)
        top_clusters = cluster_counts_pd.nlargest(10, 'count')
        other_count = cluster_counts_pd['count'].sum() - top_clusters['count'].sum()
        
        if other_count > 0:
            pie_labels = list(top_clusters['cluster_id'].astype(str)) + ['Others']
            pie_values = list(top_clusters['count']) + [other_count]
        else:
            pie_labels = list(top_clusters['cluster_id'].astype(str))
            pie_values = list(top_clusters['count'])
        
        fig.add_trace(
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                name='Top Clusters',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Box plot of cluster sizes
        fig.add_trace(
            go.Box(
                y=cluster_counts_pd['count'],
                name='Cluster Sizes',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Cluster Distribution Analysis",
            title_x=0.5
        )
        
        # Update x-axis labels for bar plot
        fig.update_xaxes(title_text="Cluster ID", row=1, col=1)
        fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
        
        # Update x-axis labels for histogram
        fig.update_xaxes(title_text="Cluster Size", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        # Update y-axis labels for box plot
        fig.update_yaxes(title_text="Cluster Size", row=2, col=2)
        
        # Save the plot
        fig.write_image(plot_path, width=1200, height=800)
        print(f"Cluster distribution plot saved to: {plot_path}")
        
        # Print summary statistics
        print("\nCluster Statistics:")
        print(f"Total clusters: {len(cluster_counts_pd)}")
        print(f"Total cases: {cluster_counts_pd['count'].sum()}")
        print(f"Average cluster size: {cluster_counts_pd['count'].mean():.2f}")
        print(f"Median cluster size: {cluster_counts_pd['count'].median():.2f}")
        print(f"Largest cluster: {cluster_counts_pd['count'].max()} cases")
        print(f"Smallest cluster: {cluster_counts_pd['count'].min()} cases")

    cluster_df = cluster_df.with_columns(
        pl.col("admission_id").cast(pl.String),
        pl.col("subject_id").cast(pl.String),
    )

    process_df = process_df.with_columns(
        pl.col("admission_id").cast(pl.String),
        pl.col("subject_id").cast(pl.String),
    )

    cluster_df = cluster_df.join(
        process_df.select(["subject_id", "admission_id", "pathway"]),
        on=["subject_id", "admission_id"],
        how="left"
    )

    return vectorizer, model, cluster_df

if __name__ == "__main__":
    for N_CLUSTERS in [8, 20, 50, 70, 100, 120]:
        print(f"Mining using {N_CLUSTERS} clusters...")

        OUTPUT_PATH = f"/home/mpham/workspace/source/tabpfn-ehr/outputs"
        DATASET_NAME = "sjh_cpe"
        DATASET_PATH = f"{OUTPUT_PATH}/tabpfn/data/{DATASET_NAME}"
        os.makedirs(f"{DATASET_PATH}/process", exist_ok=True)

        kmean_model = None
        tfidf_vectorizer = None

        for SPLIT in ['train', 'test']:
            ROOT_DIR = f"/home/mpham/spinning/data/ehrfm/sjh-cpe-1.0/tokenized_datasets/sjh-cpe/{SPLIT}"
            input_path = f"{ROOT_DIR}/09_FinalizeData" if SPLIT == 'train' else f"{ROOT_DIR}/08_FinalizeData"
            activity_df, episode_id_df = convert_to_events(input_path, ROOT_DIR, DATASET_PATH, SPLIT)
            mine_using_vocabulary(activity_df, DATASET_PATH, SPLIT)
            # if SPLIT == 'train':
            #     tfidf_vectorizer, kmean_model, cluster_df = mine_using_cluster(
            #         activity_df,  
            #         n_clusters=N_CLUSTERS,
            #         plot_path=f"{DATASET_PATH}/process/{N_CLUSTERS}_cluster_distribution_{SPLIT}.pdf"
            #     )
            # else:
            #     _, _, cluster_df = mine_using_cluster(
            #         activity_df, n_clusters=N_CLUSTERS, 
            #         model=kmean_model, vectorizer=tfidf_vectorizer,
            #         plot_path=f"{DATASET_PATH}/process/{N_CLUSTERS}_cluster_distribution_{SPLIT}.pdf"
            #     )

            # episode_id_df = episode_id_df.with_columns(
            #     pl.col("admission_id").cast(pl.String),
            #     pl.col("subject_id").cast(pl.String),
            # )

            # cluster_df = cluster_df.join(
            #     episode_id_df,
            #     on=["subject_id", "admission_id"],
            #     how="left"
            # )
            # # Save cluster results
            # cluster_df.write_csv(f"{DATASET_PATH}/process/{N_CLUSTERS}_clustered_cases_{SPLIT}.csv")