from .faiss_retriever import (
    FAISS,
    WeightedRetriever,
    RandomRetriever,
    BaselineRetriever,
    get_ehr_retrieval_weight
)
from .base import Retriever
import os
from .supcon import SupConRetriever, SupConEncoder
from .unsupcon import UnsupConRetriever, UnsupConEncoder
from .balance import BalanceRetriever, MinorityPrioritizedRetriever, InverseFrequencyRetriever
from .aware_imb import ImbalanceAwareAWARE
from .aware import SNNLAttnEncoder, SNNLAttnRetriever


def get_retriever(method:str, k:int, distance:str='l2', field_to_use:str='raw_feature', **kwargs):
    if method in ["faiss", 'knn']:
        retriever = FAISS(
            k=k, distance=distance, 
            field_to_use=field_to_use,
            return_distances=kwargs.get('return_distances', False)
        )
    elif method in ['weighted']:
        assert 'weights' in kwargs, "weights must be provided for weighted retrieval"
        retriever = WeightedRetriever(
            k=k,
            field_to_use=field_to_use,
            weights=kwargs.get('weights'),
            distance=distance,
            return_distances=kwargs.get('return_distances', False)
        )
    elif method == 'random':
        retriever = RandomRetriever(
            k=k,
            field_to_use=field_to_use
        )
    elif method == 'baseline':
        retriever = BaselineRetriever(
            k=k,
            field_to_use=field_to_use
        )
    elif method == 'minority_prioritized':
        retriever = MinorityPrioritizedRetriever(
            k=k,
            distance=distance,
            field_to_use=field_to_use,
            return_distances=kwargs.get('return_distances', False),
            positive_ratio=kwargs.get('positive_ratio', 0.7)
        )
    elif method == 'inverse_frequency':
        retriever = InverseFrequencyRetriever(
            k=k,
            distance=distance,
            field_to_use=field_to_use,
            return_distances=kwargs.get('return_distances', False),
        )
    elif method == 'siamese':
        from .siamese import SiameseEHREncoder, TripletLossRetriever
        retriever = TripletLossRetriever(
            model=SiameseEHREncoder(
                n_features=kwargs.get('n_features', 100),
                embedding_dim=kwargs.get('embedding_dim', 128)
            ),
            margin=kwargs.get('margin', 1.0),
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    elif method == 'supcon':
        from .supcon import SupConEncoder, SupConRetriever
        retriever = SupConRetriever(
            model=SupConEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim=kwargs.get('hidden_dim', 512),
                embedding_dim=kwargs.get('embedding_dim', 128),
                dropout=kwargs.get('dropout', 0.3)
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    elif method == 'unsupcon':
        from .unsupcon import UnsupConEncoder, UnsupConRetriever
        retriever = UnsupConRetriever(
            model=UnsupConEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim=kwargs.get('hidden_dim', 512),
                embedding_dim=kwargs.get('embedding_dim', 128),
                dropout=kwargs.get('dropout', 0.3)
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            temperature=kwargs.get('temperature', 0.07),
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Ablation: Attention only (Attention enabled, but no Metric Learning)
    elif method == 'aware_attn_only':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=True
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 0.0, # Pure classification loss, no metric learning
            use_balance_sampling = False,
            use_ensemble = False,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Ablation: SNNL only (Metric Learning enabled, but no Attention Scaling)
    elif method == 'aware_snnl_only':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=False
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 1.0, # Pure SNNL loss
            use_balance_sampling = False,
            use_cosine_loss=True,
            use_ensemble = False,
            pretrained_path=kwargs.get('pretrained_path', None)
        )


    # Ablation: SNNL with Attention
    elif method == 'aware_snnl_attn_only':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=True
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 1.0, # Pure SNNL loss
            use_balance_sampling = False,
            use_cosine_loss=True,
            use_ensemble = False,
            pretrained_path=kwargs.get('pretrained_path', None)
        )

    # Ablation: SNNL with Attention and balance sampling
    elif method == 'aware_snnl_attn_bln_only':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=True
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 1.0, # Pure SNNL loss
            use_balance_sampling = True,
            use_cosine_loss=True,
            use_ensemble = False,
            pretrained_path=kwargs.get('pretrained_path', None)
        )

    # Ablation: Balanced only (Metric Learning + Balanced, no Attention, no Ensemble)
    elif method == 'aware_balance_only':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=False
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            use_balance_sampling = True,
            use_ensemble = False,
            alpha = 1.0,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Ablation: Ensemble only (Metric Learning + Ensemble, no Attention, no Balance)
    elif method == 'aware_ensemble_only':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=False
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            use_balance_sampling = False,
            use_ensemble = True,
            alpha = 1.0,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Ablation: Adapter only (MLP Encoder with Classification loss, no Attention, no SNNL)
    elif method == 'aware_adapter_only':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=False
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 0.0, # Pure classification loss
            use_balance_sampling = False,
            use_ensemble = False,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Leave-One-Out: No Attention (Everything else enabled)
    elif method == 'aware_no_attn':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=False
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 1.0,
            use_balance_sampling = True,
            use_ensemble = True,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Leave-One-Out: No SNNL Metric Learning (Everything else enabled)
    elif method == 'aware_no_snnl':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=True
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 0.0, # Pure classification-based attention
            use_balance_sampling = True,
            use_ensemble = True,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Leave-One-Out: No Balance Sampling (Everything else enabled)
    elif method == 'aware_no_balance':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=True
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 1.0,
            use_balance_sampling = False,
            use_ensemble = True,
            pretrained_path=kwargs.get('pretrained_path', None)
        )
    # Leave-One-Out: No Ensemble (Everything else enabled)
    elif method == 'aware_no_ensemble':
        
        retriever = SNNLAttnRetriever(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=True
            ),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 1.0,
            use_balance_sampling = True,
            use_ensemble = False,
            pretrained_path=kwargs.get('pretrained_path', None)
        )

    # Imbalance-Aware AWARE (Weighted Sampling)
    elif method == 'aware_imb':
        from .aware_imb import ImbalanceAwareAWARE
        retriever = ImbalanceAwareAWARE(
            model = SNNLAttnEncoder(
                n_features=kwargs.get('n_features', 100),
                hidden_dim = kwargs.get('embedding_dim', 128),
                dropout = kwargs.get('dropout', 0.3),
                num_classes=kwargs.get('num_classes', 2),
                use_attention=True
            ),
            is_classification=kwargs.get('is_classification', True),
            field_to_use=field_to_use,
            distance='cosine',
            k=k,
            alpha = 1.0,
            use_balance_sampling = True,
            use_ensemble = True,
            use_cosine_loss=True,
            pretrained_path=kwargs.get('pretrained_path', None),
            save_attn_path=kwargs.get('save_attn_path', None),
            feature_names=kwargs.get('feature_names', None),
            # Imbalance-aware parameters
            candidate_multiplier=kwargs.get('candidate_multiplier', 5),
            class_weight_power=kwargs.get('class_weight_power', 1.0),
            distance_weight_power=kwargs.get('distance_weight_power', 1.0),
            deterministic=kwargs.get('deterministic', False),
            use_gpu=True#kwargs.get('use_gpu', False)
        )
        
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    
    return retriever