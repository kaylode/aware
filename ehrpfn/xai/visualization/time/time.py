import polars as pl
import numpy as np
import plotly.graph_objects as go
"""

MODEL.MODEL_NAME ┆ mean_inference_time ┆ std_inference_time ┆ mean_training_time ┆ std_training_time │
│ ---              ┆ ---                 ┆ ---                ┆ ---                ┆ ---               │
│ str              ┆ f64                 ┆ f64                ┆ f64                ┆ f64               │
╞══════════════════╪═════════════════════╪════════════════════╪════════════════════╪═══════════════════╡
│ beta             ┆ 11.614732           ┆ 6.585111           ┆ 59476.58913        ┆ 31793.564661      │
│ catboost         ┆ 0.031229            ┆ 0.00587            ┆ 98.8242            ┆ 15.513927         │
│ excelformer      ┆ 23.780073           ┆ 2.729028           ┆ 9712.583659        ┆ 1047.439515       │
│ knn              ┆ 450.210325          ┆ 370.362476         ┆ 1082.974697        ┆ 1105.215274       │
│ lightgbm         ┆ 0.071403            ┆ 0.014342           ┆ 57.211255          ┆ 20.991493 
"""

def _write_or_fallback(fig, path_pdf):
    # requires kaleido: pip install -U kaleido
    fig.write_image(path_pdf)
    print(f"Saved PDF to: {path_pdf}")

def main():
    df = pl.read_csv("/home/mpham/workspace/source/ehrpfn/data/wandb_export_2025-10-18T12_time.csv")

    df = df.select([
        pl.col("MODEL.MODEL_NAME"),
        pl.col("dataset_name"),
        pl.col("task_name"),
        pl.col("training_time"),
        pl.col("inference_time"),
    ])
    print(df.head())

    df = df.filter(pl.col('task_name').is_in(['WILL_READMIT_WITHIN_30_DAYS', 'EPISODE_CPE_INFECTION_FLAGGED']))

    df = df.group_by(['MODEL.MODEL_NAME']).agg([
        pl.mean('inference_time').alias('mean_inference_time'),
        pl.std('inference_time').alias('std_inference_time'),
        pl.mean('training_time').alias('mean_training_time'),
        pl.std('training_time').alias('std_training_time'),
    ]).sort(['MODEL.MODEL_NAME'])
    print(df.head())

    # convert to pandas for plotting
    pdf = df.rename({"MODEL.MODEL_NAME": "model"}).to_pandas()

    models = pdf['model'].astype(str).tolist()
    train_mean = pdf['mean_training_time'].values
    train_std = pdf['std_training_time'].values
    inf_mean = pdf['mean_inference_time'].values
    inf_std = pdf['std_inference_time'].values

    # layout sizing
    pixel_width = max(800, int(len(models) * 120))
    height = 600

    # linear-scale grouped bar chart
    fig = go.Figure(data=[
        go.Bar(name='Training',
               x=models,
               y=train_mean,
               error_y=dict(type='data', array=train_std, visible=True),
               marker_color='steelblue'),
        go.Bar(name='Inference',
               x=models,
               y=inf_mean,
               error_y=dict(type='data', array=inf_std, visible=True),
               marker_color='darkorange'),
    ])
    fig.update_layout(
        barmode='group',
        title='Model training vs inference time (means ± std)',
        xaxis_tickangle=-45,
        yaxis_title='Time (seconds)',
        width=pixel_width,
        height=height,
        margin=dict(l=60, r=40, t=80, b=160)
    )

    out_linear = "/home/mpham/workspace/source/ehrpfn/ehrpfn/xai/visualization/time/time_comparison_linear.png"
    _write_or_fallback(fig, out_linear)

    # log-scale grouped bar chart
    fig_log = go.Figure(data=[
        go.Bar(name='Training',
               x=models,
               y=train_mean,
               error_y=dict(type='data', array=train_std, visible=True),
               marker_color='steelblue'),
        go.Bar(name='Inference',
               x=models,
               y=inf_mean,
               error_y=dict(type='data', array=inf_std, visible=True),
               marker_color='darkorange'),
    ])
    fig_log.update_layout(
        barmode='group',
        title='Model training vs inference time (log scale, means ± std)',
        xaxis_tickangle=-45,
        yaxis_title='Time (seconds, log scale)',
        yaxis_type='log',
        width=pixel_width,
        height=height,
        margin=dict(l=60, r=40, t=80, b=160)
    )

    out_log = "/home/mpham/workspace/source/ehrpfn/ehrpfn/xai/visualization/time/time_comparison_log.png"
    _write_or_fallback(fig_log, out_log)

    stored_df = pl.read_csv("/home/mpham/workspace/source/ehrpfn/ehrpfn/xai/visualization/elo/averaged_ranks2.csv")
    global_ranks = stored_df.mean().to_dict()
    global_stds = stored_df.std().to_dict()
    breakpoint()
    # perf_name
    # perf


    # normalize performance to [0,1]
    perf_min, perf_max = float(perf.min()), float(perf.max())
    perf_norm = (perf - perf_min) / (perf_max - perf_min + 1e-9)

    # map normalized perf to marker diameters (pixels)
    min_d, max_d = 8, 48
    sizes = min_d + perf_norm * (max_d - min_d)

    # compute ranking (1 = best)
    order = np.argsort(-perf)  # descending (higher perf = better)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(perf) + 1)

    scat_fig = go.Figure(
        data=go.Scatter(
            x=train_mean,
            y=inf_mean,
            text=models,
            mode='markers+text',
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                size=sizes,
                color=perf_norm,               # color encodes performance too
                colorscale='Viridis',
                colorbar=dict(title=perf_name),
                line=dict(width=1, color='darkgray')
            )
        )
    )
    scat_fig.update_layout(
        title='Training vs Inference Time per Model (marker size = method performance)',
        xaxis_title='Mean Training Time (s)',
        yaxis_title='Mean Inference Time (s)',
        yaxis_type='log',
        xaxis_type='log',
        width=800,
        height=600
    )
    out_log = "/home/mpham/workspace/source/ehrpfn/ehrpfn/xai/visualization/time/scatter_comparison_log.png"
    _write_or_fallback(scat_fig, out_log)

if __name__ == "__main__":
    main()