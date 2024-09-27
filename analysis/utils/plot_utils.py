import numpy as np
import polars as pl
from scipy.stats import bootstrap

def agg_patch_average_ambi_ihc(df):
    ambis = []
    for j in range(5):
        fdf = df \
            .filter(
                pl.col("ihc_score") == j
            )
        size = fdf.select("pred_size").mean().item()
        ambis.append(size - 1)
    return ambis

def agg_patch_average_ambi_her2(df):
    ambis = []
    for j in range(2):
        fdf = df \
            .filter(
                pl.col("label") == j
            )
        size = fdf.select("pred_size").mean().item()
        ambis.append(size - 1)
    return ambis

def agg_patch_tpr_fpr(df):
    sub_df = df.filter(pl.col("ihc_score").is_in([2, 3]))
    tp = len(sub_df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 1)))
    tn = len(sub_df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 0)))
    fp = len(sub_df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 0)))
    fn = len(sub_df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 1)))

    tot = tp + tn + fp + fn
    ambi = len(sub_df.filter(pl.col("pred_size") == 2)) / len(sub_df)
    specificity = tp / (tp + fn + 1e-8) # recall
    sensitivity = tn / (tn + fp + 1e-8) # tnr
    return [specificity, sensitivity, ambi]

def agg_patch_coverage_her2(df):
    cover = (
                df 
                .with_columns(
                    pl.when(
                        (pl.col("final_pred") == pl.col("label")) |
                        (pl.col("final_pred") == -1)
                    ).then(pl.lit(1))
                    .otherwise(0)
                    .alias("is_cover")
                ) 
                .group_by("label") 
                .agg(
                    pl.col("is_cover").mean()
                ) 
                .sort("label").select("is_cover")
                .to_numpy().reshape(-1).tolist()
            )
    return cover

def agg_patch_coverage_ihc(df):
    cover = (
        df 
        .with_columns(
            pl.when(
                (pl.col("final_pred") == pl.col("label")) |
                (pl.col("final_pred") == -1)
            ).then(pl.lit(1))
            .otherwise(0)
            .alias("is_cover")
        ) 
        .group_by("ihc_score") 
        .agg(
            pl.col("is_cover").mean()
        ) 
        .sort("ihc_score")
        .select("is_cover")
        .to_numpy().reshape(-1).tolist()
    )
    return cover


def agg_heights(root, cv, r_min, r_max, alphas, agg_func, col_names):
    rows = []
    for r in range(r_min, r_max + 1):
        for f in range(cv):
            for alpha in alphas:
                file_path = root / f"{r}_{f}" / f"{r}_{f}_{alpha}_result.csv"

                df = pl.read_csv(file_path)
                values = agg_func(df)

                row = [r, alpha] + values
                rows.append(row)
    
    schema = ["r", "alpha"] + col_names

    result_df = pl.DataFrame(
        rows, schema=schema,
        orient="row"
    )

    agg_result = result_df \
        .group_by("r", "alpha") \
        .agg(
            [
                pl.col(col).mean()
                for col in col_names
            ]
        ) \
        .sort("r", "alpha")

    heights = {
        col: {
            "mean": [],
            "err_min": [],
            "err_max": []
        } 
        for col in col_names
    }

    for alpha in alphas:
        arr_stats = agg_result \
            .filter(pl.col("alpha") == alpha) \
            .select(col_names) \
            .to_numpy()

        for i, col in enumerate(col_names):
            bi = bootstrap((arr_stats)[:,i].reshape(1,-1), statistic=np.mean)
            ci = bi.confidence_interval
            min_ci = ci.low * 100
            high_ci = ci.high * 100
            mean_ci = bi.bootstrap_distribution.mean() * 100

            heights[col]["mean"].append(high_ci)
            heights[col]["err_min"].append(mean_ci - min_ci)
            heights[col]["err_max"].append(high_ci - mean_ci)
    return heights

