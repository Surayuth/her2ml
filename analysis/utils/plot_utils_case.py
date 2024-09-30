import numpy as np
import polars as pl
from scipy.stats import bootstrap

def agg_case(df, version="hard"):
    """
    version:
        - hard: mode and > 50 %
        - weak: mode
    """
    agg_df = df \
        .group_by("case") \
        .agg(
            pl.col("label").first(),
            pl.col("ihc_score").first(),
            (pl.col("final_pred") == 0).sum().alias("fpred0"),
            (pl.col("final_pred") == 1).sum().alias("fpred1"),
            (pl.col("final_pred") == -1).sum().alias("fpred-1"),
            pl.len().alias("count")
        ) \
        .with_columns(
            pl.when(
                (pl.col("fpred0") > pl.col("fpred1")) &
                (pl.col("fpred0") > pl.col("fpred-1"))
            ).then(pl.lit(0))
            .when(
                (pl.col("fpred1") > pl.col("fpred0")) &
                (pl.col("fpred1") > pl.col("fpred-1"))
            ).then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .alias("final_pred1"), # mode
            pl.when(
                (
                    (pl.col("fpred0") > pl.col("fpred1")) &
                    (pl.col("fpred0") > pl.col("fpred-1"))
                ) & 
                (
                    pl.col("fpred0") / pl.col("count") > 0.5
                ) 
            ).then(pl.lit(0))
            .when(
                (
                    (pl.col("fpred1") > pl.col("fpred0")) &
                    (pl.col("fpred1") > pl.col("fpred-1"))
                ) &
                (
                    pl.col("fpred1") / pl.col("count") > 0.5
                )
            ).then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .alias("final_pred2"), # mode
        ) 
    
    if version == "weak":
        selected_col = "final_pred1"
        agg_df = agg_df \
            .with_columns(
                pl.col(selected_col)
                .alias("final_pred")
            )
    elif version == "hard":
        selected_col = "final_pred2"
        agg_df = agg_df \
            .with_columns(
                pl.col(selected_col)
                .alias("final_pred")
            )
    agg_df = agg_df \
        .with_columns(
            pl.when(
                pl.col("final_pred") == -1
            ).then(pl.lit(2))
            .otherwise(pl.lit(1))
            .alias("pred_size")
        )
    return agg_df

def agg_case_average_ambi_ihc(df, version="hard"):
    agg_df = agg_case(df, version)
    ambis = []
    for j in range(5):
        fdf = agg_df \
            .filter(
                pl.col("ihc_score") == j
            )
        size = fdf.select("pred_size").mean().item()
        ambis.append(size - 1)
    return ambis

def agg_case_average_ambi_her2(df, version="hard"):
    agg_df = agg_case(df, version)
    ambis = []
    for j in range(2):
        fdf = agg_df \
            .filter(
                pl.col("label") == j
            )
        size = fdf.select("pred_size").mean().item()
        ambis.append(size - 1)
    return ambis

def agg_case_tpr_fpr(df, version="hard"):
    """
    for 2+ only
    """
    agg_df = agg_case(df, version)
    sub_df = agg_df.filter(pl.col("ihc_score").is_in([2, 3]))
    tp = len(sub_df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 1)))
    tn = len(sub_df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 0)))
    fp = len(sub_df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 0)))
    fn = len(sub_df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 1)))

    tot = tp + tn + fp + fn
    ambi = len(sub_df.filter(pl.col("final_pred") == -1)) / len(sub_df)
    sensitivity = tp / (tp + fn + 1e-8) # tnr
    specificity = tn / (tn + fp + 1e-8) # tpr
    return [sensitivity, specificity, ambi]

def agg_case_tpr_fpr_ovr(df, version="hard"):
    agg_df = agg_case(df, version)
    tp = len(agg_df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 1)))
    tn = len(agg_df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 0)))
    fp = len(agg_df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 0)))
    fn = len(agg_df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 1)))

    tot = tp + tn + fp + fn
    ambi = len(agg_df.filter(pl.col("final_pred") == -1)) / len(agg_df)
    sensitivity = tp / (tp + fn + 1e-8) # tnr
    specificity = tn / (tn + fp + 1e-8) # tpr
    return [sensitivity, specificity, ambi]

def agg_case_coverage_ihc(df, version="hard"):
    agg_df = agg_case(df, version)
    cover = (
        agg_df 
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

def agg_case_coverage_her2(df, version="hard"):
    agg_df = agg_case(df, version)
    cover = (
                agg_df 
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

def agg_case_final_pred(df, version="hard"):
    """
    2+ only
    """
    agg_df = agg_case(df, version)
    fpred = (
        agg_df \
            .filter(
                pl.col("ihc_score").is_in([2, 3])
            ) \
            .group_by("final_pred") \
            .agg(
                pl.len().alias("count")
            ) \
            .sort("final_pred") 
    )
    ambi = fpred.filter(pl.col("final_pred") == -1)
    neg = fpred.filter(pl.col("final_pred") == 0)
    pos = fpred.filter(pl.col("final_pred") == 1)

    if len(ambi) > 0:
        v_ambi = ambi.select("count").item()
    else:
        v_ambi = 0
    if len(neg) > 0:
        v_neg = neg.select("count").item()
    else:
        v_neg = 0
    if len(pos) > 0:
        v_pos = pos.select("count").item()
    else:
        v_pos = 0

    return [v_ambi, v_neg, v_pos]

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

            # handling degenerate cases
            if np.isnan(min_ci) | np.isnan(high_ci):
                min_ci = mean_ci
                high_ci = mean_ci

            heights[col]["mean"].append(high_ci)
            heights[col]["err_min"].append(mean_ci - min_ci)
            heights[col]["err_max"].append(high_ci - mean_ci)
    return heights