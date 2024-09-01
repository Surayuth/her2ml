import polars as pl

def filter_case(df, min_count, max_count):
    min_count = 10
    max_count = 30
    selected_df = (
            df \
            .with_columns(
                pl.len().over("case")
                .alias("count")
            ) 
            .filter(
                pl.col("count") >= min_count
            ) 
            .with_columns(
                pl.min_horizontal(max_count, pl.col("count"))
                .alias("cap_max")
            ) 
            .with_columns(
                pl.arange(1, pl.len() + 1).over("case")
                .alias("case_idx")
            ) 
            .filter(
                pl.col("case_idx") <= pl.col("cap_max")
            ) 
        )
    return selected_df
