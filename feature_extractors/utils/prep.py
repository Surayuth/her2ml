import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import polars as pl
from pillow_heif import open_heif
from joblib import Parallel, delayed
from functools import partial


def read_image(path, scale=1 / 4):
    try:
        if Path(path).suffix.lower() == ".heic":
            img = open_heif(path, convert_hdr_to_8bit=True)
        else:
            img = Image.open(path)
        rgb = np.asarray(img)
        H, W = rgb.shape[:-1]
        new_H = int(H * scale)
        new_W = int(W * scale)
    except:
        print(path)
        raise
    return cv2.resize(rgb, (new_W, new_H))


def prep_case(df):
    df = df.with_columns(
        pl.when(pl.col("case").str.contains("1+", literal=True))
        .then(pl.lit("1+"))
        .when(pl.col("case").str.contains("score 0 case 2", literal=True))
        .then(pl.lit("0"))
        .when(pl.col("case").str.contains("3+ D+ 01", literal=True))
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("2+ DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("3+", literal=True))
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("28 Jun HER2 IHC negative", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ D+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ DISH -", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("HER2 0", literal=True))
        .then(pl.lit("0"))
        .when(pl.col("case").str.contains("HER2 score 1", literal=True))
        .then(pl.lit("1+"))
        .when(pl.col("case").str.contains("2+ DISH-", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ Dish -", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ DISH +", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ Dish+", literal=True))
        .then(pl.lit("2+"))
        .when(
            pl.col("case").str.contains(
                "13 Sep HER2 different brightness", literal=True
            )
        )
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("2+DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("HER2 neg case 01", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2 + DISH +", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("score 0", literal=True))
        .then(pl.lit("0"))
        .otherwise(None)
        .alias("ihc_score")
    ).with_columns(
        pl.when(pl.col("ihc_score").is_in(["0", "1+", "2-"]))
        .then(pl.lit(0))
        .otherwise(1)
        .alias("label")
    )
    return df


class ProgressParallel(Parallel):
    def __init__(self, n_total_tasks=None, **kwargs):
        super().__init__(**kwargs)
        self.n_total_tasks = n_total_tasks

    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.n_total_tasks:
            self._pbar.total = self.n_total_tasks
        else:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def mod_parallel(func, workers, inputs, **kwargs):
    data = ProgressParallel(n_jobs=workers, n_total_tasks=len(inputs))(
        delayed(partial(func, **kwargs))(input) for input in inputs
    )
    return data
