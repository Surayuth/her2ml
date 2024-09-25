import pickle
import argparse
import numpy as np
import polars as pl
from glob import glob
from scipy import stats
from pathlib import Path
from utils.prep import prep_case, mod_parallel

def read_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def get_lifespan(data):
    births = []
    deaths = []
    for k, v in data.items():
        b = v["birth"]
        d = v["death"]
        if d is not None:
            births.append(b)
            deaths.append(d)

    births = np.array(births)
    deaths = np.array(deaths)
    return births, deaths

def extract_feat(path, remove_noise):
    print(path)
    data = read_file(path)
    births, deaths = get_lifespan(data)

    mask = deaths < 255
    births = births[mask]
    deaths = deaths[mask]
    lifespan = deaths - births
    if remove_noise:
        lifespan = lifespan[lifespan > 5]
    # features
    # --------
    # min, max
    # mode,
    # q25, q50, q75
    # mean, std, skewness, kurtosis
    case_name = Path(path).parent.name

    if len(lifespan) > 0:
        features = [
            lifespan.min(), lifespan.max(), stats.mode(lifespan).mode,
            np.percentile(lifespan, q=25), np.percentile(lifespan, q=50),
            np.percentile(lifespan, q=75), lifespan.mean(), lifespan.std(),
            stats.kurtosis(lifespan), stats.skew(lifespan)
        ]
    else:
        features = [0] * 10
    return [path, case_name] + features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--remove_noise", action="store_true")
    parser.add_argument("--white_balance", action="store_true")
    parser.add_argument("--filter_bg", action="store_true")
    parser.add_argument("--src", default="./extracted_persist_features", type=str, help="source to the dataset")
    args = parser.parse_args()

    scale = args.scale
    filter_bg = args.filter_bg
    src = Path(args.src)
    remove_noise = args.remove_noise
    is_white_balance = args.white_balance

    paths = glob(str(src / f"white_balance_{is_white_balance}_filter_bg_{filter_bg}_scale_{scale}" / "*/*"))   
    data = mod_parallel(
        extract_feat,
        workers=1,
        inputs=paths,
        remove_noise=remove_noise,
    )

    cols = ["path", "case"] + \
        [
            "min", "max", "mode",
            "q25", "q50", "q75",
            "mean", "std", "skew", "kurtosis", 
        ]
    
    df = pl.DataFrame(
        data, schema=cols, orient="row"
    )
    df = prep_case(df)
    dst_file = (
        Path("extracted_features")
        / f"persist_feat_remove_noise_{remove_noise}_white_balance_{is_white_balance}_filter_bg_{filter_bg}_scale_{scale}.csv"
    )
    if not dst_file.parent.is_dir():
        dst_file.parent.mkdir(parents=True)
    df.fill_nan(0).write_csv(dst_file)
