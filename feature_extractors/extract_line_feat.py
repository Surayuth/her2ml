import cv2
import argparse
import numpy as np
import scipy.stats
import polars as pl
from glob import glob
from pathlib import Path
from skimage.measure import euler_number, label
from utils.prep import read_image, mod_parallel, prep_case

def extract_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = lab[:,:,0] / 255
    rb_mask = img[:,:,0] > img[:,:,2]
    bg_l = scipy.stats.mode(l.reshape(-1)).mode
    cont_mask = (np.abs(l - bg_l) / bg_l) * rb_mask
    return cont_mask


def extract_line(cont_mask, min_hole, max_hole, max_res):
    cs = np.arange(0.15, 1, .01)
    holes = np.zeros(len(cs))
    for i, c in enumerate(cs):
        cont_dab = cont_mask > c
        e8 = euler_number(cont_dab, connectivity=2)
        object_nb_8 = label(cont_dab, connectivity=2).max()
        holes_nb_8 = object_nb_8 - e8
        holes[i] = holes_nb_8

    cond = (holes > min_hole) & (holes < max_hole)
    x = cs[cond]
    y = -np.log(holes[cond])
    if len(x) > 1:
        for deg in np.arange(1, 100):
            params, res, _, _,_ = np.polyfit(x, y, deg=deg, full=True)
            if len(res) > 0:
                if res[0] < max_res:
                    if deg == 1:
                        if params[0] > 0:
                            return len(x), deg, params[0], (x, y)
                    else:
                        return len(x), deg, -1., (x, y)
            else:
                return len(x), deg, params[0], (x, y)
    else:
        return 0, 0, -1., ([0.], [0.])

def extract_feat(path, scale, min_hole, max_hole, max_res):
    img = read_image(path, scale)
    cont_mask = extract_mask(img)
    n, deg, param_v, (x, y) = extract_line(cont_mask, min_hole, max_hole, max_res)
    case_name = Path(path).parent.name
    return path, case_name, n, deg, param_v, np.mean(y)

if __name__ == "__main__":
    # python feature_extractors/extract_line_feat.py

    parser = argparse.ArgumentParser()
    # default args
    parser.add_argument("--workers", default=8, type=int, help="default number of workers")

    # specific args
    parser.add_argument("--src", default="./Data_Chula", type=str, help="source to the dataset")
    parser.add_argument("--dst", default="./extracted_features", type=str, help="dst to store features")
    parser.add_argument("--scale", default=0.25, type=float, help="scaling image")
    parser.add_argument("--min_hole", default=10, type=int, help="min number of holes")
    parser.add_argument("--max_hole", default=1000, type=int, help="max number of holes")
    parser.add_argument("--max_res", default=0.25, type=float, help="max residual")

    args = parser.parse_args()

    scale = args.scale
    min_hole=args.min_hole
    max_hole=args.max_hole
    max_res=args.max_res

    paths = glob(f"{args.src}/*/*")
    data = mod_parallel(
        extract_feat,
        workers=args.workers,
        inputs=paths,
        scale=scale,
        min_hole=min_hole,
        max_hole=max_hole,
        max_res=max_res
    )
    df = pl.DataFrame(
        data,
        schema=[
            "path", "case", "n", 
            "deg", "param_v", "mean_y"
        ],
        orient="row"
    )
    df = prep_case(df)
    dst_file = Path(args.dst) / f"linefeat|scale_{scale}|minhole_{min_hole}|maxhole_{max_hole}|maxres_{max_res}.csv"
    df.write_csv(dst_file)
    