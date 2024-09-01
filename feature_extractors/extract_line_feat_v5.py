import cv2
import argparse
import numpy as np
import scipy.stats
import polars as pl
from glob import glob
from pathlib import Path
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from utils.prep import read_image, prep_case, mod_parallel


def extract_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0] / 255
    rb_mask = img[:, :, 0] > img[:, :, 2]
    bg_l = scipy.stats.mode(l.reshape(-1)).mode
    cont_mask = (np.abs(l - bg_l) / bg_l) 
    dab_cont_mask = cont_mask * rb_mask
    h_cont_mask = cont_mask * (~rb_mask)
    return dab_cont_mask, h_cont_mask

def extract_feat(path, level, min_cont=0.05):
    level = level - 1
    img = read_image(path)
    cont_mask, h_cont_mask = extract_mask(img)

    # calculate q_gray
    q_gray = np.zeros(cont_mask.shape).astype(np.uint8)
    d = (1 - min_cont) / level
    ths = np.arange(min_cont, 1. + 1E-8, d)
    for i in range(len(ths[:-1])):
        v = i + 1
        th1 = ths[i]
        th2 = ths[i+1]
        level_mask = (cont_mask >= th1) & (cont_mask < th2)
        q_gray += (level_mask * v).astype(np.uint8)

    # color feat
    color_feat = (q_gray * (q_gray > 0)).mean()

    # lbp
    lbp = local_binary_pattern(q_gray, P=8, R=1, method="uniform")    
    hists = np.histogram(lbp, bins=10)[0]
    hists = (hists / (hists.sum() + 1e-8)).tolist()
    lbp0, lbp1, lbp2, lbp3, lbp4, lbp5, lbp6, lbp7, lbp8, lbp9 = hists
    # glcm
    glcm = graycomatrix(
        q_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=level+1, symmetric=True, normed=False
    ).astype(float)

    contrast = graycoprops(glcm, "contrast").reshape(-1).mean()
    dissim = graycoprops(glcm, "dissimilarity").reshape(-1).mean()
    homo = graycoprops(glcm, "homogeneity").reshape(-1).mean()
    asm = graycoprops(glcm, "ASM").reshape(-1).mean()
    energy = graycoprops(glcm, "energy").reshape(-1).mean()
    corrs = graycoprops(glcm, "correlation").reshape(-1).mean()
    entropy = np.array([shannon_entropy(glcm[:,:,0,i]) for i in range(4)]).mean()

    case_name = Path(path).parent.name
    return \
        path, case_name, \
        color_feat, \
        lbp0, lbp1, lbp2, lbp3, lbp4, lbp5, lbp6, lbp7, lbp8, lbp9, \
        contrast, dissim, homo, asm, \
        energy, corrs, entropy, 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default args
    parser.add_argument("--workers", default=8, type=int, help="default number of workers")

    parser.add_argument("--level", default=8, type=int, help="quantized level")
    parser.add_argument("--min_cont", default=0.05, type=float, help="min cont")
    parser.add_argument("--src", default="./Data_Chula", type=str, help="source to the dataset")
    parser.add_argument("--dst", default="./extracted_features", type=str, help="dst to store features")

    args = parser.parse_args()

    level = args.level
    min_cont = args.min_cont

    paths = glob(f"{args.src}/*/*")
    data = mod_parallel(
        extract_feat,
        workers=args.workers,
        inputs=paths,
        level=level,
        min_cont=min_cont
    )
    df = pl.DataFrame(
        data, schema=[
            "path", "case", 
            "color_feat",
            "lbp0", "lbp1", "lbp2", "lbp3", 
            "lbp4", "lbp5", "lbp6", "lbp7", 
            "lbp8", "lbp9", 
            "contrast", "dissim", "homo", "asm", 
            "energy", "corrs", "entropy", 
        ], orient="row"
    )
    df = prep_case(df)
    dst_file = (
        Path(args.dst)
        / f"cont_feat|level_{level}_min_cont{min_cont}.csv"
    )
    df.write_csv(dst_file)
