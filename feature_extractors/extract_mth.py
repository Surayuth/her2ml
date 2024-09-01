import cv2
import argparse
import numpy as np
import polars as pl
from glob import glob
from pathlib import Path
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from utils.prep import read_image, prep_case, mod_parallel
from utils.prep import NumpyMacenkoNormalizer, SWARM, read_image

def get_dab(img, level=3):
    # gray macenko
    normalizer = NumpyMacenkoNormalizer()
    normalizer.fit(img)
    Inorm, H, DAB = normalizer.normalize(img)
    gray = cv2.cvtColor(DAB, cv2.COLOR_RGB2GRAY)
    # mth
    swarm = SWARM(gray, D=level)
    swarm.run()
    mth = swarm.get_best()
    # get q image
    th1 = 0
    img_list = []
    for i, th in enumerate(mth + [256]):
        th2 = th
        mask = (th1 <= gray) & (gray < th2)
        img_list.append(mask.T)
        th1 = th
    q_gray = np.array(img_list[:3]).T 
    q_gray = (q_gray * np.array([3,2,1]).reshape(1,1,3)).sum(axis=-1)
    return gray, q_gray, mth

def extract_feat(path, level):
    img = read_image(path)
    gray, q_gray, mth = get_dab(img, level)

    # color feat
    color_feat = (q_gray * (q_gray > 0)).mean()
    # stain ratio
    tot = (q_gray > 0).sum()
    strong_ratio = (q_gray == 3).sum() / tot
    med_ratio = (q_gray == 2).sum() / tot
    weak_ratio = (q_gray == 1).sum() / tot
    stain_feat = [strong_ratio, med_ratio, weak_ratio]

    # lbp
    lbp = local_binary_pattern(q_gray, P=8, R=1, method="uniform")    
    hists = np.histogram(lbp, bins=10)[0]
    hists = (hists / (hists.sum() + 1e-8)).tolist()

    # glcm
    glcm = graycomatrix(
        q_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=level+1, symmetric=True, normed=True
    ).astype(float)

    contrast = graycoprops(glcm, "contrast").reshape(-1).mean()
    dissim = graycoprops(glcm, "dissimilarity").reshape(-1).mean()
    homo = graycoprops(glcm, "homogeneity").reshape(-1).mean()
    asm = graycoprops(glcm, "ASM").reshape(-1).mean()
    energy = graycoprops(glcm, "energy").reshape(-1).mean()
    corrs = graycoprops(glcm, "correlation").reshape(-1).mean()
    entropy = np.array([shannon_entropy(glcm[:,:,0,i]) for i in range(4)]).mean()
    hara_feat = [contrast, dissim, homo, asm, energy, corrs, entropy]

    case_name = Path(path).parent.name
    return [path, case_name] + [color_feat] + hists + hara_feat + mth + stain_feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default args
    parser.add_argument("--workers", default=8, type=int, help="default number of workers")

    parser.add_argument("--level", default=8, type=int, help="quantized level")
    parser.add_argument("--src", default="./Data_Chula", type=str, help="source to the dataset")
    parser.add_argument("--dst", default="./extracted_features", type=str, help="dst to store features")

    args = parser.parse_args()

    level = args.level

    paths = glob(f"{args.src}/*/*")
    data = mod_parallel(
        extract_feat,
        workers=args.workers,
        inputs=paths,
        level=level,
    )
    cols = ["path", "case"] + \
            ["color_feat"] + \
            [
                "lbp0", "lbp1", "lbp2",
                "lbp3", "lbp4", "lbp5", "lbp6",
                "lbp7", "lbp8", "lbp9"
            ] + \
            [
                "contrast", "dissim", "homo", "asm",
                "energy", "corrs", "entropy"
            ] + \
            [
                f"mth{i+1}" for i in range(level)
            ] + \
            [
                "strong_ratio", "med_ratio", "weak_ratio"
            ]

    df = pl.DataFrame(
        data, schema=cols, orient="row"
    )
    df = prep_case(df)
    dst_file = (
        Path(args.dst)
        / f"mth_feat|level_{level}.csv"
    )
    df.write_csv(dst_file)