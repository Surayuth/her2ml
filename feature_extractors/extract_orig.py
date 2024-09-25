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
from utils.prep import NumpyMacenkoNormalizer, SWARM, read_image, white_balance
from scipy.stats import mode

def get_dab(img):
    # gray macenko
    try:
        normalizer = NumpyMacenkoNormalizer()
        normalizer.fit(img)
        Inorm, H, DAB, C2 = normalizer.normalize(img)
        c = C2[1].reshape(img.shape[:-1])
        gray = 255 - cv2.cvtColor(DAB, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        gray = np.zeros(img.shape[:-1]).astype(np.uint8)
    _,q_gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return gray, q_gray

def extract_feat(path, orig_level, scale, is_white_balance):
    img = read_image(path, scale)
    # white balance
    if is_white_balance:
        img = white_balance(img)

    gray, q_gray = get_dab(img)
    mask = q_gray > 0
    # color feat
    color_feat = (gray * q_gray).mean()
    area_ratio = (q_gray > 0).mean()

    # lbp
    g_values = gray[mask]
    if len(g_values) == 0:
        min_g = 0
        max_g = 0
    else:
        min_g = g_values.min()
        max_g = g_values.max()
    norm_gray = (np.round((gray - min_g) / (max_g - min_g + 1e-8) * mask * (orig_level - 1)) + 1 * mask).astype(np.uint8)

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")    
    # select only stained region
    selected_lbp = lbp[mask]
    hists = np.histogram(selected_lbp, bins=10)[0]
    hists = (hists / (hists.sum() + 1e-8)).tolist()
    # glcm
    glcm = graycomatrix(
        norm_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=orig_level+2, symmetric=True, normed=False
    ).astype(float)
    # set the background = 0
    for i in range(4):
        glcm[0,:,0,i] = 0
        glcm[:,:,0,i] = glcm[:,:,0,i] / (glcm[:,:,0,i].sum() + 1e-8)

    contrast = graycoprops(glcm, "contrast").reshape(-1).mean()
    dissim = graycoprops(glcm, "dissimilarity").reshape(-1).mean()
    homo = graycoprops(glcm, "homogeneity").reshape(-1).mean()
    asm = graycoprops(glcm, "ASM").reshape(-1).mean()
    energy = graycoprops(glcm, "energy").reshape(-1).mean()
    corrs = graycoprops(glcm, "correlation").reshape(-1).mean()
    entropy = np.array([shannon_entropy(glcm[:,:,0,i]) for i in range(4)]).mean()
    hara_feat = [contrast, dissim, homo, asm, energy, corrs, entropy]

    case_name = Path(path).parent.name
    return [path, case_name] + [color_feat, area_ratio] + hists + hara_feat

if __name__ == "__main__":
    """
    1. segment: new method
    2. feature calculating: new method
    """
    parser = argparse.ArgumentParser()
    # default args
    parser.add_argument("--workers", default=12, type=int, help="default number of workers")
    parser.add_argument("--scale", default=0.25, type=float, help="resizing scale")
    parser.add_argument("--level", default=16, type=int, help="quantized level")
    parser.add_argument("--src", default="./Data_Chula", type=str, help="source to the dataset")
    parser.add_argument("--dst", default="./extracted_features", type=str, help="dst to store features")
    parser.add_argument("--white_balance", action="store_true")
    args = parser.parse_args()

    level = args.level
    is_white_balance = args.white_balance
    scale = args.scale
    paths = glob(f"{args.src}/*/*")
    data = mod_parallel(
        extract_feat,
        workers=args.workers,
        inputs=paths,
        orig_level=level,
        scale=scale,
        is_white_balance=is_white_balance
    )
    cols = ["path", "case"] + \
            ["color_feat", "area_ratio"] + \
            [
                "lbp0", "lbp1", "lbp2",
                "lbp3", "lbp4", "lbp5", "lbp6",
                "lbp7", "lbp8", "lbp9"
            ] + \
            [
                "contrast", "dissim", "homo", "asm",
                "energy", "corrs", "entropy"
            ] 

    df = pl.DataFrame(
        data, schema=cols, orient="row"
    )
    df = prep_case(df)
    dst_file = (
        Path(args.dst)
        / f"orig_feat_level_{level}_white_balance_{is_white_balance}_scale_{scale}.csv"
    )
    if not dst_file.parent.is_dir():
        dst_file.parent.mkdir(parents=True)
    df.write_csv(dst_file)