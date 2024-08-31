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
from skimage.measure import euler_number, label
from utils.prep import read_image, mod_parallel, prep_case, NumpyMacenkoNormalizer


def extract_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0] / 255
    rb_mask = img[:, :, 0] > img[:, :, 2]
    bg_l = scipy.stats.mode(l.reshape(-1)).mode
    cont_mask = (np.abs(l - bg_l) / bg_l) 
    dab_cont_mask = cont_mask * rb_mask
    h_cont_mask = cont_mask * (~rb_mask)
    return dab_cont_mask, h_cont_mask

def count_hole(mask, min_area_hole=100):
    h, w = mask.shape
    analysis = cv2.connectedComponentsWithStats(((~mask) * 255).astype(np.uint8), 
                                                4, 
                                                cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 
    
    # Loop through each component 
    count = 0
    for i in range(1, totalLabels): 
        area = values[i, cv2.CC_STAT_AREA]   
        if (area > min_area_hole) and (area < h * w / 100): 
            count += 1
    return count

def count_cell(h_mask, min_area1=100, min_area2=200):
    """
    min_area = min area to count as cell. We use 2 values to get the range of number of cells
    min_area1: min area of cell 1
    min_area2: min area of cell 2
    """
    prep_h_mask = h_mask > 0.05
    min_areas = [min_area1, min_area2]
    h, w = prep_h_mask.shape
    analysis = cv2.connectedComponentsWithStats((prep_h_mask * 255).astype(np.uint8), 
                                                4, 
                                                cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 
    # Loop through each component 
    h_counts = [0, 0]
    for i in range(1, totalLabels): 
        area = values[i, cv2.CC_STAT_AREA]   
        for j, min_area in enumerate(min_areas):
            if (area > min_area) and (area < h * w / 100): 
                h_counts[j] += 1
    return h_counts

def extract_line(cont_mask, min_cont, min_hole, min_area_hole):
    cs = np.arange(10, 255)
    holes = np.zeros(len(cs))
    #prep_mask = np.round(cont_mask / (cont_mask.max() + 1e-8) * 255).astype(np.uint8)
    h, w = cont_mask.shape

    holes = []
    q_cont_mask = np.zeros(cont_mask.shape).astype(np.uint8)
    d = .005
    ths = np.arange(min_cont, 1., d)
    for i in range(len(ths[:-1])):
        th1 = ths[i]
        th2 = ths[i+1]
        holes.append(count_hole(cont_mask > th1, min_area_hole))
        q_cont_mask += (
            (cont_mask >= th1) & 
            (cont_mask < th2)
        ).astype(np.uint8)

    # lbp
    lbp = local_binary_pattern(q_cont_mask, P=8, R=1, method="uniform")
    lbp = lbp[cont_mask >= 0.1]
    hists = np.histogram(lbp, bins=10)[0]
    hists = (hists / (hists.sum() + 1e-8)).tolist()
    # haralick
    #print(q_cont_mask.max(), len(th_ranges))
    glcm = graycomatrix(
        q_cont_mask, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=len(ths), symmetric=True, normed=False
    ).astype(float)

    for i in range(4):
        glcm[0,:,0,i] = 0
        glcm[:,:,0,i] /= (glcm[:,:,0,i].sum() + 1e-8)

    contrast = graycoprops(glcm, "contrast").reshape(-1).mean()
    dissim = graycoprops(glcm, "dissimilarity").reshape(-1).mean()
    homo = graycoprops(glcm, "homogeneity").reshape(-1).mean()
    asm = graycoprops(glcm, "ASM").reshape(-1).mean()
    energy = graycoprops(glcm, "energy").reshape(-1).mean()
    corrs = graycoprops(glcm, "correlation").reshape(-1).mean()
    entropy = np.array([shannon_entropy(glcm[:,:,0,i]) for i in range(4)]).mean()
    hara_feats = [
        contrast, dissim, homo, asm,
        energy, corrs, entropy
    ]

    holes = np.array(holes)
    holes = holes[holes > min_hole]
    if len(holes) > 0:
        holes = holes[np.argmax(holes):]
        return [len(holes), holes.max(), h, w], hists, hara_feats
    else:
        return [0, 0, h, w], hists, hara_feats


def extract_feat(path, scale, min_cont, min_hole, min_area_hole, min_cell_area1, min_cell_area2):
    img = read_image(path, scale)
    cont_mask, h_cont_mask = extract_mask(img)
    n_cell1, n_cell2 = count_cell(h_cont_mask, min_cell_area1, min_cell_area2)
    line_feat, lbp_feat, hara_feat = extract_line(cont_mask, min_cont, min_hole, min_area_hole)
    n, h_max, h, w = line_feat
    lbp0, lbp1, lbp2, lbp3, lbp4, lbp5, lbp6, lbp7, lbp8, lbp9 = lbp_feat
    contrast, dissim, homo, asm, energy, corrs, entropy = hara_feat

    case_name = Path(path).parent.name
    return \
        path, case_name, \
        n, h_max, h, w, \
        lbp0, lbp1, lbp2, lbp3, lbp4, lbp5, lbp6, lbp7, lbp8, lbp9, \
        contrast, dissim, homo, asm, \
        energy, corrs, entropy, \
        n_cell1, n_cell2


if __name__ == "__main__":
    # python feature_extractors/extract_line_feat.py

    parser = argparse.ArgumentParser()
    # default args
    parser.add_argument("--workers", default=16, type=int, help="default number of workers")

    # specific args
    parser.add_argument("--src", default="./Data_Chula", type=str, help="source to the dataset")
    parser.add_argument("--dst", default="./extracted_features", type=str, help="dst to store features")
    parser.add_argument("--scale", default=0.5, type=float, help="scaling image")
    parser.add_argument("--min_hole", default=10, type=int, help="min number of holes")
    parser.add_argument("--min_cont", default=0.1, type=float, help="min contrast for DAB")
    parser.add_argument("--min_area_hole", default=25, type=int, help="min area hole")
    parser.add_argument("--min_cell_area1", default=25, type=int, help="min cell area 1")
    parser.add_argument("--min_cell_area2", default=64, type=int, help="min cell area 2")

    args = parser.parse_args()

    scale = args.scale
    min_hole = args.min_hole
    min_cont = args.min_cont
    min_cell_area1 = args.min_cell_area1
    min_cell_area2 = args.min_cell_area2
    min_area_hole = args.min_area_hole

    paths = glob(f"{args.src}/*/*") 
    data = mod_parallel(
        extract_feat,
        workers=args.workers,
        inputs=paths,
        scale=scale,
        min_hole=min_hole,
        min_cont=min_cont,
        min_cell_area1=min_cell_area1,
        min_cell_area2=min_cell_area2,
        min_area_hole=min_area_hole
    )
    df = pl.DataFrame(
        data, schema=[
            "path", "case", 
            "n", "h_max", "h", "w", 
            "lbp0", "lbp1", "lbp2", "lbp3", 
            "lbp4", "lbp5", "lbp6", "lbp7", 
            "lbp8", "lbp9", 
            "contrast", "dissim", "homo", "asm", 
            "energy", "corrs", "entropy", 
            f"n_cell_{min_cell_area1}", f"n_cell_{min_cell_area2}"
        ], orient="row"
    )
    df = prep_case(df)
    dst_file = (
        Path(args.dst)
        / f"linefeat_v4|scale_{scale}|minhole_{min_hole}|n_cell_{min_cell_area1}|n_cell_{min_cell_area2}|min_cont_{min_cont}|min_area_hole_{min_area_hole}.csv"
    )
    df.write_csv(dst_file)
