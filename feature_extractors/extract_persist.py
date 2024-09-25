from glob import glob
from utils.prep import *
from utils.persist import *
from scipy.stats import mode
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

def get_dab(img):
    # gray macenko
    normalizer = NumpyMacenkoNormalizer()
    normalizer.fit(img)
    Inorm, H, DAB, C2 = normalizer.normalize(img)
    c = C2[1].reshape(img.shape[:-1])
    gray = np.round((c - c.min()) / (c.max() - c.min()) * 255).astype(np.uint8)
    return gray

def white_balance(img, q=99):
    norm_img = (img / np.percentile(img, axis=(0, 1), q=q)).clip(0, 1) * 255
    return np.round(norm_img).astype(np.uint8)

def extract_feat(path, scale, is_white_balance, filter_bg, dst):
    img = read_image(path, scale)
    gray = get_dab(img)
    if is_white_balance:
        gray = white_balance(get_dab(img))
    bg_mode = int(mode(gray.reshape(-1)).mode)

    if filter_bg:
        prep_gray = gray * (gray > int(bg_mode) * 1.)
    else:
        prep_gray = gray

    path = Path(path)
    case_name = path.parent.name
    dst_parent = Path(dst) / case_name
    dst_file = dst_parent / (Path(path).stem + ".pkl")

    if not dst_file.is_file():
        timestep = 0
        max_th = 255
        mask = prep_mask(prep_gray < max_th + 1)
        life_dict, labeled_img, max_cur_label = init_hist(mask)
        ths = np.arange(max_th)[::-1]
        for i in tqdm(range(len(ths))):
            th = ths[i]
            timestep += 1
            mask = prep_mask(prep_gray < th)
            life_dict, labeled_img, max_cur_label = update_hist(
                labeled_img, mask, 
                life_dict, max_cur_label, 
                timestep
                )
        
        if not dst_parent.is_dir():
            dst_parent.mkdir(parents=True)

        with open(dst_file, "wb") as f:
            pickle.dump(life_dict, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=12, type=int, help="default number of workers")
    parser.add_argument("--scale", default=0.25, type=float, help="resizing scale")
    parser.add_argument("--src", default="./Data_Chula", type=str, help="source to the dataset")
    parser.add_argument("--dst", default="./extracted_persist_features", type=str, help="dst to store features")
    parser.add_argument("--white_balance", action="store_true")
    parser.add_argument("--filter_bg", action="store_true")
    args = parser.parse_args()

    paths = glob(str(Path(args.src) / "*/*"))
    case_names = [Path(path).parent.name for path in paths]
    scale = args.scale
    filter_bg = args.filter_bg

    df = pl.DataFrame({
            "path": paths,
            "case": case_names
        })
    df = filter_case(df)
    filtered_paths = df.select("path").to_series().to_list()

    is_white_balance = args.white_balance
    mod_parallel(
        extract_feat,
        workers=12,
        inputs=filtered_paths,
        scale=scale,
        is_white_balance=is_white_balance,
        filter_bg=filter_bg,
        dst=Path(args.dst) / f"white_balance_{is_white_balance}_filter_bg_{filter_bg}_scale_{scale}"
    )
