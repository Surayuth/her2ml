from glob import glob
from utils.prep import *
from utils.persist import *
from scipy.stats import mode
import pickle
from pathlib import Path
from tqdm import tqdm

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

def extract_feat(path, dst):
    img = read_image(path)
    gray = white_balance(get_dab(img))
    bg_mode = int(mode(gray.reshape(-1)).mode)
    prep_gray = gray * (gray > int(bg_mode) * 1.)

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

    return 1
    

if __name__ == "__main__":
    paths = glob("Data_Chula/*/*")
    case_names = [Path(path).parent.name for path in paths]

    df = pl.DataFrame({
            "path": paths,
            "case": case_names
        })
    df = filter_case(df)
    filtered_paths = df.select("path").to_series().to_list()

    data = mod_parallel(
        extract_feat,
        workers=12,
        inputs=filtered_paths,
        dst="./extracted_persist_features/"
    )
