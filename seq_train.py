import os

# List of scripts to run sequentially
scripts = [
    #"python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features color --n_jobs 10 --trials 40 --r_min 0 --r_max 50",

    # "python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features color --n_jobs 10 --trials 40 --r_min 0 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features lbp --n_jobs 10 --trials 40 --r_min 0 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features hara --n_jobs 10 --trials 40 --r_min 0 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features color lbp --n_jobs 10 --trials 40 --r_min 0 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features color hara --n_jobs 10 --trials 40 --r_min 0 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features lbp hara --n_jobs 10 --trials 40 --r_min 0 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat_v2\\|level_3.csv --features color lbp hara --n_jobs 10 --trials 40 --r_min 0 --r_max 50",

    # "python ./train/train_mth.py --path extracted_features/baseline_feat\\|level_32.csv --features color --n_jobs 10 --trials 40 --r_min 20 --r_max 50",
    # "python ./train/train_mth.py --path extracted_features/baseline_feat\\|level_32.csv --features lbp --n_jobs 10 --trials 40 --r_min 20 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/baseline_feat\\|level_32.csv --features hara --n_jobs 10 --trials 40 --r_min 20 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/baseline_feat\\|level_32.csv --features color lbp --n_jobs 10 --trials 40 --r_min 20 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/baseline_feat\\|level_32.csv --features color hara --n_jobs 10 --trials 40 --r_min 20 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/baseline_feat\\|level_32.csv --features lbp hara --n_jobs 10 --trials 40 --r_min 20 --r_max 50",
    "python ./train/train_mth.py --path extracted_features/baseline_feat\\|level_32.csv --features color lbp hara --n_jobs 10 --trials 40 --r_min 20 --r_max 50",
]

for script in scripts:
    print(f"Running {script}...")
    result = os.system(script)

print("All scripts have been executed.")