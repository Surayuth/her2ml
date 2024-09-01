import os

# List of scripts to run sequentially
scripts = [
    "python ./train/train_mth.py --path extracted_features/mth_feat\|level_3.csv --features color --n_jobs 16 --trials 50 --repeat 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat\|level_3.csv --features color --n_jobs 16 --trials 50 --repeat 50",
    "python ./train/train_mth.py --path extracted_features/mth_feat\|level_3.csv --features color --n_jobs 16 --trials 50 --repeat 50",
]

for script in scripts:
    print(f"Running {script}...")
    result = os.system(script)
    
    # Print the output and error (if any) from the script
    print(result.stdout)
    if result.stderr:
        print(f"Error occurred while running {script}:")
        print(result.stderr)

    # Check if the script ran successfully
    if result.returncode != 0:
        print(f"Script {script} failed with return code {result.returncode}")
        break
    else:
        print(f"Script {script} completed successfully.\n")

print("All scripts have been executed.")