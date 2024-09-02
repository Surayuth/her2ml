import argparse
import polars as pl
import optuna
import numpy as np
from pathlib import Path
from utils import filter_case
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, df, selected_features, cv, inner_case):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1E-3, 0.1, log=True)
    }
    th = trial.suggest_float("th", 0.0, 1.0)

    inner_skf = StratifiedKFold(n_splits=cv-1, random_state=r, shuffle=True)
    inner_splits = inner_skf.split(inner_case.select("case"), inner_case.select("label"))

    val_f1s = []
    for j, (train_idx, val_idx) in enumerate(inner_splits):
        train_case = inner_case[train_idx].select("case") 
        val_case = inner_case[val_idx].select("case")
        train_df = df.filter(pl.col("case").is_in(train_case)).select(*selected_features, "label")
        val_df = df.filter(pl.col("case").is_in(val_case)).select(*selected_features, "label")
        
        X_train = train_df.drop("label").to_numpy()
        y_train = train_df.select("label").to_numpy().reshape(-1)

        X_val = val_df.drop("label").to_numpy()
        y_val = val_df.select("label").to_numpy().reshape(-1)

        model = GradientBoostingClassifier(**params, random_state=0)
        model.fit(X_train, y_train)

        val_pred = (model.predict_proba(X_val)[:, 1] > th) * 1
        val_f1 = f1_score(y_val, val_pred)
        val_f1s.append(val_f1)
    return np.mean(val_f1s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path to the features")
    parser.add_argument("--dst", type=str, default="./results", help="path to results")
    parser.add_argument("--min_img", type=int, default=10, help="min image/case")
    parser.add_argument("--max_img", type=int, default=30, help="max image/case")
    parser.add_argument("--trials", type=int, default=10, help="no. of trials")
    parser.add_argument("--n_jobs", type=int, default=4, help="no. of trials")
    parser.add_argument("--r_min", type=int, default=0, help="repeat min")
    parser.add_argument("--r_max", type=int, default=19, help="repeat max")

    parser.add_argument("--cv", type=int, default=4, help="cross validation")

    parser.add_argument(
        "--features", 
        nargs='+',  # '+' means at least one argument is expected
        type=str,   # Specifies that each item in the list should be an integer
        help="color, lbp, hara"
    )
    
    args = parser.parse_args()
    
    path = args.path
    dst = args.dst
    min_img = args.min_img
    max_img = args.max_img
    features = sorted(args.features)
    trials = args.trials
    cv = args.cv
    n_jobs = args.n_jobs
    r_min = args.r_min
    r_max = args.r_max

    dst_root = Path(dst) / Path(path).stem
    if not dst_root.is_dir():
        dst_root.mkdir(parents=True)

    if not (dst_root / "_".join(features)).is_dir():
        (dst_root / "_".join(features)).mkdir(parents=True)


    selected_features = []
    for f in features:
        if f not in ["color", "lbp", "hara"]:
            raise Exception("Please specify correct features!") 

    if "color" in features:
        selected_features += ["color_feat"]
    if "lbp" in features:
        selected_features += [f"lbp{i}" for i in range(10)]
    if "hara" in features:
        selected_features += [
            "contrast", "dissim", "homo", "asm",
            "energy", "corrs", "entropy"
        ]

    df = filter_case(pl.read_csv(path), min_img, max_img) \
        .select("path", "case", "ihc_score", "label", *selected_features)
    case_df = df.group_by("case").agg(pl.col("label").min())

    rows = []
    for r in range(r_min, r_max+1, 1):
        skf = StratifiedKFold(n_splits=cv, random_state=r, shuffle=True)
        splits = skf.split(case_df.select("case"), case_df.select("label"))
        for i, (inner_idx, outer_idx) in enumerate(splits):
            inner_case = case_df[inner_idx].select("case", "label")
            outer_case = case_df[outer_idx].select("case")

            test_df = df.filter(pl.col("case").is_in(outer_case)).select(*selected_features, "case", "path", "ihc_score", "label")
            X_test = test_df.drop("case", "path", "ihc_score", "label").to_numpy()
            y_test = test_df.select("label").to_numpy().reshape(-1)

            f_objective = partial(
                objective, df=df, 
                selected_features=selected_features, 
                cv=cv, inner_case=inner_case,
                )
            
            study = optuna.create_study(direction="maximize")
            study.optimize(f_objective, n_trials=trials, n_jobs=n_jobs)
            best_params = study.best_params
            th = best_params["th"]
            del best_params["th"]

            X_train = df.filter(pl.col("case").is_in(inner_case.drop("label"))).select(selected_features).to_numpy()
            y_train = df.filter(pl.col("case").is_in(inner_case.drop("label"))).select("label").to_numpy().reshape(-1)

            model = GradientBoostingClassifier(**best_params, random_state=0)
            model.fit(X_train, y_train)

            test_prob = model.predict_proba(X_test)[:, 1]
            test_pred = (test_prob > th) * 1

            # patch
            acc = accuracy_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred)
            recall = recall_score(y_test, test_pred)
            auc = roc_auc_score(y_test, test_pred)
            cm = confusion_matrix(y_test, test_pred)
            tn, fp, fn, tp = cm.ravel()

            test_df = test_df \
                .with_columns(
                    pl.lit(th).alias("th"),
                    pl.Series(test_prob).alias("prob"),
                    pl.Series(test_pred).alias("pred")
                )
            dst_file = dst_root / "_".join(features) / f"{r}_{i}_{"_".join(features)}.csv"
            print(dst_file)
            test_df.write_csv(
                dst_file
            )

            # agg_test_df = test_df \
            #     .group_by("case") \
            #     .agg(
            #         pl.col("prob").mean(),
            #         pl.col("label").min(),
            #     ) \
            #     .with_columns(
            #         pl.when((pl.col("prob") > 0.5)).then(1)
            #         .otherwise(0)
            #         .alias("pred")
            #     )

            # case_y_test = agg_test_df.select("label").to_numpy().reshape(-1)
            # case_test_pred = agg_test_df.select("pred").to_numpy().reshape(-1)

            # # case
            # case_acc = accuracy_score(case_y_test, case_test_pred)
            # case_f1 = f1_score(case_y_test, case_test_pred)
            # case_precision = precision_score(case_y_test, case_test_pred)
            # case_recall = recall_score(case_y_test, case_test_pred)
            # case_auc = roc_auc_score(case_y_test, case_test_pred)
            # case_cm = confusion_matrix(case_y_test, case_test_pred)
            # case_tn, case_fp, case_fn, case_tp = case_cm.ravel()


            # patch_result = [acc, f1, precision, recall, auc, tn, fp, fn, tp]
            # case_result = [
            #     case_acc, case_f1, case_precision, case_recall, 
            #     case_auc, case_tn, case_fp, case_fn, case_tp]
            
            # rows.append(
            #     [r, i] + [th] + patch_result + case_result
            # )

    # result_df = pl.DataFrame(
    #     np.array(rows), 
    #     schema=[
    #         "repeat", "cv", "th",
    #         "acc", "f1", "precision", "recall", "auc", "tn", "fp", "fn", "tp",
    #         "case_acc", "case_f1", "case_precision", "case_recall", "case_auc", "case_tn", "case_fp", "case_fn", "case_tp",
    #     ]
    # )

    # result_df.write_csv(dst_root)

        