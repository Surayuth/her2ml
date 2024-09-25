import optuna
import argparse
import numpy as np
import polars as pl
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def filter_case(df, min_count, max_count):
    min_count = 10
    max_count = 30
    selected_df = (
            df \
            .with_columns(
                pl.len().over("case")
                .alias("count")
            ) 
            .filter(
                pl.col("count") >= min_count
            ) 
            .with_columns(
                pl.min_horizontal(max_count, pl.col("count"))
                .alias("cap_max")
            ) 
            .with_columns(
                pl.arange(1, pl.len() + 1).over("case")
                .alias("case_idx")
            ) 
            .filter(
                pl.col("case_idx") <= pl.col("cap_max")
            ) 
        ) \
        .with_columns(
            pl.when(pl.col("ihc_score") == "0").then(pl.lit(0))
            .when(pl.col("ihc_score") == "1+").then(pl.lit(1))
            .when(pl.col("ihc_score") == "2-").then(pl.lit(2))
            .when(pl.col("ihc_score") == "2+").then(pl.lit(3))
            .when(pl.col("ihc_score") == "3+").then(pl.lit(4))
            .otherwise(None)
            .alias("ihc_score")
        )
    return selected_df

def objective(trial, df, base_model, inner_case, random_state, n_estimators):
    if base_model == GradientBoostingClassifier:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, n_estimators, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
            "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1E-3, 0.1, log=True)
        }
    elif base_model == RandomForestClassifier:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, n_estimators, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
            "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
        }


    inner_skf = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)
    inner_splits = inner_skf.split(inner_case.select("case"), inner_case.select("ihc_score"))
    val_aucs = []
    for j, (train_idx, val_idx) in enumerate(inner_splits):
        inner_train_case = inner_case[train_idx].select("case", "label", "ihc_score")
        inner_val_case = inner_case[val_idx].select("case", "label", "ihc_score")

        inner_train_df = df \
            .filter(
                pl.col("case") 
                .is_in(inner_train_case.select("case"))
            ) \
            .drop("count", "cap_max", "case_idx")

        inner_val_df = df \
            .filter(
                pl.col("case") 
                .is_in(inner_val_case.select("case"))
            ) \
            .drop("count", "cap_max", "case_idx")

        inner_X_train = inner_train_df.drop("case", "path", "ihc_score", "label").to_numpy()
        inner_y_train = inner_train_df.select("label").to_numpy().reshape(-1)

        inner_X_val = inner_val_df.drop("case", "path", "ihc_score", "label").to_numpy()
        inner_y_val = inner_val_df.select("label").to_numpy().reshape(-1)

        model = base_model(**params, random_state=0)
        model.fit(inner_X_train, inner_y_train)

        val_prob = model.predict_proba(inner_X_val)[:, 1]
        val_auc = roc_auc_score(inner_y_val, val_prob)
        val_aucs.append(val_auc)
    return np.mean(val_aucs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path to the features")
    parser.add_argument("--dst", type=str, default="./results_cp", help="path to results")
    parser.add_argument("--min_img", type=int, default=10, help="min image/case")
    parser.add_argument("--max_img", type=int, default=30, help="max image/case")
    parser.add_argument("--trials", type=int, default=50, help="no. of trials")
    parser.add_argument("--n_jobs", type=int, default=12, help="no. of trials")
    parser.add_argument("--r_min", type=int, default=0, help="repeat min")
    parser.add_argument("--r_max", type=int, default=19, help="repeat max")
    parser.add_argument("--n_estimators", type=int, default=200, help="no. of estimators")
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--dry_run", action="store_true", help="train model without hyperparameter tuning")
    parser.add_argument(
        '--alphas', nargs='+',
        type=float, default=[0.01, 0.05, 0.1],
        help="A list of float numbers"
    )

    args = parser.parse_args()
    path = args.path
    dst = args.dst
    min_img = args.min_img
    max_img = args.max_img
    trials = args.trials
    n_jobs = args.n_jobs
    r_min = args.r_min
    r_max = args.r_max
    model_name = args.model
    n_estimators = args.n_estimators
    alphas = args.alphas
    dry_run = args.dry_run

    dst_root = Path(dst) / (
            Path(path).stem + 
            f"_trial_{trials}" + 
            f"_n_est_{n_estimators if not dry_run else 100}" + 
            f"_model_{model_name}" +
            f"_dryrun_ {dry_run}"
        ) 
    if not dst_root.is_dir():
        dst_root.mkdir(parents=True)

    df = filter_case(pl.read_csv(path), min_img, max_img)
    case_df = df.group_by("case") \
        .agg(
            pl.col("label").min(), 
            pl.col("ihc_score").first())

    for r in range(r_min, r_max + 1):
        calib_skf = StratifiedKFold(n_splits=5, random_state=r, shuffle=True)
        calib_splits = calib_skf.split(case_df.select("case"), case_df.select("ihc_score"))

        # 1) split 1/5 folds for calib
        for train_idx, calib_idx in calib_splits:
            train_case = case_df[train_idx].select("case", "label", "ihc_score")
            calib_case = case_df[calib_idx].select("case", "label", "ihc_score")

            calib_df = df \
                .filter(
                    pl.col("case")
                    .is_in(calib_case.select("case"))
                ) \
                .drop("count", "cap_max", "case_idx")
            X_calib = calib_df.drop("case", "path", "ihc_score", "label").to_numpy()
            y_calib = calib_df.select("label").to_numpy().reshape(-1)
            ihc_calib = calib_df.select("ihc_score") 
            break

        # 2) used the rest of folds 4/5 for train(2)/val(1)/test(1)
        train_skf = StratifiedKFold(n_splits=4, random_state=r, shuffle=True)
        train_splits = train_skf.split(train_case.select("case"), train_case.select("ihc_score"))   

        for i, (inner_idx, outer_idx) in enumerate(train_splits):
            inner_case = train_case[inner_idx].select("case", "label", "ihc_score")
            outer_case = train_case[outer_idx].select("case", "label", "ihc_score")

            # hyperparameters tuning
            if model_name == "rf":
                base_model = RandomForestClassifier
            elif model_name == "gbt":
                base_model = GradientBoostingClassifier

            if dry_run:
                best_params = {
                    "n_estimators": 100
                }
            else:
                f_objective = partial(
                        objective, df=df, 
                        base_model=base_model,
                        inner_case=inner_case,
                        random_state=r,
                        n_estimators=n_estimators
                    )
                study = optuna.create_study(direction="maximize")
                study.optimize(f_objective, n_trials=trials, n_jobs=n_jobs)
                best_params = study.best_params

            model = base_model(**best_params, random_state=0)

            train_df = df \
                .filter(
                    pl.col("case") 
                    .is_in(train_case.select("case"))
                ) \
                .drop("count", "cap_max", "case_idx")

            X_train = train_df.drop("case", "path", "ihc_score", "label").to_numpy()
            y_train = train_df.select("label").to_numpy().reshape(-1)

            test_df = df \
                .filter(
                    pl.col("case") 
                    .is_in(outer_case.select("case"))
                ) \
                .drop("count", "cap_max", "case_idx")
            X_test = test_df.drop("case", "path", "ihc_score", "label").to_numpy()
            y_test = test_df.select("label").to_numpy().reshape(-1)

            # print(i, len(X_calib), len(X_train), len(X_test), len(X_calib) + len(X_train) + len(X_test))
            model.fit(X_train, y_train)

            # calib (conformal prediction)
            for alpha in alphas:
                prob_calib = model.predict_proba(X_calib)
                
                scores0 = 1 - prob_calib[:,0][y_calib == 0]
                scores1 = 1 - prob_calib[:,1][y_calib == 1]
                n_calib = len(prob_calib)
                q_level = np.ceil((n_calib+1) * (1 - alpha)) / n_calib
                qhat0 = np.quantile(scores0, q_level, method='higher')
                qhat1 = np.quantile(scores1, q_level, method='higher')
                # test
                prob_test = model.predict_proba(X_test)
                prob0 = prob_test[:, 0]
                prob1 = prob_test[:, 1]
                preds0 = (prob0 >= 1 - qhat0) * 1
                preds1 = (prob1 >= 1 - qhat1) * 1
                
                pred_df = pl.DataFrame({
                        "path": test_df.select("path"),
                        "case": test_df.select("case"),
                        "random_state": r,
                        "alpha": alpha,
                        "fold": i,
                        "q_level": q_level,
                        "qhat0": qhat0,
                        "qhat1": qhat1,
                        "prob0": prob0,
                        "prob1": prob1,
                        "pred0": preds0,
                        "pred1": preds1,
                        "label": y_test,
                        "ihc_score": test_df.select("ihc_score"),
                    }) \
                    .with_columns(
                        (pl.col("pred0") + pl.col("pred1"))
                        .alias("pred_size")
                    ) \
                    .with_columns(
                        pl.when(
                            (pl.col("pred0") == 1) & (pl.col("pred1") == 0)
                        ).then(pl.lit(0))
                        .when(
                            (pl.col("pred0") == 0) & (pl.col("pred1") == 1)
                        ).then(pl.lit(1))
                        .otherwise(pl.lit(-1))
                        .alias("final_pred")
                    ) 
                
                dst_exp = dst_root / f"{r}_{i}" 
                if not dst_exp.is_dir():
                    dst_exp.mkdir(parents=True)
                dst_file = dst_exp / f"{r}_{i}_{alpha}_result.csv"
                pred_df.write_csv(dst_file)

