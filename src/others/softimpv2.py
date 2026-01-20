# softimp.py

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from fancyimpute import SoftImpute as FancySoftImpute

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pickle


class SoftImputer:
    """
    Multi-omic low-rank imputer using fancyimpute's SoftImpute algorithm.

    Notes:
      - Unsupervised matrix completion: fit_transform operates directly on the
        (corrupted) matrix with NaNs.
      - We skip BiScaler because your inputs are already standardized.
      - We concatenate modalities across columns, run SoftImpute, then split
        back into per-modality DataFrames.
    """

    def __init__(
        self,
        J: int = 20,                      # rank cap (max_rank)
        thresh: float = 1e-3,            # convergence_threshold
        lambda_: Optional[float] = None, # shrinkage_value (None -> fancyimpute heuristic)
        maxit: int = 100,                # max_iters
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_power_iterations: int = 1,
        init_fill_method: str = "zero",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """
        Args:
            J: maximum rank used in truncated SVD each iteration (maps to max_rank).
            thresh: convergence threshold on relative change of missing entries
                    (maps to convergence_threshold).
            lambda_: shrinkage amount applied to singular values each iteration
                     (maps to shrinkage_value). If None, fancyimpute uses
                     (max singular value of init)/50 heuristic.
            maxit: maximum number of iterations (maps to max_iters).
            random_state: forwarded when possible (randomized_svd uses it internally
                          but fancyimpute doesn't expose; we store for API parity).
            verbose: print per-iter observed MAE + rank.
            n_power_iterations: randomized SVD power iterations.
            init_fill_method: how to fill missing at init ("zero" is default).
            min_value/max_value: optional clipping bounds on reconstructed values.
        """
        self.J = J
        self.thresh = thresh
        self.lambda_ = lambda_
        self.maxit = maxit
        self.random_state = random_state
        self.verbose = verbose

        # fancyimpute SoftImpute (skip BiScaler; assume standardized)
        self.soft_imputer = FancySoftImpute(
            shrinkage_value=lambda_,
            convergence_threshold=thresh,
            max_iters=maxit,
            max_rank=J,
            n_power_iterations=n_power_iterations,
            init_fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value,
            normalizer=None,
            verbose=verbose,
        )

        # Set in fit()
        self.modalities_: Optional[List[str]] = None
        self.feature_counts_: Optional[Dict[str, int]] = None
        self.samples_: Optional[List[str]] = None
        self.columns_per_mod_: Optional[Dict[str, pd.Index]] = None

        # Stored reconstruction
        self._X_imp: Optional[np.ndarray] = None

    def _concat_modalities(
        self,
        data: Dict[str, pd.DataFrame],
        samples: List[str],
        modalities: List[str],
    ) -> np.ndarray:
        blocks = []
        for mod in modalities:
            df_mod = data[mod].loc[samples]
            blocks.append(df_mod.values.astype(np.float64, copy=False))
        return np.concatenate(blocks, axis=1)

    def fit(
        self,
        data_corrupted: Dict[str, pd.DataFrame],
        samples: Optional[List[str]] = None,
        use_modalities: Optional[List[str]] = None,
    ):
        if use_modalities is None:
            use_modalities = list(data_corrupted.keys())

        missing_mods = [m for m in use_modalities if m not in data_corrupted]
        if missing_mods:
            raise ValueError(f"Modalities not in data_corrupted: {missing_mods}")

        self.modalities_ = list(use_modalities)

        if samples is None:
            samples = list(data_corrupted[self.modalities_[0]].index)
        self.samples_ = list(samples)

        self.feature_counts_ = {}
        self.columns_per_mod_ = {}
        for mod in self.modalities_:
            df_mod = data_corrupted[mod]
            self.feature_counts_[mod] = df_mod.shape[1]
            self.columns_per_mod_[mod] = df_mod.columns

        X = self._concat_modalities(data_corrupted, self.samples_, self.modalities_)

        # fancyimpute does everything in fit_transform; keep a fit() API for consistency
        self._X_imp = self.soft_imputer.fit_transform(X)
        return self

    def transform(self) -> Dict[str, pd.DataFrame]:
        if (
            self.modalities_ is None
            or self.feature_counts_ is None
            or self.samples_ is None
            or self.columns_per_mod_ is None
            or self._X_imp is None
        ):
            raise RuntimeError("SoftImputer has not been fit yet.")

        X_imp = self._X_imp
        pred_dfs: Dict[str, pd.DataFrame] = {}

        start = 0
        for mod in self.modalities_:
            n_feats = self.feature_counts_[mod]
            end = start + n_feats

            block = X_imp[:, start:end]
            cols = self.columns_per_mod_[mod]
            pred_dfs[mod] = pd.DataFrame(block, index=self.samples_, columns=cols)

            start = end

        return pred_dfs

    def fit_transform(
        self,
        data_corrupted: Dict[str, pd.DataFrame],
        samples: Optional[List[str]] = None,
        use_modalities: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        self.fit(data_corrupted=data_corrupted, samples=samples, use_modalities=use_modalities)
        return self.transform()



def _eval_on_masked_positions(
    pred_dfs: Dict[str, pd.DataFrame],
    corrupt_dfs: Dict[str, pd.DataFrame],
    truth_dfs: Dict[str, pd.DataFrame],
    modalities: List[str],
    samples: List[str],
    metric: str = "rmse",
) -> Dict[str, Any]:
    """
    Evaluate predictions ONLY on artificially masked entries:
      - corrupt is NaN
      - truth is NOT NaN

    Returns dict with overall metric and per-modality metrics.
    """
    per_mod = {}
    all_true = []
    all_pred = []

    for mod in modalities:
        # align
        truth = truth_dfs[mod].loc[samples]
        corrupt = corrupt_dfs[mod].loc[samples]
        pred = pred_dfs[mod].loc[samples]

        mask = corrupt.isna() & ~truth.isna()
        if mask.values.sum() == 0:
            per_mod[mod] = {"n": 0, metric: np.nan}
            continue

        y_true = truth.values[mask.values].astype(float)
        y_pred = pred.values[mask.values].astype(float)

        all_true.append(y_true)
        all_pred.append(y_pred)

        if metric == "rmse":
            score = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        elif metric == "mae":
            score = float(np.mean(np.abs(y_pred - y_true)))
        elif metric == "pearson":
            # handle degenerate
            if y_true.size < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
                score = np.nan
            else:
                score = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        per_mod[mod] = {"n": int(mask.values.sum()), metric: score}

    # aggregate
    if len(all_true) == 0:
        overall = np.nan
        n_total = 0
    else:
        y_true_all = np.concatenate(all_true) if len(all_true) > 1 else all_true[0]
        y_pred_all = np.concatenate(all_pred) if len(all_pred) > 1 else all_pred[0]
        n_total = int(y_true_all.size)

        if metric == "rmse":
            overall = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
        elif metric == "mae":
            overall = float(np.mean(np.abs(y_pred_all - y_true_all)))
        elif metric == "pearson":
            if y_true_all.size < 2 or np.std(y_true_all) < 1e-12 or np.std(y_pred_all) < 1e-12:
                overall = np.nan
            else:
                overall = float(np.corrcoef(y_true_all, y_pred_all)[0, 1])

    return {"overall": overall, "n_total": n_total, "per_modality": per_mod}


def impute_from_corrupt_soft_valtest(
    val_corrupt_pickle_path: str,
    test_corrupt_pickle_path: str,
    multi_omic_data: Dict[str, pd.DataFrame],
    val_samples: List[str],
    test_samples: List[str],
    use_modalities: Optional[List[str]] = None,
    J_grid: Optional[List[int]] = None,
    lambda_grid: Optional[List[float]] = None,
    thresh: float = 1e-5,
    maxit: int = 200,
    random_state: Optional[int] = 0,
    verbose: bool = False,
    select_metric: str = "rmse",   # "rmse" (min), "mae" (min), or "pearson" (max)
    save_test_pred_pickle_path: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Tune SoftImpute hyperparameters on a VAL corrupted pickle (using the original
    data as truth at the masked positions), then run on TEST corrupted pickle.

    Returns:
      test_pred_dfs, info_dict (best params + val scores + full grid results)
    """
    if J_grid is None:
        J_grid = [64, 128, 256]
    if lambda_grid is None:
        lambda_grid = [0.0, 1.0, 3.0, 10.0]

    # Load corrupted dicts
    with open(val_corrupt_pickle_path, "rb") as f:
        val_corrupt_dfs: Dict[str, pd.DataFrame] = pickle.load(f)
    with open(test_corrupt_pickle_path, "rb") as f:
        test_corrupt_dfs: Dict[str, pd.DataFrame] = pickle.load(f)

    # Modalities
    if use_modalities is None:
        use_modalities = [m for m in val_corrupt_dfs.keys() if m in multi_omic_data]
    if len(use_modalities) == 0:
        raise ValueError("No overlapping modalities between corrupt pickles and multi_omic_data.")

    # sanity: ensure modalities exist in test too
    missing_in_test = [m for m in use_modalities if m not in test_corrupt_dfs]
    if missing_in_test:
        raise ValueError(f"Modalities missing in test corrupt pickle: {missing_in_test}")

    # Decide optimization direction
    maximize = (select_metric == "pearson")

    grid_results = []
    best = None

    # Subset dicts to modalities for speed/consistency
    val_subset = {m: val_corrupt_dfs[m] for m in use_modalities}
    test_subset = {m: test_corrupt_dfs[m] for m in use_modalities}

    # ---- Grid search on VAL ----
    for J in J_grid:
        for lam in lambda_grid:
            soft_imp = SoftImputer(
                J=J,
                thresh=thresh,
                lambda_=lam,
                maxit=maxit,
                random_state=random_state,
                verbose=verbose,
            )
            val_pred = soft_imp.fit_transform(
                data_corrupted=val_subset,
                samples=val_samples,
                use_modalities=use_modalities,
            )

            val_score = _eval_on_masked_positions(
                pred_dfs=val_pred,
                corrupt_dfs=val_subset,
                truth_dfs=multi_omic_data,
                modalities=use_modalities,
                samples=val_samples,
                metric=select_metric,
            )

            entry = {"J": J, "lambda_": lam, "val": val_score}
            grid_results.append(entry)

            # pick best
            score = val_score["overall"]
            if np.isnan(score):
                continue
            if best is None:
                best = entry
            else:
                if maximize:
                    if score > best["val"]["overall"]:
                        best = entry
                else:
                    if score < best["val"]["overall"]:
                        best = entry

    if best is None:
        raise RuntimeError("All validation scores were NaN. Check that your val corruption actually masks observed entries.")

    best_J = best["J"]
    best_lambda = best["lambda_"]

    # ---- Fit on TEST with best params ----
    soft_imp_best = SoftImputer(
        J=best_J,
        thresh=thresh,
        lambda_=best_lambda,
        maxit=maxit,
        random_state=random_state,
        verbose=verbose,
    )
    test_pred = soft_imp_best.fit_transform(
        data_corrupted=test_subset,
        samples=test_samples,
        use_modalities=use_modalities,
    )

    # Optionally save
    if save_test_pred_pickle_path is not None:
        with open(save_test_pred_pickle_path, "wb") as f:
            pickle.dump(test_pred, f)
        print(f"[Saved SoftImpute TEST predictions] {save_test_pred_pickle_path}")

    info = {
        "best_params": {"J": best_J, "lambda_": best_lambda, "thresh": thresh, "maxit": maxit},
        "best_val": best["val"],
        "grid_results": grid_results,
        "select_metric": select_metric,
        "use_modalities": use_modalities,
    }

    return test_pred, info




def impute_from_corrupt_soft(
    corrupt_pickle_path: str,
    multi_omic_data: Dict[str, pd.DataFrame],   # kept for API consistency; only used to pick modalities if None
    samples: List[str],
    use_modalities: Optional[List[str]] = None,
    J: int = 128,
    lambda_: Optional[float] = None,            # fancyimpute shrinkage_value; None => heuristic
    thresh: float = 1e-3,                       # fancyimpute convergence_threshold
    maxit: int = 200,                           # fancyimpute max_iters
    random_state: Optional[int] = 0,            # isn't actually used
    verbose: bool = False,
    save_pred_pickle_path: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Impute NaNs in a single corrupted pickle using SoftImputer (fancyimpute SoftImpute backend),
    with user-specified hyperparameters (no val tuning).

    Returns:
      pred_dfs, info_dict (params + basic metadata)
    """
    # 1) Load corrupted dict
    with open(corrupt_pickle_path, "rb") as f:
        corrupt_dfs: Dict[str, pd.DataFrame] = pickle.load(f)

    # 2) Decide modalities
    if use_modalities is None:
        use_modalities = [m for m in corrupt_dfs.keys() if m in multi_omic_data]
    if len(use_modalities) == 0:
        raise ValueError("No overlapping modalities between corrupt pickle and multi_omic_data.")

    missing_mods = [m for m in use_modalities if m not in corrupt_dfs]
    if missing_mods:
        raise ValueError(f"Modalities missing in corrupt pickle: {missing_mods}")

    # 3) Subset dict
    corrupt_subset = {m: corrupt_dfs[m] for m in use_modalities}

    # 4) Run SoftImpute
    soft_imp = SoftImputer(
        J=J,
        thresh=thresh,
        lambda_=lambda_,
        maxit=maxit,
        random_state=random_state,
        verbose=verbose,
    )
    pred_dfs = soft_imp.fit_transform(
        data_corrupted=corrupt_subset,
        samples=samples,
        use_modalities=use_modalities,
    )

    # 5) Optionally save predictions
    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(pred_dfs, f)
        print(f"[Saved SoftImpute predictions] {save_pred_pickle_path}")

    info = {
        "params": {"J": J, "lambda_": lambda_, "thresh": thresh, "maxit": maxit, "random_state": random_state},
        "use_modalities": use_modalities,
        "corrupt_pickle_path": corrupt_pickle_path,
        "n_samples": len(samples),
    }

    return pred_dfs, info

