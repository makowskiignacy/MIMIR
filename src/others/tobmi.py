# tobmi.py

import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


class TOBMIKNNImputer:
    """
    TOBMI-style kNN imputer for trans-omics block missing data,
    implemented with sklearn's KNeighborsRegressor.

    - Uses only training samples (train_samples) as donors.
    - Distance is computed in the concatenated feature space of
      the modalities present in the scenario.
    - Default metric is 'cosine', with distance-based weights.
    """

    def __init__(
        self,
        multi_omic_data: Dict[str, pd.DataFrame],
        train_samples: List[str],
        k: Optional[int] = None,
        metric: str = "cosine",
        scale: bool = False,
    ):
        """
        Args:
            multi_omic_data: dict(modality -> DataFrame [samples x features])
                             full dataset (all samples, all modalities).
            train_samples: list of sample IDs used as donors.
                           Only these samples are compared against scenario samples.
            k: number of neighbors. If None, k = floor(sqrt(N_train)).
            metric: distance metric for KNeighborsRegressor (default 'cosine').
            scale: if True, z-score concatenate features by donor stats
                   before distance computation.
        """
        self.multi_omic_data = multi_omic_data
        self.metric = metric
        self.scale = scale
        self.k = k

        # intersect train_samples with indices present in all modalities
        train_samples_set = set(train_samples)
        common_train = train_samples_set.copy()
        for df in multi_omic_data.values():
            common_train &= set(df.index)
        self.train_samples = sorted(common_train)

        if len(self.train_samples) == 0:
            raise ValueError("No training samples remain after intersecting with multi_omic_data indices.")

        # cache donor dataframes per modality (restricted to train_samples)
        self.donors_by_mod: Dict[str, pd.DataFrame] = {
            mod: df.loc[self.train_samples].copy()
            for mod, df in multi_omic_data.items()
        }

    def _build_feature_matrices(
        self,
        scenario_data: Dict[str, pd.DataFrame],
        modalities_present: List[str],
        scenario_samples: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build concatenated feature matrices for donors (X_donor)
        and scenario samples (X_rec) using only modalities_present.

        Returns:
            X_donor: [N_donor, D_concat]
            X_rec:   [N_rec,   D_concat]
        """
        donor_blocks = []
        recip_blocks = []

        for mod in modalities_present:
            if mod not in self.donors_by_mod:
                raise ValueError(f"Modality '{mod}' not found in donors_by_mod.")

            donors_df_full = self.donors_by_mod[mod]
            scenario_df_full = scenario_data[mod]

            donors_df = donors_df_full.loc[self.train_samples]
            common_cols = donors_df.columns.intersection(scenario_df_full.columns)
            if len(common_cols) == 0:
                raise ValueError(f"No overlapping features for modality '{mod}' between donors and scenario.")
            donors_df = donors_df[common_cols]
            recip_df = scenario_df_full.loc[scenario_samples, common_cols]

            # fill NaNs with donor column means
            col_means = donors_df.mean(axis=0)
            donors_df = donors_df.fillna(col_means)
            recip_df = recip_df.fillna(col_means)

            donor_blocks.append(donors_df.to_numpy(dtype=float))
            recip_blocks.append(recip_df.to_numpy(dtype=float))

        X_donor = np.concatenate(donor_blocks, axis=1)
        X_rec = np.concatenate(recip_blocks, axis=1)

        # optional z-score scaling based on donors
        if self.scale:
            mean = X_donor.mean(axis=0)
            std = X_donor.std(axis=0)
            std[std == 0.0] = 1.0
            X_donor = (X_donor - mean) / std
            X_rec = (X_rec - mean) / std

        return X_donor, X_rec

    def _prepare_target_matrix(self, target_mod: str) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare donor target matrix for one modality.

        Returns:
            Y_donor: [N_donor, D_target]
            target_cols: list of feature names
        """
        if target_mod not in self.donors_by_mod:
            raise ValueError(f"Target modality '{target_mod}' not found in donors_by_mod.")

        donors_target_df = self.donors_by_mod[target_mod].loc[self.train_samples]
        target_cols = list(donors_target_df.columns)

        # fill NaNs with column means
        col_means_y = donors_target_df.mean(axis=0)
        donors_target_df = donors_target_df.fillna(col_means_y)
        Y_donor = donors_target_df.to_numpy(dtype=float)

        return Y_donor, target_cols

    def impute_for_scenario(self, scenario_payload: dict) -> pd.DataFrame:
        """
        Impute the missing modality for a single scenario.

        scenario_payload should be a dict with:
            - "modalities_present": List[str]
            - "missing_modality": str
            - "samples": List[str]
            - "data": Dict[str, DataFrame]  (present modalities only)

        Returns:
            imputed_df: DataFrame [scenario_samples x target_features]
        """
        modalities_present: List[str] = scenario_payload["modalities_present"]
        target_mod: str = scenario_payload["missing_modality"]
        scenario_samples: List[str] = scenario_payload["samples"]
        scenario_data: Dict[str, pd.DataFrame] = scenario_payload["data"]

        # 1) feature matrices for donors and scenario samples
        X_donor, X_rec = self._build_feature_matrices(
            scenario_data=scenario_data,
            modalities_present=modalities_present,
            scenario_samples=scenario_samples,
        )

        n_donor = X_donor.shape[0]
        n_rec = X_rec.shape[0]

        if n_donor == 0 or n_rec == 0:
            raise ValueError("Empty donor or scenario matrix in impute_for_scenario.")

        # 2) donor targets for the missing modality
        Y_donor, target_cols = self._prepare_target_matrix(target_mod)

        if Y_donor.shape[0] != n_donor:
            # in this implementation we always use self.train_samples for both,
            # so shapes should match; just sanity check
            raise RuntimeError(
                f"Mismatch between X_donor rows ({n_donor}) and Y_donor rows ({Y_donor.shape[0]})."
            )

        # 3) choose k if not set
        if self.k is None:
            k_eff = int(np.sqrt(n_donor))
            k_eff = max(1, min(k_eff, n_donor))
        else:
            k_eff = max(1, min(self.k, n_donor))

        # 4) fit KNN regressor on donors and predict for scenario
        if self.metric == "mahalanobis":
            # Compute covariance from donor features
            # X_donor: [n_samples, n_features]
            cov = np.cov(X_donor, rowvar=False)
        
            # Regularize in case it's singular (common for high-dimensional omics)
            eps = 1e-6
            cov_reg = cov + eps * np.eye(cov.shape[0])
        
            # Compute inverse covariance
            try:
                VI = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                VI = np.linalg.pinv(cov_reg)
        
            knn = KNeighborsRegressor(
                n_neighbors=k_eff,
                metric="mahalanobis",
                metric_params={"VI": VI},
                weights="distance",
                algorithm="auto",
                n_jobs =-1
            )
        else:
            knn = KNeighborsRegressor(
                n_neighbors=k_eff,
                metric=self.metric,
                weights="distance",
                algorithm="auto",
                n_jobs=-1
            )
        knn.fit(X_donor, Y_donor)
        Y_pred = knn.predict(X_rec)  # [N_rec, D_target]

        imputed_df = pd.DataFrame(Y_pred, index=scenario_samples, columns=target_cols)
        return imputed_df


def impute_missing_modalities_for_scenario(
    multi_omic_data: Dict[str, pd.DataFrame],
    train_samples: List[str],
    scenario_pickle_path: str,
    k: Optional[int] = None,
    metric: str = "cosine",
    scale: bool = False,
) -> pd.DataFrame:
    """
    Convenience function:
    - loads one scenario pickle
    - instantiates TOBMIKNNImputer
    - returns imputed DataFrame

    Args:
        multi_omic_data: full multi-omic dataset (all modalities).
        train_samples: list of sample IDs to use as donors.
        scenario_pickle_path: path to scenario pickle.
        k: number of neighbors (None => sqrt(N_train)).
        metric: distance metric (default 'cosine').
        scale: whether to z-score features based on donors.

    Returns:
        imputed_df: DataFrame [scenario_samples x target_features]
    """
    with open(scenario_pickle_path, "rb") as f:
        scenario_payload = pickle.load(f)

    imputer = TOBMIKNNImputer(
        multi_omic_data=multi_omic_data,
        train_samples=train_samples,
        k=k,
        metric=metric,
        scale=scale,
    )
    return imputer.impute_for_scenario(scenario_payload)


def translate_from_scenario_dir(
    scenarios_dir: str,
    multi_omic_data: Dict[str, pd.DataFrame],
    train_samples: List[str],
    k: Optional[int] = None,
    metric: str = "cosine",
    scale: bool = False,
    save_pred_pickle_path: str = None  # where to save the predictions dict (optional)
) -> Dict[Tuple[Tuple[str, ...], str], pd.DataFrame]:
    """
    Loop over all scenario pickles in a directory and impute the missing modality
    for each scenario using TOBMI-style KNN (via KNeighborsRegressor).

    Returns:
        predictions dict:
          keys: (tuple(sorted_present_mods), target_mod)
          values: DataFrame [scenario_samples x target_features]
    """
    imputer = TOBMIKNNImputer(
        multi_omic_data=multi_omic_data,
        train_samples=train_samples,
        k=k,
        metric=metric,
        scale=scale,
    )

    predictions: Dict[Tuple[Tuple[str, ...], str], pd.DataFrame] = {}

    for fname in sorted(os.listdir(scenarios_dir)):
        if not fname.endswith(".pkl"):
            continue

        path = os.path.join(scenarios_dir, fname)
        with open(path, "rb") as f:
            scenario_payload = pickle.load(f)

        modalities_present: List[str] = scenario_payload["modalities_present"]
        target_mod: str = scenario_payload["missing_modality"]

        imputed_df = imputer.impute_for_scenario(scenario_payload)

        key = (tuple(sorted(modalities_present)), target_mod)
        predictions[key] = imputed_df

    # Optionally save all predictions to a pickle
    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"[Saved predictions] {save_pred_pickle_path}")

    return predictions
