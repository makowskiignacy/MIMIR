# knnimp.py

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pickle

from sklearn.impute import KNNImputer


class KNN_Imputer:
    """
    Multi-omic KNN-based value imputer.

    Internally:
      - Concatenates selected modalities (features stacked along columns).
      - Fits sklearn.impute.KNNImputer on train samples.
      - Transforms (imputes) arbitrary sample sets with the same feature layout.
      - Splits the imputed matrix back into per-modality DataFrames.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        **kwargs,
    ):
        """
        Args:
            n_neighbors, weights, metric: passed directly to sklearn.impute.KNNImputer
            **kwargs: any additional keyword args for KNNImputer
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            **kwargs,
        )

        # Will be set in fit()
        self.modalities_: Optional[List[str]] = None
        self.feature_counts_: Optional[Dict[str, int]] = None
        self.feature_columns_: Optional[pd.MultiIndex] = None

    def _concat_modalities(
        self,
        data: Dict[str, pd.DataFrame],
        samples: List[str],
        modalities: List[str],
    ) -> np.ndarray:
        """
        Concatenate modalities (columns stacked) for given samples and modality list.

        Assumes each data[mod] has all samples and columns aligned as desired.
        """
        blocks = []
        for mod in modalities:
            df_mod = data[mod].loc[samples]
            blocks.append(df_mod.values.astype(float))

        X = np.concatenate(blocks, axis=1)
        return X

    def fit(
        self,
        multi_omic_data: Dict[str, pd.DataFrame],
        train_samples: List[str],
        use_modalities: Optional[List[str]] = None,
    ):
        """
        Fit the underlying KNNImputer using the training samples.

        Args:
            multi_omic_data: {mod: DataFrame} full dataset
            train_samples: list of sample IDs to use for fitting
            use_modalities: which modalities to use. If None, use all keys in multi_omic_data.
        """
        if use_modalities is None:
            use_modalities = list(multi_omic_data.keys())

        # Sanity check: all modalities must be present
        missing = [m for m in use_modalities if m not in multi_omic_data]
        if missing:
            raise ValueError(f"Modalities not in multi_omic_data: {missing}")

        self.modalities_ = list(use_modalities)

        # Concatenate training data
        X_train = self._concat_modalities(multi_omic_data, train_samples, self.modalities_)

        # Track feature counts per modality for splitting
        self.feature_counts_ = {
            mod: multi_omic_data[mod].shape[1] for mod in self.modalities_
        }

        # Build MultiIndex columns (modality, feature) for bookkeeping
        col_tuples = []
        for mod in self.modalities_:
            cols = list(multi_omic_data[mod].columns)
            col_tuples.extend((mod, c) for c in cols)
        self.feature_columns_ = pd.MultiIndex.from_tuples(
            col_tuples, names=["modality", "feature"]
        )

        # Fit sklearn KNNImputer
        self.imputer.fit(X_train)

    def transform(
        self,
        data_corrupted: Dict[str, pd.DataFrame],
        samples: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Impute missing values in `data_corrupted`.

        Args:
            data_corrupted: {mod: DataFrame} with NaNs where values are missing/corrupted.
                            Must include the same modalities & features as used in fit().
            samples: optional explicit sample order. If None, uses the index of the
                     first modality in self.modalities_.

        Returns:
            pred_dfs: {mod: DataFrame} of imputed values, same shape as input.
        """
        if self.modalities_ is None or self.feature_counts_ is None:
            raise RuntimeError("KNN_Imputer has not been fit yet.")

        # Decide sample ordering
        if samples is None:
            # Use index from the first modality
            first_mod = self.modalities_[0]
            samples = list(data_corrupted[first_mod].index)

        # Sanity: ensure all required modalities are present
        missing = [m for m in self.modalities_ if m not in data_corrupted]
        if missing:
            raise ValueError(f"Missing modalities in data_corrupted: {missing}")

        # Concatenate evaluation data in the same modality order
        X_eval = self._concat_modalities(data_corrupted, samples, self.modalities_)

        # Impute
        X_imputed = self.imputer.transform(X_eval)  # shape [N, total_features]

        # Split back into per-modality DataFrames
        pred_dfs: Dict[str, pd.DataFrame] = {}

        start = 0
        for mod in self.modalities_:
            n_feats = self.feature_counts_[mod]
            end = start + n_feats

            block = X_imputed[:, start:end]
            cols = data_corrupted[mod].columns
            df_imp = pd.DataFrame(block, index=samples, columns=cols)
            pred_dfs[mod] = df_imp

            start = end

        return pred_dfs

    def fit_transform(
        self,
        multi_omic_data: Dict[str, pd.DataFrame],
        train_samples: List[str],
        use_modalities: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience method: fit on train_samples and then transform them.

        Returns:
            {mod: DataFrame} of imputed values for train_samples.
        """
        self.fit(multi_omic_data, train_samples, use_modalities=use_modalities)
        # Build data_corrupted dict from the same samples (no extra NaNs injected here)
        data_corrupted = {
            mod: multi_omic_data[mod].loc[train_samples].copy()
            for mod in self.modalities_
        }
        return self.transform(data_corrupted, samples=train_samples)


def impute_values_from_corrupt(
    corrupt_pickle_path: str,
    multi_omic_data: Dict[str, pd.DataFrame],
    train_samples: List[str],
    use_modalities: Optional[List[str]] = None,
    n_neighbors: int = None,
    weights: str = "uniform",
    metric: str = "nan_euclidean",
    save_pred_pickle_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load a corrupted data pickle (e.g., from mask_and_predict) and impute its NaNs
    using a KNN_Imputer fit on the train samples.

    Args:
        corrupt_pickle_path: path to pickle with {mod: DataFrame} of corrupted data.
                             Typically this is the `corrupt_dfs` dict from mask_and_predict.
        multi_omic_data: {mod: DataFrame} full dataset (used to fit the KNN model).
        train_samples: sample IDs to use for fitting the KNN imputer.
        use_modalities: modalities to use. If None, uses intersection of
                        corrupt modalities and multi_omic_data keys.
        n_neighbors, weights, metric: KNNImputer hyperparameters.
        save_pred_pickle_path: if not None, save predictions dict here.

    Returns:
        pred_dfs: {mod: DataFrame} of imputed values for the corrupted data.
    """
    # 1) Load corrupted dict
    with open(corrupt_pickle_path, "rb") as f:
        corrupt_dfs: Dict[str, pd.DataFrame] = pickle.load(f)

    # 2) Decide which modalities to use
    if use_modalities is None:
        use_modalities = [m for m in corrupt_dfs.keys() if m in multi_omic_data]

    if n_neighbors == None:
        n_neighbors = int(np.sqrt(len(train_samples)))

    # 3) Instantiate and fit KNN_Imputer
    knn_imp = KNN_Imputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )
    knn_imp.fit(
        multi_omic_data=multi_omic_data,
        train_samples=train_samples,
        use_modalities=use_modalities,
    )

    # 4) Impute the corrupted data
    pred_dfs = knn_imp.transform(data_corrupted=corrupt_dfs)

    # 5) Optionally save predictions
    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(pred_dfs, f)
        print(f"[Saved KNN predictions] {save_pred_pickle_path}")

    return pred_dfs
