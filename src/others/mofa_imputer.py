# mofa_imputer.py

import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from mofapy2.run.entry_point import entry_point
import mofax as mofa


# ----------------------------------------------------------------------
# 1) TRAINING: global MOFA model on training samples only
# ----------------------------------------------------------------------


def train_global_mofa(
    multi_omic_data: Dict[str, pd.DataFrame],
    train_samples: List[str],
    out_hdf5_path: str,
    views_order: Optional[List[str]] = None,
    n_factors: Optional[int] = None,
    train_iter: Optional[int] = None,
    seed: Optional[int] = None,
    use_float32: bool = True,
    verbose: bool = True,
) -> str:
    """
    Train a global MOFA model using mofapy2 on training samples only,
    and save it as an HDF5 file.

    Args
    ----
    multi_omic_data
        dict(modality -> DataFrame [samples x features]) with full dataset.
    train_samples
        Sample IDs to use for MOFA training.
    out_hdf5_path
        Path to write the MOFA model HDF5 file.
    views_order
        Optional explicit order of modalities (views). If None, uses
        sorted(multi_omic_data.keys()).
    n_factors
        If provided, overrides the default number of factors via
        `set_model_options(factors=...)`. If None, keep MOFA defaults.
    train_iter
        If provided, overrides the default number of iterations via
        `set_train_options(iter=...)`. If None, keep MOFA defaults.
    seed
        Optional random seed for training (passed to `set_train_options`).
    use_float32
        Cast data to float32 before sending to MOFA (saves memory).
    verbose
        Print progress messages.

    Returns
    -------
    out_hdf5_path
        The path to the saved MOFA model.
    """
    if views_order is None:
        views_order = sorted(multi_omic_data.keys())

    # Sanity: ensure train_samples exist in all modalities
    for mod, df in multi_omic_data.items():
        missing = set(train_samples) - set(df.index)
        if missing:
            raise ValueError(
                f"Some train_samples are missing in modality '{mod}': "
                f"{sorted(list(missing))[:5]} ..."
            )

    if verbose:
        print("[MOFA train] Views:", views_order)
        print("[MOFA train] N_train:", len(train_samples))

    # --------------------------------------------------------------
    # Build data_mat exactly like the mofapy2 tutorial:
    #   data_mat[m][g] = numpy array with shape (samples x features)
    # Here we have:
    #   M = number of views (modalities)
    #   G = 1 group
    # --------------------------------------------------------------
    M = len(views_order)
    G = 1
    data_mat = [[None for _ in range(G)] for _ in range(M)]

    for m, mod in enumerate(views_order):
        df = multi_omic_data[mod].loc[train_samples]
        X = df.to_numpy()          # (N_samples x D_features)
        if use_float32:
            X = X.astype(np.float32)
        data_mat[m][0] = X

    ent = entry_point()

    # Data options (we assume you've already scaled/z-scored as needed)
    ent.set_data_options(
        scale_views=False,
    )

    # Use Gaussian likelihood for all views (standard for continuous omics)
    likelihoods = ["gaussian"] * M

    # Set the data matrix + likelihoods
    ent.set_data_matrix(
        data_mat,
        likelihoods=likelihoods,
    )

    # --------------------------------------------------------------
    # Model + training options: MUST set model_options before train_options
    # --------------------------------------------------------------

    # Always define model options, even if you just want defaults
    if n_factors is not None:
        ent.set_model_options(factors=n_factors)
    else:
        ent.set_model_options()

    # Always define training options, even if you just want defaults
    train_kwargs = {}
    if train_iter is not None:
        train_kwargs["iter"] = train_iter
    if seed is not None:
        train_kwargs["seed"] = seed

    if train_kwargs:
        ent.set_train_options(**train_kwargs)
    else:
        ent.set_train_options()

    if verbose:
        print("[MOFA train] Building model...")
    ent.build()

    if verbose:
        print("[MOFA train] Running model...")
    ent.run()

    # Save the model (and data) to HDF5
    out_dir = os.path.dirname(out_hdf5_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ent.save(out_hdf5_path, save_data=True)

    if verbose:
        print(f"[MOFA train] Model saved to {out_hdf5_path}")

    return out_hdf5_path


# ----------------------------------------------------------------------
# 2) IMPUTATION: use mofax.project_data + get_weights
# ----------------------------------------------------------------------


class MOFAGlobalImputer:
    """
    MOFA-based imputer using a single global model trained on train samples.

    Workflow:
      1. Train MOFA once on train_samples (all modalities) using `train_global_mofa`.
      2. Load the HDF5 with `mofax.mofa_model`.
      3. For each scenario:
         - Choose a projection view (one of the present modalities).
         - Use project_data(...) to get Z for scenario samples.
         - Use get_weights(view=target_mod) to reconstruct the missing modality.

    This avoids retraining MOFA per scenario and uses mofax's recommended
    projection API.
    """

    def __init__(
        self,
        hdf5_path: str,
        multi_omic_data: Dict[str, pd.DataFrame],
        views_order: Optional[List[str]] = None,
        projection_view: Optional[str] = None,
        use_multi_view_projection: bool = False,
        verbose: bool = False,
    ):
        """
        Args
        ----
        hdf5_path
            Path to the global MOFA model (HDF5) trained on train_samples.
        multi_omic_data
            Same dict(modality -> DataFrame [samples x features]) used to
            train the model, containing train + val/test.
        views_order
            Optional explicit order of modalities (views). If None, uses
            sorted(multi_omic_data.keys()).
        projection_view
            Optional name of the modality to use when projecting scenario
            samples into factor space. If None, we pick the first present
            modality in `views_order` for each scenario.
        use_multi_view_projection
            If False (default): use one view only via mofax.project_data().
            If True: use a custom multi-view least-squares projection that
            combines all present views (see docstring below).
        verbose
            Print progress information.
        """
        self.hdf5_path = hdf5_path
        self.multi_omic_data = multi_omic_data
        self.views_order = (
            list(views_order) if views_order is not None else sorted(multi_omic_data.keys())
        )
        self.projection_view = projection_view
        self.use_multi_view_projection = use_multi_view_projection
        self.verbose = verbose

        # Load the model once
        self.model = mofa.mofa_model(hdf5_path)

        # Map modality names -> view indices (0, 1, 2, ...)
        # This must match the order used in train_global_mofa
        self.mod_to_view_index = {
            mod: i for i, mod in enumerate(self.views_order)
        }

    # ------------------------------------------------------------------
    # Helper: choose projection view per scenario
    # ------------------------------------------------------------------
    def _choose_projection_view(self, modalities_present: List[str]) -> str:
        if self.projection_view is not None:
            if self.projection_view not in modalities_present:
                raise ValueError(
                    f"projection_view='{self.projection_view}' not in modalities_present={modalities_present}"
                )
            return self.projection_view

        # Default: first modality in views_order that is present
        for mod in self.views_order:
            if mod in modalities_present:
                return mod

        raise ValueError(
            f"No projection view available: modalities_present={modalities_present}, "
            f"views_order={self.views_order}"
        )

    # ------------------------------------------------------------------
    # Helper: single-view projection (using mofax.project_data)
    # ------------------------------------------------------------------
    def _project_single_view(
        self,
        scenario_samples: List[str],
        proj_view: str,
    ) -> pd.DataFrame:
        """
        Project scenario samples onto the factor space using a single view.

        Uses mofax.project_data(data=..., view=view_idx).

        Returns
        -------
        Z_new : DataFrame [scenario_samples x K]
        """
        df_view = self.multi_omic_data[proj_view].loc[scenario_samples].copy()

        # For projection, handle any NaNs — your data are usually z-scored,
        # so filling with 0 (mean) is reasonable.
        df_view = df_view.fillna(0.0)

        view_idx = self.mod_to_view_index[proj_view]

        Z_new = self.model.project_data(
            data=df_view,
            view=view_idx,          # integer index, e.g. 0 for first view
            df=True,
            feature_intersection=False,
        )
        # project_data returns samples x factors, index should match scenario samples
        return Z_new

    # ------------------------------------------------------------------
    # Helper: multi-view projection
    # ------------------------------------------------------------------
    def _project_multi_view(
        self,
        scenario_samples: List[str],
        modalities_present: List[str],
    ) -> pd.DataFrame:
        """
        Multi-view least-squares projection:
    
          Z_new = (sum_v X_new^(v) W^(v)) (sum_v W^(v)T W^(v))^{-1}
    
        Here we align features **by position**, assuming:
          - multi_omic_data[v].columns are in the same order as in training
          - MOFA's weights rows correspond to those columns in the same order.
        """
        XW_sum = None          # (N x K)
        WtW_sum = None         # (K x K)
        Z_index = None
        Z_columns = None
    
        for v in modalities_present:
            df_view = self.multi_omic_data[v].loc[scenario_samples].copy()
            df_view = df_view.fillna(0.0)
    
            view_idx = self.mod_to_view_index[v]
    
            # weights: features x factors (as DataFrame)
            W_v_df = self.model.get_weights(views=view_idx, df=True)
    
            X_v = df_view.to_numpy()          # (N x D_v)
            W_v = W_v_df.to_numpy()           # (D_v x K)
    
            if X_v.shape[1] != W_v.shape[0]:
                raise ValueError(
                    f"Shape mismatch for view '{v}': "
                    f"X_v has {X_v.shape[1]} features, W_v has {W_v.shape[0]} rows."
                )
    
            XW = X_v @ W_v                    # (N x K)
            WtW = W_v.T @ W_v                 # (K x K)
    
            if XW_sum is None:
                XW_sum = XW
                WtW_sum = WtW
                Z_index = df_view.index
                Z_columns = list(W_v_df.columns)  # factor names
            else:
                XW_sum += XW
                WtW_sum += WtW
    
        if XW_sum is None or WtW_sum is None:
            raise ValueError(
                "Multi-view projection failed: no usable views "
                "with matching shapes for the present modalities."
            )
    
        # Solve for Z
        eps = 1e-6
        WtW_reg = WtW_sum + eps * np.eye(WtW_sum.shape[0])
        Z_new = XW_sum @ np.linalg.inv(WtW_reg)          # (N x K)
    
        Z_df = pd.DataFrame(Z_new, index=Z_index, columns=Z_columns)
        return Z_df


    # ------------------------------------------------------------------
    # Helper: reconstruct target modality from Z_new
    # ------------------------------------------------------------------
    def _reconstruct_target(
        self,
        Z_new: pd.DataFrame,
        target_mod: str,
        scenario_samples: List[str],
    ) -> pd.DataFrame:
        """
        Given:
          - Z_new (scenario_samples x K)
          - target_mod (view name)
    
        Use W_target (features x K) to reconstruct:
    
          Y_hat = Z_new @ W_target^T
    
        We:
          - align by factor names (columns)
          - align features by **position**
          - set output columns to the original feature names of target_mod
        """
        view_idx = self.mod_to_view_index[target_mod]
        W_target_df = self.model.get_weights(views=view_idx, df=True)  # features x factors
    
        # Align by factor (column) names
        common_factors = Z_new.columns.intersection(W_target_df.columns)
        if len(common_factors) == 0:
            raise ValueError(
                f"No overlapping factors between Z_new and W for target '{target_mod}'."
            )
    
        Z_use = Z_new[common_factors].to_numpy()          # (N x K')
        W_use = W_target_df[common_factors].to_numpy()    # (D_target x K')
    
        # Use original feature names from the multi_omic_data for target_mod
        feat_names = list(self.multi_omic_data[target_mod].columns)
    
        if W_use.shape[0] != len(feat_names):
            raise ValueError(
                f"Feature count mismatch for target '{target_mod}': "
                f"W has {W_use.shape[0]} rows, but target modality has "
                f"{len(feat_names)} columns."
            )
    
        Y_hat = Z_use @ W_use.T                           # (N x D_target)
    
        imputed_df = pd.DataFrame(
            Y_hat,
            index=scenario_samples,
            columns=feat_names,
        )
        return imputed_df

    # ------------------------------------------------------------------
    # Public API: impute one scenario
    # ------------------------------------------------------------------
    def impute_for_scenario(self, scenario_payload: dict) -> pd.DataFrame:
        """
        Impute the missing modality for a single scenario.

        scenario_payload structure (same as your TOBMI payload):
          - "modalities_present": List[str]
          - "missing_modality": str      (target modality)
          - "samples": List[str]         (scenario samples)
          - "data": Dict[str, DataFrame] (present modalities only; unused here)

        Returns
        -------
        imputed_df : DataFrame [scenario_samples x features_target_mod]
        """
        modalities_present: List[str] = scenario_payload["modalities_present"]
        target_mod: str = scenario_payload["missing_modality"]
        scenario_samples: List[str] = scenario_payload["samples"]

        if target_mod not in self.multi_omic_data:
            raise ValueError(f"Target modality '{target_mod}' not found in multi_omic_data.")

        if self.verbose:
            print(
                f"[MOFA impute] present={modalities_present}, "
                f"target={target_mod}, n_scenario={len(scenario_samples)}"
            )

        # Decide how to get Z_new
        if self.use_multi_view_projection:
            Z_new = self._project_multi_view(
                scenario_samples=scenario_samples,
                modalities_present=modalities_present,
            )
        else:
            proj_view = self._choose_projection_view(modalities_present)
            if self.verbose:
                print(f"[MOFA impute] Using projection view: {proj_view}")
            Z_new = self._project_single_view(
                scenario_samples=scenario_samples,
                proj_view=proj_view,
            )

        imputed_df = self._reconstruct_target(
            Z_new=Z_new,
            target_mod=target_mod,
            scenario_samples=scenario_samples,
        )
        return imputed_df


# ----------------------------------------------------------------------
# 3) Convenience wrapper: loop over scenarios (TOBMI-style)
# ----------------------------------------------------------------------


def translate_from_scenario_dir(
    scenarios_dir: str,
    mofa_hdf5_path: str,
    multi_omic_data: Dict[str, pd.DataFrame],
    views_order: Optional[List[str]] = None,
    projection_view: Optional[str] = None,
    use_multi_view_projection: bool = True,
    verbose: bool = False,
    save_pred_pickle_path: Optional[str] = None,
) -> Dict[Tuple[Tuple[str, ...], str], pd.DataFrame]:
    """
    Loop over all scenario pickles in a directory and impute the missing modality
    for each scenario using a *global* MOFA model + mofax projection.

    Args
    ----
    scenarios_dir
        Directory containing scenario .pkl files.
    mofa_hdf5_path
        Path to the global MOFA model HDF5 (trained on train samples).
    multi_omic_data
        Full multi-omic dataset as dict(modality -> DataFrame [samples x features]).
    views_order
        Optional explicit ordering of modalities (views).
    projection_view
        If provided, use this view for projection in all scenarios; otherwise,
        choose the first present view in `views_order` per scenario.
    use_multi_view_projection
        If True, use least-squares multi-view projection across all present
        views instead of single-view `project_data`.
    verbose
        Print progress information.
    save_pred_pickle_path
        If provided, save the predictions dict to this pickle path.

    Returns
    -------
    predictions
        dict:
          keys   = (tuple(sorted_present_mods), target_mod)
          values = DataFrame [scenario_samples x features_target_mod]
    """
    imputer = MOFAGlobalImputer(
        hdf5_path=mofa_hdf5_path,
        multi_omic_data=multi_omic_data,
        views_order=views_order,
        projection_view=projection_view,
        use_multi_view_projection=use_multi_view_projection,
        verbose=verbose,
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

        if verbose:
            print(f"\n[MOFA translate] Scenario file: {fname}")

        imputed_df = imputer.impute_for_scenario(scenario_payload)

        key = (tuple(sorted(modalities_present)), target_mod)
        predictions[key] = imputed_df

    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(predictions, f)
        if verbose:
            print(f"[MOFA translate] Saved predictions to {save_pred_pickle_path}")

    return predictions
