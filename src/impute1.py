import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import pickle

from .data_utils import MultiOmicDataset, get_dataloader
from .mae_masked import MultiModalWithSharedSpace


def _make_mask_random(X: np.ndarray, masking_fraction: float, rng: np.random.Generator) -> np.ndarray:
    # Mask only where X is not NaN (avoid "masking" natural missingness)
    eligible = ~np.isnan(X)
    mask = (rng.random(X.shape) < masking_fraction) & eligible
    return mask

def _make_mask_low_vals(
    X: np.ndarray,
    masking_fraction: float,
    rng: np.random.Generator,
    alpha: float = 1.0,
    transform: str = "rank",   # "rank" or "minmax"
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Create a mask that preferentially selects LOW values.

    We sample exactly K entries (without replacement) among eligible (non-NaN) entries,
    with probability proportional to a "low-value weight".
    """
    eligible = ~np.isnan(X)
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        return np.zeros_like(X, dtype=bool)

    K = int(np.floor(masking_fraction * idx.size))
    if K <= 0:
        return np.zeros_like(X, dtype=bool)

    v = X.ravel()[idx].astype(float)

    if transform == "minmax":
        # weight high when value is low: w = (max - v) normalized, then ^alpha
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        denom = (vmax - vmin) if (vmax > vmin) else 1.0
        low_score = (vmax - v) / denom  # in [0,1]
        w = (low_score + eps) ** alpha
    elif transform == "rank":
        # robust: use ranks so scaling/outliers don't dominate.
        # lowest value => rank 0 => highest low_score.
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.size)
        # low_score: highest for lowest values
        low_score = (order.size - 1 - ranks).astype(float)  # 0..(n-1)
        low_score = low_score / max(1.0, (order.size - 1))  # -> [0,1]
        w = (low_score + eps) ** alpha
    else:
        raise ValueError(f"Unknown low_vals_transform: {transform}")

    w = w / w.sum()

    chosen = rng.choice(idx, size=K, replace=False, p=w)
    mask = np.zeros(X.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(X.shape)





@torch.no_grad()
def impute_missing_values(
    model: "MultiModalWithSharedSpace",
    mask_values: Dict[str, float],
    data_corrupted: Dict[str, pd.DataFrame],   # all modalities retained, with NaNs where masked
    batch_size: int,
    device: torch.device,
    self_weight: float =10.0,
) -> Dict[str, Tuple[torch.Tensor, List[str]]]:
    """
    Impute missing VALUES (not whole modalities) given corrupted data.

    Args:
        model: trained MultiModalWithSharedSpace
        mask_values: {mod: mask_value} used during training
        data_corrupted: {mod: DataFrame}, indexed by samples, with NaNs indicating
                        entries to be imputed (plus any naturally-missing entries)
        batch_size: batch size for DataLoader
        device: torch.device

    Returns:
        imputed_raw:
            {
              mod: (X_imp: torch.Tensor [N, D], samples_used: List[str])
            }
        where X_imp is the full reconstruction for that modality.
    """
    all_modalities = list(model.modalities)
    modalities_used = list(data_corrupted.keys())

    # sanity checks
    for m in modalities_used:
        if m not in all_modalities:
            raise ValueError(f"Modality '{m}' in data_corrupted not in model.modalities")


    # Build dataset / loader
    ds = MultiOmicDataset(data_corrupted)
    loader = get_dataloader(ds, batch_size=batch_size, shuffle=False, split_idx=None)
    samples_used = ds.common_samples  # ordering used in the tensors

    # Prepare storage
    preds_torch: Dict[str, List[torch.Tensor]] = {mod: [] for mod in modalities_used}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            # batch: {mod: tensor} for all modalities_used
            batch = {m: x.to(device) for m, x in batch.items()}

            # 1) NaNs -> sentinel
            batch_clean = {}
            for mod, xb in batch.items():
                xb_clean = xb.clone()
                orig_missing = torch.isnan(xb_clean)
                xb_clean[orig_missing] = mask_values.get(mod, 0.0)
                batch_clean[mod] = xb_clean

            # 2) encode to shared space
            shared_all = {}
            for mod, xb_clean in batch_clean.items():
                h = model.encoders[mod](xb_clean)
                z = model.projections[mod](h)
                shared_all[mod] = z

            # 3) reconstruct each modality from shared mean over *all* modalities
            # (can later change this to a more complex fusion if desired)
            all_shared_mods = list(shared_all.keys())
            if len(all_shared_mods) == 0:
                continue

            # 3) reconstruct each modality using a self-weighted combination of z's
            #self_weight=10.0
            for target_mod in modalities_used:
                if target_mod not in shared_all:
                    # should not happen here, but just in case
                    continue

                # build unnormalized weights
                weights = []
                z_list = []
                for m in all_shared_mods:
                    if m == target_mod:
                        w = self_weight
                    else:
                        w = 1.0
                    weights.append(w)
                    z_list.append(shared_all[m])

                weights = torch.tensor(weights, device=device, dtype=z_list[0].dtype)
                weights = weights / weights.sum()   # normalize to sum to 1

                # stack z's: shape [M, N, D]
                z_stack = torch.stack(z_list, dim=0)    # M x N x D
                w_view = weights.view(-1, 1, 1)        # M x 1 x 1

                # weighted sum over modalities -> [N, D]
                z_weighted = (w_view * z_stack).sum(dim=0)

                h_hat = model.rev_projections[target_mod](z_weighted)
                x_imp = model.decoders[target_mod](h_hat)
                preds_torch[target_mod].append(x_imp.cpu())
                
            

    # Pack outputs
    imputed_raw: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
    for mod, chunks in preds_torch.items():
        if not chunks:
            continue
        X_m = torch.cat(chunks, dim=0)  # [N, D]
        imputed_raw[mod] = (X_m, samples_used)

    return imputed_raw

def mask_and_predict(
    model: "MultiModalWithSharedSpace",
    mask_values: Dict[str, float],
    multi_omic_data: Dict[str, pd.DataFrame],
    samples: List[str],
    masking_policy: str = "random",
    use_modalities: Optional[List[str]] = None,
    mask_modalities: Optional[List[str]] = None,   # NEW
    low_vals_alpha: float = 1.0,                   # NEW
    low_vals_transform: str = "rank",              # NEW
    seed: int = 0,                                 # NEW
    masking_fraction: float = 0.2,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    save_mask_pickle_path: Optional[str] = None,
    save_pred_pickle_path: Optional[str] = None,
    save_corrupt_pickle_path: Optional[str] = None,
    self_weight:float = 10.0,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Create artificial missing values (via masking) and impute them.

    Args:
        model: trained MultiModalWithSharedSpace
        mask_values: {mod: mask_value} used during training
        multi_omic_data: {mod: DataFrame} with full data
        samples: list of sample IDs to subset to (e.g. val / test samples)
        masking_policy: masking policy name (currently only 'random')
        use_modalities: list of modalities to retain from multi_omic_data.
                        If None, uses all modalities in both model and data.
        masking_fraction: fraction of entries to mask per modality (0-1)
        batch_size: DataLoader batch size
        device: torch.device
        save_mask_pickle_path: if not None, save mask dict here
        save_pred_pickle_path: if not None, save predictions dict here

    Returns:
        mask_dfs, pred_dfs where:
            mask_dfs[mod] is a DataFrame (bool) indicating which entries were masked
            pred_dfs[mod] is a DataFrame of reconstructed values (full matrices)
    """
    # 1) decide which modalities to use
    if use_modalities is None:
        # intersection of model.modalities and data keys
        use_modalities = [m for m in multi_omic_data.keys() if m in model.modalities]
    else:
        # sanity: all must exist in both data and model
        missing_in_data = [m for m in use_modalities if m not in multi_omic_data]
        missing_in_model = [m for m in use_modalities if m not in model.modalities]
        if missing_in_data:
            raise ValueError(f"Modalities not in multi_omic_data: {missing_in_data}")
        if missing_in_model:
            raise ValueError(f"Modalities not in model.modalities: {missing_in_model}")

    rng = np.random.default_rng(seed)

    if mask_modalities is None:
        mask_modalities = list(use_modalities)
    else:
        bad = [m for m in mask_modalities if m not in use_modalities]
        if bad:
            raise ValueError(f"mask_modalities must be a subset of use_modalities. Bad: {bad}")


    # 2) subset data to samples and selected modalities
    data_sub: Dict[str, pd.DataFrame] = {
        m: df.loc[samples].copy()
        for m, df in multi_omic_data.items()
        if m in use_modalities
    }

    # ensure alignment of samples (if any sample missing in df, .loc should raise)
    
    n_samples = len(samples)

    # 3) create masks per modality
    mask_arrays: Dict[str, np.ndarray] = {}
    for mod, df_mod in data_sub.items():
        X = df_mod.values.astype(float, copy=False)
    
        # if this modality shouldn't be masked, mask is all-False
        if mod not in mask_modalities:
            mask = np.zeros_like(X, dtype=bool)
        else:
            if masking_policy == "random":
                mask = _make_mask_random(X, masking_fraction, rng)
            elif masking_policy == "low_vals":
                mask = _make_mask_low_vals(
                    X,
                    masking_fraction,
                    rng,
                    alpha=low_vals_alpha,
                    transform=low_vals_transform,
                )
            else:
                raise ValueError(f"Unknown masking_policy: {masking_policy}")
    
        mask_arrays[mod] = mask

    # 4) apply masks to create NaN-corrupted data
    #    (NaNs -> sentinel will happen inside impute_missing_values)
    data_corrupted: Dict[str, pd.DataFrame] = {}
    for mod, df_mod in data_sub.items():
        df_cor = df_mod.copy()
        mask = mask_arrays[mod]
        values = df_cor.values.astype(float, copy=True)  # ensure float for NaNs
        values[mask] = np.nan
        df_cor.loc[:, :] = values
        data_corrupted[mod] = df_cor

    # 5) impute missing values using the shared autoencoder
    imputed_raw = impute_missing_values(
        model=model,
        mask_values=mask_values,
        data_corrupted=data_corrupted,
        batch_size=batch_size,
        device=device,
        self_weight = self_weight,
    )

    # 6) convert masks, corrupted, and predictions to DataFrames with same indices/columns
    mask_dfs: Dict[str, pd.DataFrame] = {}
    pred_dfs: Dict[str, pd.DataFrame] = {}
    corrupt_dfs: Dict[str, pd.DataFrame] = {}

    # impute_missing_values may potentially drop some samples → use its ordering
    # remember to double check this
    # (we assume for now that MultiOmicDataset uses the same sample set)
    for mod in use_modalities:
        if mod not in imputed_raw:
            # could not reconstruct this modality for some reason
            continue

        X_imp, samples_used = imputed_raw[mod]
        cols = data_sub[mod].columns

        # predicted values (reconstructions)
        df_pred = pd.DataFrame(X_imp.numpy(), index=samples_used, columns=cols)
        pred_dfs[mod] = df_pred

        # mask in the same order
        mask_full = mask_arrays[mod]
        mask_df_full = pd.DataFrame(mask_full, index=samples, columns=cols)
        mask_dfs[mod] = mask_df_full.loc[samples_used]

        # corrupted data in the same order (NaNs already applied in data_corrupted)
        df_cor_full = data_corrupted[mod]
        corrupt_dfs[mod] = df_cor_full.loc[samples_used, cols]

    # 7) optionally save all 3  dicts as pickles
    if save_mask_pickle_path is not None:
        with open(save_mask_pickle_path, "wb") as f:
            pickle.dump(mask_dfs, f)
        print(f"[Saved mask dict] {save_mask_pickle_path}")

    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(pred_dfs, f)
        print(f"[Saved predictions dict] {save_pred_pickle_path}")

    if save_corrupt_pickle_path is not None:
        with open(save_corrupt_pickle_path, "wb") as f:
            pickle.dump(corrupt_dfs, f)
        print(f"[Saved corrupted data dict] {save_corrupt_pickle_path}")

    return mask_dfs, pred_dfs
