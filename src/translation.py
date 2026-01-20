import torch
import pandas as pd
import os
import pickle
import itertools
from typing import Dict, List, Tuple, Optional, Any


from .data_utils import MultiOmicDataset, get_dataloader
from .mae_masked import MultiModalWithSharedSpace

def impute_missing_modalities_for_scenario(
    model: "MultiModalWithSharedSpace",
    mask_values: Dict[str, float],
    data_present: Dict[str, pd.DataFrame],   # already missing target mods
    target_modalities: List[str],
    batch_size: int,
    device: torch.device,
) -> Dict[str, pd.DataFrame]:
    """
    Impute target_modalities given that data_present already contains only the
    modalities that are actually present (target modalities are NOT in data_present).

    Args:
        model: trained MultiModalWithSharedSpace
        mask_values: {mod: mask_value} used during training
        data_present: {mod: DataFrame} for present modalities only
        target_modalities: modalities to impute
        batch_size: batch size for DataLoader
        device: torch.device

    Returns:
        {target_mod: DataFrame} of imputed values, indexed by dataset's common_samples,
        columns = original feature names for that modality (you must have access to
        multi_omic_data[target_mod].columns externally when you wrap this).
    """
    all_modalities = list(model.modalities)
    modalities_present = list(data_present.keys())

    # sanity checks
    for m in modalities_present:
        if m not in all_modalities:
            raise ValueError(f"Modality '{m}' in data_present not in model.modalities")

    # Build dataset from present modalities only
    ds = MultiOmicDataset(data_present)
    loader = get_dataloader(ds, batch_size=batch_size, shuffle=False, split_idx=None)

    samples_used = ds.common_samples  # intersection/ordering decided here

    preds_torch = {mod: [] for mod in target_modalities if mod in all_modalities}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {m: x.to(device) for m, x in batch.items()}  # only present mods

            # 1) NaNs -> sentinel
            batch_clean = {}
            for mod, xb in batch.items():
                xb_clean = xb.clone()
                orig_missing = torch.isnan(xb_clean)
                xb_clean[orig_missing] = mask_values.get(mod, 0.0)
                batch_clean[mod] = xb_clean

            # 2) shared embeddings for present modalities
            shared_all = {}
            for mod, xb_clean in batch_clean.items():
                h = model.encoders[mod](xb_clean)
                z = model.projections[mod](h)
                shared_all[mod] = z

            # 3) impute each target modality from present ones
            for target_mod in target_modalities:
                if target_mod not in all_modalities:
                    continue
                other_mods = [m for m in modalities_present if m != target_mod and m in shared_all]
                if len(other_mods) == 0:
                    continue  # cannot impute

                z_mean = torch.stack([shared_all[m] for m in other_mods], dim=0).mean(dim=0)
                h_hat = model.rev_projections[target_mod](z_mean)
                x_imp = model.decoders[target_mod](h_hat)
                preds_torch[target_mod].append(x_imp.cpu())

    # Return raw tensors + sample order; wrapping into DataFrame is easier outside
    imputed_raw: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
    for target_mod, chunks in preds_torch.items():
        if not chunks:
            continue
        X_t = torch.cat(chunks, dim=0)  # [N, D]
        imputed_raw[target_mod] = (X_t, samples_used)

    return imputed_raw



def leave_one_out_imputation(
    model: "MultiModalWithSharedSpace",
    mask_values: Dict[str, float],
    multi_omic_data: Dict[str, pd.DataFrame],
    common_samples: List[str],
    batch_size: int,
    device: torch.device,
    scenarios_dir: str = None,         # where to save scenario pickles (optional)
    save_pred_pickle_path: str = None  # where to save the predictions dict (optional)
) -> Dict[Tuple[Tuple[str, ...], str], pd.DataFrame]:
    """
    For each modality t:
      - build data_present: multi_omic_data without t (restricted to common_samples)
      - optionally save that scenario
      - run imputation for t
      - return predictions as {((present_mods), t): DataFrame}

    Args:
        model, mask_values: trained shared model + sentinels
        multi_omic_data: {mod: DataFrame} with *all* modalities
        common_samples: samples to align on
        batch_size, device
        scenarios_dir: if not None, save each scenario as a pickle here
        save_pred_pickle_path: if not None, save predictions dict here

    Returns:
        predictions dict: keys ((sorted_present_mods), target_mod), values DataFrame.
    """
    all_modalities = list(multi_omic_data.keys())
    predictions: Dict[Tuple[Tuple[str, ...], str], pd.DataFrame] = {}

    if scenarios_dir is not None:
        os.makedirs(scenarios_dir, exist_ok=True)

    for target_mod in all_modalities:
        # 1) Scenario data: all modalities except target_mod
        modalities_present = [m for m in all_modalities if m != target_mod]
        data_present = {
            m: df.loc[common_samples].copy()
            for m, df in multi_omic_data.items()
            if m in modalities_present
        }

        # Optionally save this scenario so you can reuse it with other models
        if scenarios_dir is not None:
            scenario_payload = {
                "modalities_present": modalities_present,
                "missing_modality": target_mod,
                "samples": common_samples,
                "data": data_present,   # dict[mod] -> DataFrame
            }
            fname = f"scenario_present_{'_'.join(sorted(modalities_present))}_missing_{target_mod}.pkl"
            scenario_path = os.path.join(scenarios_dir, fname)
            with open(scenario_path, "wb") as f:
                pickle.dump(scenario_payload, f)
            print(f"[Saved scenario] {scenario_path}")

        # 2) Run imputation for this scenario
        imputed_raw = impute_missing_modalities_for_scenario(
            model=model,
            mask_values=mask_values,
            data_present=data_present,              # already missing target_mod
            target_modalities=[target_mod],
            batch_size=batch_size,
            device=device,
        )

        if target_mod not in imputed_raw:
            print(f"[WARN] Could not impute modality '{target_mod}' from {modalities_present}")
            continue

        X_t, samples_used = imputed_raw[target_mod]
        cols = multi_omic_data[target_mod].columns
        df_imp = pd.DataFrame(X_t.numpy(), index=samples_used, columns=cols)

        key = (tuple(sorted(modalities_present)), target_mod)
        predictions[key] = df_imp

    # Optionally save all predictions to a pickle
    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"[Saved predictions] {save_pred_pickle_path}")

    return predictions



def all_possible_imputation(
    model: "MultiModalWithSharedSpace",
    mask_values: Dict[str, float],
    multi_omic_data: Dict[str, pd.DataFrame],
    common_samples: List[str],
    batch_size: int,
    device: torch.device,
    scenarios_dir: Optional[str] = None,          # optional: save scenario pickles
    save_pred_pickle_path: Optional[str] = None,  # optional: save predictions dict
    *,
    # Controls for the scenario enumeration:
    max_missing_others: Optional[int] = None,     # cap size of "additional missing" set; None = all sizes
    min_present_modalities: int = 1,              # require at least this many present modalities to impute from
    include_no_extra_missing: bool = True,        # include the scenario where only target is missing
    max_scenarios_total: Optional[int] = None,    # hard cap on total scenarios (across all targets)
    skip_if_exists: bool = False,                 # if scenarios_dir set and pickle exists, don't overwrite
) -> Dict[Tuple[Tuple[str, ...], str], pd.DataFrame]:
    """
    Enumerate scenarios for imputing each modality under all combinations of
    additional missing modalities.

    For each target modality t:
      - pick missing_others subset of (all_modalities \ {t})
      - present = (all_modalities \ {t}) \ missing_others
      - impute t from present

    Returns:
      predictions: dict keyed by ((sorted_present_mods), target_mod) -> DataFrame (imputed)
    """
    all_modalities = list(multi_omic_data.keys())
    predictions: Dict[Tuple[Tuple[str, ...], str], pd.DataFrame] = {}

    if scenarios_dir is not None:
        os.makedirs(scenarios_dir, exist_ok=True)

    scenarios_done = 0

    for target_mod in all_modalities:
        other_modalities = [m for m in all_modalities if m != target_mod]

        # Determine subset sizes to enumerate
        max_k = len(other_modalities) if max_missing_others is None else min(max_missing_others, len(other_modalities))
        k_start = 0 if include_no_extra_missing else 1

        for k in range(k_start, max_k + 1):
            for missing_others in itertools.combinations(other_modalities, k):
                missing_others = list(missing_others)

                # present modalities are those not missing and not the target
                modalities_present = [m for m in other_modalities if m not in missing_others]

                # must have enough present modalities to form an embedding mean
                if len(modalities_present) < min_present_modalities:
                    continue

                # Optional global cap
                if max_scenarios_total is not None and scenarios_done >= max_scenarios_total:
                    if save_pred_pickle_path is not None:
                        with open(save_pred_pickle_path, "wb") as f:
                            pickle.dump(predictions, f)
                        print(f"[Saved predictions early due to cap] {save_pred_pickle_path}")
                    return predictions

                # Build present data restricted to common_samples
                data_present = {
                    m: multi_omic_data[m].loc[common_samples].copy()
                    for m in modalities_present
                }

                # Optionally save scenario
                if scenarios_dir is not None:
                    scenario_payload = {
                        "modalities_present": modalities_present,
                        "missing_modality": target_mod,
                        "missing_others": missing_others,
                        "samples": common_samples,
                        "data": data_present,
                    }

                    present_tag = "_".join(sorted(modalities_present)) if modalities_present else "NONE"
                    missing_tag = "_".join(sorted([target_mod] + missing_others))
                    fname = f"scenario_present_{present_tag}_missing_{missing_tag}.pkl"
                    scenario_path = os.path.join(scenarios_dir, fname)

                    if (not skip_if_exists) or (not os.path.exists(scenario_path)):
                        with open(scenario_path, "wb") as f:
                            pickle.dump(scenario_payload, f)
                        print(f"[Saved scenario] {scenario_path}")

                # Run imputation for this scenario
                imputed_raw = impute_missing_modalities_for_scenario(
                    model=model,
                    mask_values=mask_values,
                    data_present=data_present,
                    target_modalities=[target_mod],
                    batch_size=batch_size,
                    device=device,
                )

                if target_mod not in imputed_raw:
                    print(f"[WARN] Could not impute '{target_mod}' from present={modalities_present}")
                    continue

                X_t, samples_used = imputed_raw[target_mod]
                cols = multi_omic_data[target_mod].columns
                df_imp = pd.DataFrame(X_t.numpy(), index=samples_used, columns=cols)

                key = (tuple(sorted(modalities_present)), target_mod)
                predictions[key] = df_imp

                scenarios_done += 1

    # Optionally save all predictions
    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"[Saved predictions] {save_pred_pickle_path}")

    return predictions

