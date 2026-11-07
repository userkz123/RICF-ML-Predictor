import numpy as np
import pandas as pd

REQUIRED = ["project_id", "checkpoint", "bac", "ev", "pv", "ac", "planned_duration", "n_activities", "sector"]
# These aliases are only resolved when unambiguous; an unspecific sector is rejected.
ALIAS = {"Construction (Civil)": "Construction (Civil)", "construction (civil)": "Construction (Civil)",
         "event management": "Event Management", "Construction (civil)": "Construction (civil)"}

def sector_options(artifact):
    return sorted({s[len("sector_"):] for s in artifact["sector_columns"]})

def validate_and_prepare(data, artifact):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a CSV-style table")
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    d = data[REQUIRED].copy()
    if d.empty:
        raise ValueError("No project checkpoints provided")
    if d["project_id"].isna().any() or d["sector"].isna().any():
        raise ValueError("Every checkpoint must have project_id and sector")
    d["project_id"] = d["project_id"].astype(str).str.strip()
    d["sector"] = d["sector"].astype(str).str.strip()
    allowed = sector_options(artifact)
    # Accept exact training labels and unambiguous case-insensitive matches.
    lookup = {}
    for s in allowed:
        lookup.setdefault(s.casefold(), []).append(s)
    sectors = []
    for raw in d["sector"]:
        if raw in allowed:
            sectors.append(raw)
        elif raw.casefold() in lookup and len(lookup[raw.casefold()]) == 1:
            sectors.append(lookup[raw.casefold()][0])
        else:
            raise ValueError(f"Unknown or ambiguous sector '{raw}'. Select an exact training category: " + ", ".join(allowed))
    d["sector"] = sectors
    for col in ["checkpoint", "bac", "ev", "pv", "ac", "planned_duration", "n_activities"]:
        d[col] = pd.to_numeric(d[col],errors="coerce")
    if d[["checkpoint", "bac", "ev", "pv", "ac", "planned_duration", "n_activities"]].isna().any().any():
        raise ValueError("Input has missing or non-numeric numeric values")
    if not np.isfinite(d[["checkpoint", "bac", "ev", "pv", "ac", "planned_duration", "n_activities"]].to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite values")
    if (d[["bac", "ev", "pv", "ac", "planned_duration", "n_activities"]] <= 0).any().any():
        raise ValueError("BAC, EV, PV, AC, duration and activity count must be positive")
    if (d["ev"] >= d["bac"]).any():
        raise ValueError("Only incomplete projects: require 0 < EV < BAC")
    if (d['checkpoint'] <= 0).any() or (d['checkpoint'] % 1 != 0).any():
        raise ValueError('Checkpoints must be positive integer periods')
    if d.duplicated(["project_id", "checkpoint"]).any():
        raise ValueError("Duplicated project/checkpoint pair")
    if d.groupby('project_id')['bac'].nunique().gt(1).any():
        raise ValueError('BAC must be constant within each uploaded project; separate approved rebaselines')
    for col in ['sector','planned_duration','n_activities']:
        if d.groupby('project_id')[col].nunique().gt(1).any():
            raise ValueError(f'{col} must remain constant within an uploaded project')
    d = d.sort_values(["project_id", "checkpoint"]).reset_index(drop=True)
    for name in ["ev", "pv", "ac"]:
        changes=d.groupby('project_id')[name].diff()
        if (changes.dropna() < -1e-8).any():
            raise ValueError(f'{name.upper()} must not decrease between checkpoints (check corrections/rebaselines)')
    return d

def predict_history(data, artifact):
    d=validate_and_prepare(data, artifact)
    d["progress"] = d.ev/d.bac
    d["remaining_progress"] = 1-d.progress
    d["ac_ratio"] = d.ac/d.bac
    d["pv_ratio"] = d.pv/d.bac
    d["cpi"] = d.ev/d.ac
    d["spi"] = d.ev/d.pv
    for name in ["cpi", "spi", "ac_ratio", "progress", "pv_ratio"]:
        d[name+"_lag1"] = d.groupby("project_id")[name].shift(1)
    # A zero/negative change in EV must not be converted into an artificial large trend.
    delta_progress=d.progress-d.progress_lag1
    d["cost_run_rate"] = ((d.ac_ratio-d.ac_ratio_lag1)/delta_progress.where(delta_progress>0.005)).clip(0,10)
    d["cpi_change"] = d.cpi-d.cpi_lag1
    d["evm_eac"] = d.bac/d.cpi
    d["evm_etc_ratio"] = (d.evm_eac-d.ac)/d.bac
    d["cost_variance_ratio"] = (d.ev-d.ac)/d.bac
    d["schedule_variance_ratio"] = (d.ev-d.pv)/d.bac
    for col in artifact["sector_columns"]:
        d[col]=(d.sector==col.removeprefix("sector_")).astype(int)
    X=d.reindex(columns=artifact["features"])
    residual=np.asarray(artifact["model"].predict(X),dtype=float)
    d["experimental_ETC"]=(d.bac*(d.evm_etc_ratio+0.20*(1-d.progress)*residual)).clip(lower=0)
    d["experimental_EAC"]=d.ac+d.experimental_ETC
    d["EVM_EAC"]=d.evm_eac
    return d
