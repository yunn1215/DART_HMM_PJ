# run_hmm_regime_model.py
#
# Install required libraries:
#   pip install pandas numpy matplotlib scikit-learn hmmlearn openpyxl seaborn
#
# Usage:
#   cd "C:\dart_s&p\처음 쓰는 데이터"
#   python run_hmm_regime_model.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import math
import warnings
import textwrap
from datetime import datetime

# Force UTF-8 output on Windows (avoids CP949 UnicodeEncodeError)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# PATHS  — change only here
# ═════════════════════════════════════════════════════════════════════════════
DATA_DIR   = r"C:\dart_s&p\처음 쓰는 데이터"
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
FIGURE_DIR = os.path.join(DATA_DIR, "figures")

SP500_FILE  = r"C:\dart_s&p\SP500_monthly_returns_1990_2026.xlsx"
VIX_FILE    = r"C:\dart_s&p\VIX.xlsx"
TERM_FILE   = os.path.join(DATA_DIR, "T10Y3MM_diff.xlsx")
CREDIT_FILE = r"C:\dart_s&p\BaaAaa_Spread.xlsx"

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
IS_END_YM    = "2018-12"
OOS_START_YM = "2019-01"
N_STATES        = 3
N_ITER          = 1000
N_INITS         = 20
RANDOM_SEED     = 42
TRANSMAT_PRIOR  = 1.05   # Dirichlet prior for transition matrix (>1 prevents exact-zero transitions)
                          # default 1.0 = pure MLE (can produce 0%); 1.05 adds pseudo-count 0.05 per cell

ALL_FEATURES  = ["sp500_log_return", "vix_eom", "term_spread", "credit_spread"]
REGIME_COLORS = {
    "Low-Vol Expansion": "#4CAF50",   # green  — stable, good Sharpe
    "High-Beta Bull":    "#FF9800",   # orange — volatile positive returns
    "Crisis":            "#F44336",   # red    — stressed / negative
}
LABEL_ORDER = ["Low-Vol Expansion", "High-Beta Bull", "Crisis"]

# ═════════════════════════════════════════════════════════════════════════════
# MODEL SPECIFICATIONS
# ═════════════════════════════════════════════════════════════════════════════
MODEL_SPECS = {
    "M4_all": {
        "name": "S&P500 + VIX + Term Spread + Credit Spread",
        "features": ["sp500_log_return", "vix_eom", "term_spread", "credit_spread"],
    },
    "M3_sp500_vix_term": {
        "name": "S&P500 + VIX + Term Spread",
        "features": ["sp500_log_return", "vix_eom", "term_spread"],
    },
    "M3_sp500_vix_credit": {
        "name": "S&P500 + VIX + Credit Spread",
        "features": ["sp500_log_return", "vix_eom", "credit_spread"],
    },
    "M3_sp500_term_credit": {
        "name": "S&P500 + Term Spread + Credit Spread",
        "features": ["sp500_log_return", "term_spread", "credit_spread"],
    },
    "M2_sp500_vix": {
        "name": "S&P500 + VIX",
        "features": ["sp500_log_return", "vix_eom"],
    },
    "M2_sp500_term": {
        "name": "S&P500 + Term Spread",
        "features": ["sp500_log_return", "term_spread"],
    },
    "M2_sp500_credit": {
        "name": "S&P500 + Credit Spread",
        "features": ["sp500_log_return", "credit_spread"],
    },
    "M1_sp500": {
        "name": "S&P500 only",
        "features": ["sp500_log_return"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_dirs():
    """Create output/figure directories for all models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for mid in MODEL_SPECS:
        os.makedirs(os.path.join(OUTPUT_DIR, mid), exist_ok=True)
        os.makedirs(os.path.join(FIGURE_DIR, mid), exist_ok=True)


def _to_ym(dates):
    """Convert DatetimeIndex or Series to year-month PeriodIndex/Series."""
    dt = pd.to_datetime(dates)
    if isinstance(dt, pd.Series):
        return dt.dt.to_period("M")
    return dt.to_period("M")


def _n_params(n_states, n_features):
    """Free parameters for diagonal-covariance GaussianHMM."""
    init_prob = n_states - 1
    transmat  = n_states * (n_states - 1)
    means     = n_states * n_features
    variances = n_states * n_features
    return init_prob + transmat + means + variances


def _mean_duration(transmat, state_to_label):
    """Return dict {label: duration} where duration=1/(1-p_ii)."""
    durations = {}
    for s, label in state_to_label.items():
        p_ii = transmat[s, s]
        if p_ii >= 0.9990:
            durations[label] = np.nan
        else:
            durations[label] = 1.0 / (1.0 - p_ii)
    return durations


def _ym_str(period):
    """Period → 'YYYY-MM' string."""
    return str(period)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1] Loading data files...")

    # ── SP500 ─────────────────────────────────────────────────
    sp = pd.read_excel(SP500_FILE, sheet_name="Monthly_Returns", parse_dates=["observation_date"])
    sp["month"] = _to_ym(sp["observation_date"])
    sp = sp[["month", "log_return"]].rename(columns={"log_return": "sp500_log_return"})
    print(f"  SP500  : {len(sp)} rows, {sp['month'].min()} ~ {sp['month'].max()}")

    # ── VIX ───────────────────────────────────────────────────
    vx = pd.read_excel(VIX_FILE, sheet_name="Monthly_EOM",
                       parse_dates=["month_end_observation_date"])
    vx["month"] = _to_ym(vx["month_end_observation_date"])
    vx = vx[["month", "VIX_EOM"]].rename(columns={"VIX_EOM": "vix_eom"})
    print(f"  VIX    : {len(vx)} rows, {vx['month'].min()} ~ {vx['month'].max()}")

    # ── Term Spread (T10Y3MM level) ───────────────────────────
    ts = pd.read_excel(TERM_FILE, parse_dates=["observation_date"])
    ts["month"] = _to_ym(ts["observation_date"])
    ts = ts[["month", "T10Y3MM"]].rename(columns={"T10Y3MM": "term_spread"})
    print(f"  T10Y3MM: {len(ts)} rows, {ts['month'].min()} ~ {ts['month'].max()}")

    # ── Credit Spread (BaaAaa – daily → month-end) ────────────
    cs = pd.read_excel(CREDIT_FILE, parse_dates=["Date"])
    cs = cs.set_index("Date").sort_index()
    cs_monthly = cs["Spread"].resample("ME").last().reset_index()
    cs_monthly["month"] = _to_ym(cs_monthly["Date"])
    cs_monthly = cs_monthly[["month", "Spread"]].rename(columns={"Spread": "credit_spread"})
    print(f"  Credit : {len(cs_monthly)} rows, {cs_monthly['month'].min()} ~ {cs_monthly['month'].max()}")

    return sp, vx, ts, cs_monthly


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING & MERGE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(sp, vx, ts, cs):
    print("\n[2] Merging and preprocessing...")

    master = (sp.merge(vx,  on="month", how="inner")
                .merge(ts,  on="month", how="inner")
                .merge(cs,  on="month", how="inner"))

    before = len(master)
    master = master.dropna(subset=ALL_FEATURES)
    after  = len(master)
    print(f"  Rows before dropna : {before}")
    print(f"  Rows after  dropna : {after}  (dropped {before - after})")

    master = master.sort_values("month").reset_index(drop=True)
    print(f"  Date range         : {master['month'].min()} ~ {master['month'].max()}")
    return master


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAMPLE SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_sample(master):
    is_end    = pd.Period(IS_END_YM,    "M")
    oos_start = pd.Period(OOS_START_YM, "M")

    master = master.copy()
    master["sample_period"] = np.where(
        master["month"] <= is_end, "in_sample", "out_of_sample"
    )

    df_is  = master[master["sample_period"] == "in_sample"].copy()
    df_oos = master[master["sample_period"] == "out_of_sample"].copy()

    print(f"\n[3] Sample split:")
    print(f"  In-sample  : {len(df_is)} months  ({df_is['month'].min()} ~ {df_is['month'].max()})")
    print(f"  OOS        : {len(df_oos)} months  ({df_oos['month'].min()} ~ {df_oos['month'].max()})")
    return df_is, df_oos


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCALING
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(df_is, df_oos, features):
    """Fit StandardScaler on IS only; transform IS and OOS."""
    scaler = StandardScaler()
    X_is  = scaler.fit_transform(df_is[features].values)
    X_oos = scaler.transform(df_oos[features].values)
    return X_is, X_oos, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT HMM
# ─────────────────────────────────────────────────────────────────────────────

def fit_hmm(X_train, model_id):
    """
    Fit GaussianHMM with N_INITS random starts; keep best log-likelihood.
    Returns (best_model, converged, best_logL).
    """
    print(f"  Fitting HMM for {model_id}  ({N_INITS} inits) ...", end=" ", flush=True)
    best_ll, best_model = -np.inf, None

    for i in range(N_INITS):
        try:
            m = GaussianHMM(
                n_components=N_STATES,
                covariance_type="diag",
                n_iter=N_ITER,
                tol=1e-6,
                random_state=RANDOM_SEED + i,
                verbose=False,
                transmat_prior=TRANSMAT_PRIOR,
            )
            m.fit(X_train)
            ll = m.score(X_train)
            if ll > best_ll:
                best_ll    = ll
                best_model = m
        except Exception:
            continue

    converged = best_model.monitor_.converged if best_model is not None else False
    print(f"logL={best_ll:.3f}  converged={converged}")
    return best_model, converged, best_ll


# ─────────────────────────────────────────────────────────────────────────────
# 6. REGIME LABELING
# ─────────────────────────────────────────────────────────────────────────────

def label_regimes(model, X_is, df_is, scaler, features):
    """
    Assign economic labels to HMM states using empirical IS statistics.

    Step 1 — Crisis identification (priority order):
      1. credit_spread in features → state with highest mean credit_spread
      2. vix_eom in features       → state with highest mean VIX
      3. fallback                  → state with lowest mean sp500_log_return

    Step 2 — Remaining 2 states ranked by pseudo-Sharpe
             (mean / std of sp500_log_return, in-sample):
      highest Sharpe → "Low-Vol Expansion"
      lower   Sharpe → "High-Beta Bull"

    Returns:
      state_to_label : {raw_state_int: label_str}
      means_orig     : DataFrame (n_states x n_features), original scale
    """
    raw_is      = model.predict(X_is)            # integer states for IS data
    sp_arr      = df_is["sp500_log_return"].values

    state_stats = {}
    for s in range(N_STATES):
        mask    = (raw_is == s)
        sp_vals = sp_arr[mask]

        mean_sp = sp_vals.mean() if mask.sum() > 0 else 0.0
        std_sp  = sp_vals.std()  if mask.sum() > 1 else 1e-8
        std_sp  = max(std_sp, 1e-8)                 # guard against zero
        sharpe_sp = mean_sp / std_sp

        credit_arr = df_is["credit_spread"].values[mask] if "credit_spread" in df_is.columns else np.array([])
        vix_arr    = df_is["vix_eom"].values[mask]       if "vix_eom"       in df_is.columns else np.array([])

        state_stats[s] = {
            "mean_sp500":   mean_sp,
            "std_sp500":    std_sp,
            "sharpe_sp500": sharpe_sp,
            "mean_credit":  credit_arr.mean() if len(credit_arr) > 0 else np.nan,
            "mean_vix":     vix_arr.mean()    if len(vix_arr)    > 0 else np.nan,
        }

    states = list(range(N_STATES))

    # --- Step 1: find crisis state ---
    if "credit_spread" in features:
        crisis = max(states, key=lambda s: state_stats[s]["mean_credit"])
    elif "vix_eom" in features:
        crisis = max(states, key=lambda s: state_stats[s]["mean_vix"])
    else:
        crisis = min(states, key=lambda s: state_stats[s]["mean_sp500"])

    # --- Step 2: rank remaining 2 by Sharpe (descending) ---
    remaining = sorted(
        [s for s in states if s != crisis],
        key=lambda s: state_stats[s]["sharpe_sp500"],
        reverse=True,
    )

    state_to_label = {
        remaining[0]: "Low-Vol Expansion",   # highest Sharpe
        remaining[1]: "High-Beta Bull",       # lower Sharpe
        crisis:        "Crisis",
    }

    means_orig_arr = scaler.inverse_transform(model.means_)
    means_orig     = pd.DataFrame(means_orig_arr, columns=features)
    return state_to_label, means_orig


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLASSIFY REGIMES (VITERBI)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regimes(model, X_is, X_oos, state_to_label):
    """
    Viterbi decoding on IS and OOS (OOS uses IS-trained params, no refit).
    Returns Series of label strings.
    """
    raw_is  = model.predict(X_is)
    raw_oos = model.predict(X_oos)

    labels_is  = [state_to_label[s] for s in raw_is]
    labels_oos = [state_to_label[s] for s in raw_oos]

    return (np.array(raw_is),  labels_is,
            np.array(raw_oos), labels_oos)


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPUTE MODEL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_stats(model, X_is, scaler, features,
                        raw_is, labels_is, raw_oos, labels_oos,
                        df_is, df_oos, master, state_to_label, converged, logL):
    """
    Returns a dict with all statistics needed for outputs and comparison.
    """
    n_features = len(features)
    n_train    = len(X_is)
    n_oos      = len(raw_oos)
    k          = _n_params(N_STATES, n_features)

    # AIC / BIC
    aic = -2 * logL + 2 * k
    bic = -2 * logL + k * math.log(n_train)

    # Transition matrix (labeled)
    transmat = model.transmat_
    durations = _mean_duration(transmat, state_to_label)

    # Means / stds in original scale
    means_orig_arr = scaler.inverse_transform(model.means_)
    # covars_ can be (n_states, n_features) or (n_states, n_features, n_features)
    # depending on hmmlearn version; extract diagonal variances in either case
    if model.covars_.ndim == 3:
        variances = np.array([np.diag(model.covars_[s]) for s in range(N_STATES)])
    else:
        variances = model.covars_
    stds_orig_arr = np.sqrt(variances) * scaler.scale_  # (n_states, n_features)

    # IS regime summary
    df_is_aug = df_is.copy()
    df_is_aug["regime_raw"]   = raw_is
    df_is_aug["regime_label"] = labels_is

    regime_stats_is = {}
    for label in LABEL_ORDER:
        sub = df_is_aug[df_is_aug["regime_label"] == label]
        regime_stats_is[label] = {
            "n_months": len(sub),
            "share":    len(sub) / n_train,
            **{f"mean_{f}": sub[f].mean() for f in ALL_FEATURES if f in df_is.columns},
            **{f"std_{f}":  sub[f].std()  for f in ALL_FEATURES if f in df_is.columns},
        }

    # OOS regime summary
    df_oos_aug = df_oos.copy()
    df_oos_aug["regime_raw"]   = raw_oos
    df_oos_aug["regime_label"] = labels_oos

    regime_stats_oos = {}
    for label in LABEL_ORDER:
        sub = df_oos_aug[df_oos_aug["regime_label"] == label]
        regime_stats_oos[label] = {
            "n_months": len(sub),
            "share":    len(sub) / n_oos if n_oos > 0 else 0,
            **{f"mean_{f}": sub[f].mean() for f in ALL_FEATURES if f in df_oos.columns},
            **{f"std_{f}":  sub[f].std()  for f in ALL_FEATURES if f in df_oos.columns},
        }

    # Min IS regime share (for comparison table warning)
    min_share = min(v["share"] for v in regime_stats_is.values())

    # Max self-transition
    max_self = max(transmat[s, s] for s in range(N_STATES))

    # Interpretation note
    notes = []
    if min_share < 0.05:
        notes.append("WARNING: at least one regime has <5% IS share")
    for s in range(N_STATES):
        if transmat[s, s] >= 0.999:
            notes.append(f"WARNING: state {state_to_label[s]} has p_ii≥0.999 → duration unstable")
    if n_features == 1:
        notes.append("Single-variable model: regime labeling based on returns only")
    interpretation_note = "; ".join(notes) if notes else "OK"

    return {
        "n_features": n_features,
        "n_train": n_train,
        "n_oos": n_oos,
        "k": k,
        "logL": logL,
        "aic": aic,
        "bic": bic,
        "converged": converged,
        "transmat": transmat,
        "durations": durations,
        "state_to_label": state_to_label,
        "means_orig_arr": means_orig_arr,
        "stds_orig_arr": stds_orig_arr,
        "regime_stats_is": regime_stats_is,
        "regime_stats_oos": regime_stats_oos,
        "df_is_aug": df_is_aug,
        "df_oos_aug": df_oos_aug,
        "min_share": min_share,
        "max_self": max_self,
        "interpretation_note": interpretation_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE MODEL EXCEL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_model_excel_outputs(model_id, spec, stats, df_is, df_oos, X_is, X_oos, scaler):
    features   = spec["features"]
    out_dir    = os.path.join(OUTPUT_DIR, model_id)
    s2l        = stats["state_to_label"]
    transmat   = stats["transmat"]
    durations  = stats["durations"]
    df_is_aug  = stats["df_is_aug"]
    df_oos_aug = stats["df_oos_aug"]

    # ── 01 Model Input ────────────────────────────────────────
    df_in = pd.concat([df_is_aug[["month","sample_period"] + features],
                       df_oos_aug[["month","sample_period"] + features]])
    # Add z_ columns
    X_all = np.vstack([X_is, X_oos])
    for j, f in enumerate(features):
        df_in[f"z_{f}"] = X_all[:, j]
    df_in["month"] = df_in["month"].astype(str)
    df_in.to_excel(os.path.join(out_dir, "01_hmm_model_input.xlsx"), index=False)

    # ── 02 IS Regime Classification ───────────────────────────
    is_out = df_is_aug[["month","sample_period"] + features + ["regime_raw","regime_label"]].copy()
    is_out["month"] = is_out["month"].astype(str)
    is_out.to_excel(os.path.join(out_dir, "02_insample_regime_classification.xlsx"), index=False)

    # ── 03 OOS Viterbi Regime Classification ──────────────────
    oos_out = df_oos_aug[["month","sample_period"] + features + ["regime_raw","regime_label"]].copy()
    oos_out["month"] = oos_out["month"].astype(str)
    oos_out.to_excel(os.path.join(out_dir, "03_oos_viterbi_regime_classification.xlsx"), index=False)

    # ── 04 Regime Summary Statistics ─────────────────────────
    def _make_summary_df(regime_stats, used_features, all_feats):
        rows_used, rows_all = [], []
        for label in LABEL_ORDER:
            if label not in regime_stats:
                continue
            st = regime_stats[label]
            row_used = {"regime_label": label, "n_months": st["n_months"],
                        "share_pct": round(st["share"] * 100, 2)}
            row_all  = {"regime_label": label, "n_months": st["n_months"],
                        "share_pct": round(st["share"] * 100, 2)}
            for f in used_features:
                row_used[f"mean_{f}"] = round(st.get(f"mean_{f}", np.nan), 6)
                row_used[f"std_{f}"]  = round(st.get(f"std_{f}",  np.nan), 6)
            for f in all_feats:
                row_all[f"mean_{f}"] = round(st.get(f"mean_{f}", np.nan), 6)
                row_all[f"std_{f}"]  = round(st.get(f"std_{f}",  np.nan), 6)
            rows_used.append(row_used)
            rows_all.append(row_all)
        return pd.DataFrame(rows_used), pd.DataFrame(rows_all)

    df_is_used,  df_is_all  = _make_summary_df(stats["regime_stats_is"],  features, ALL_FEATURES)
    df_oos_used, df_oos_all = _make_summary_df(stats["regime_stats_oos"], features, ALL_FEATURES)

    with pd.ExcelWriter(os.path.join(out_dir, "04_regime_summary_statistics.xlsx"),
                        engine="openpyxl") as xw:
        df_is_used.to_excel( xw, sheet_name="insample_used_features",        index=False)
        df_is_all.to_excel(  xw, sheet_name="insample_all_features_diag",    index=False)
        df_oos_used.to_excel(xw, sheet_name="oos_used_features",             index=False)
        df_oos_all.to_excel( xw, sheet_name="oos_all_features_diag",         index=False)

    # ── 05 Transition Matrix ──────────────────────────────────
    labels_ordered = [s2l[s] for s in range(N_STATES)]
    tm_df = pd.DataFrame(transmat, index=labels_ordered, columns=labels_ordered).round(6)
    dur_df = pd.DataFrame([
        {"regime_label": lbl,
         "p_ii": round(transmat[s, s], 6),
         "mean_duration_months": round(durations[lbl], 2) if not np.isnan(durations[lbl]) else "NaN (p_ii≥0.999)",
        }
        for s, lbl in s2l.items()
    ])
    with pd.ExcelWriter(os.path.join(out_dir, "05_transition_matrix.xlsx"),
                        engine="openpyxl") as xw:
        tm_df.to_excel(xw, sheet_name="transition_matrix")
        dur_df.to_excel(xw, sheet_name="mean_duration", index=False)

    # 06_hmm_model_parameters.xlsx is saved by _save_model_params_with_model()
    # which has direct access to the model object (needed for start probs)


def _save_model_params_with_model(model_id, model, spec, stats, scaler):
    """Separate pass to save start probs properly."""
    features = spec["features"]
    out_dir  = os.path.join(OUTPUT_DIR, model_id)
    s2l      = stats["state_to_label"]
    transmat = stats["transmat"]

    param_rows = []
    for s in range(N_STATES):
        label = s2l[s]
        for j, f in enumerate(features):
            param_rows.append({
                "state_raw": s,
                "regime_label": label,
                "feature": f,
                "emission_mean_orig_scale": round(float(stats["means_orig_arr"][s, j]), 6),
                "emission_std_orig_scale":  round(float(stats["stds_orig_arr"][s, j]),  6),
            })

    startprob_df = pd.DataFrame({
        "state_raw":        list(range(N_STATES)),
        "regime_label":     [s2l[s] for s in range(N_STATES)],
        "start_probability": [round(float(p), 6) for p in model.startprob_],
    })

    meta_df = pd.DataFrame([{
        "model_id":        model_id,
        "feature_list":    str(features),
        "n_features":      len(features),
        "n_states":        N_STATES,
        "covariance_type": "diag",
        "n_iter":          N_ITER,
        "converged":       stats["converged"],
        "log_likelihood":  round(stats["logL"], 4),
        "n_parameters":    stats["k"],
        "AIC":             round(stats["aic"], 4),
        "BIC":             round(stats["bic"], 4),
        "n_train":         stats["n_train"],
    }])

    labels_ordered = [s2l[s] for s in range(N_STATES)]
    tm_df = pd.DataFrame(transmat, index=labels_ordered, columns=labels_ordered).round(6)

    with pd.ExcelWriter(os.path.join(out_dir, "06_hmm_model_parameters.xlsx"),
                        engine="openpyxl") as xw:
        meta_df.to_excel(       xw, sheet_name="model_info",      index=False)
        startprob_df.to_excel(  xw, sheet_name="start_probs",     index=False)
        pd.DataFrame(param_rows).to_excel(xw, sheet_name="emission_params", index=False)
        tm_df.to_excel(         xw, sheet_name="transition_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def make_model_figures(model_id, spec, stats):
    features   = spec["features"]
    fig_dir    = os.path.join(FIGURE_DIR, model_id)
    s2l        = stats["state_to_label"]
    df_is_aug  = stats["df_is_aug"]
    df_oos_aug = stats["df_oos_aug"]
    transmat   = stats["transmat"]

    df_is_aug  = df_is_aug.copy()
    df_oos_aug = df_oos_aug.copy()
    df_is_aug["month_dt"]  = df_is_aug["month"].dt.to_timestamp()
    df_oos_aug["month_dt"] = df_oos_aug["month"].dt.to_timestamp()

    # ── fig_01: IS regime timeline ───────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    _plot_regime_timeline(ax, df_is_aug, s2l, model_id, "In-Sample (1990–2018)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_01_insample_regime_timeline.png"), dpi=300)
    plt.close(fig)

    # ── fig_02: OOS regime timeline ──────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    _plot_regime_timeline(ax, df_oos_aug, s2l, model_id, "OOS Viterbi (2019–2026)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_02_oos_viterbi_regime_timeline.png"), dpi=300)
    plt.close(fig)

    # ── fig_03: Regime variable means ────────────────────────
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 5))
    if len(features) == 1:
        axes = [axes]
    means_df = pd.DataFrame(stats["means_orig_arr"], columns=features)
    means_df["label"] = [s2l[s] for s in range(N_STATES)]
    means_df = means_df.set_index("label").reindex(
        [l for l in LABEL_ORDER if l in means_df.index]
    )
    FEAT_NAMES = {
        "sp500_log_return": "S&P500 Log Ret",
        "vix_eom":          "VIX (EOM)",
        "term_spread":      "Term Spread\n(10Y-3M, %)",
        "credit_spread":    "Credit Spread\n(BAA-AAA, %)",
    }
    for ax, f in zip(axes, features):
        colors = [REGIME_COLORS.get(lbl, "#888") for lbl in means_df.index]
        bars = ax.bar(means_df.index, means_df[f], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(FEAT_NAMES.get(f, f), fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean (original scale)", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(f"{model_id}: Regime Mean by Feature (IS)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_03_regime_variable_means.png"), dpi=300)
    plt.close(fig)

    # ── fig_04: Transition matrix heatmap ────────────────────
    labels_ordered = [s2l[s] for s in range(N_STATES)]
    tm_df = pd.DataFrame(transmat, index=labels_ordered, columns=labels_ordered)
    # Reorder for readability
    order = [l for l in LABEL_ORDER if l in labels_ordered]
    tm_df = tm_df.reindex(index=order, columns=order)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(tm_df, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5,
                vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Transition Probability"})
    ax.set_title(f"{model_id}: Transition Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("To Regime");  ax.set_ylabel("From Regime")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_04_transition_matrix_heatmap.png"), dpi=300)
    plt.close(fig)


def _plot_regime_timeline(ax, df, s2l, model_id, subtitle):
    """Background bands + S&P500 log return line."""
    dates  = df["month_dt"].values
    ret    = df["sp500_log_return"].values
    labels = df["regime_label"].values

    # Background bands
    prev_idx = 0
    for i in range(1, len(dates)):
        if labels[i] != labels[i - 1] or i == len(dates) - 1:
            end_idx = i if labels[i] != labels[i - 1] else len(dates) - 1
            ax.axvspan(dates[prev_idx], dates[min(end_idx, len(dates) - 1)],
                       color=REGIME_COLORS.get(labels[prev_idx], "#ccc"), alpha=0.3)
            prev_idx = i

    ax.plot(dates, ret, color="black", linewidth=1.0, label="SP500 log return")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    patches = [mpatches.Patch(color=c, alpha=0.5, label=l)
               for l, c in REGIME_COLORS.items()]
    ax.legend(handles=patches + [mpatches.Patch(color="black", label="SP500 log ret")],
              loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"{model_id} — {subtitle}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month");  ax.set_ylabel("Log Return")
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 11. MODEL MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_model_markdown(model_id, spec, stats):
    features = spec["features"]
    s2l      = stats["state_to_label"]
    dur      = stats["durations"]
    rs_is    = stats["regime_stats_is"]
    rs_oos   = stats["regime_stats_oos"]

    def _dur_str(label):
        d = dur.get(label, np.nan)
        if d is None or np.isnan(d):
            return "NaN (p_ii>=0.999)"
        return f"{d:.1f} months"

    # Pre-compute text blocks (avoid backslash inside f-string expressions)
    desc_block = "\n".join(
        "- **{lbl}**: {n} months ({sh:.1f}%)  |  {ms}".format(
            lbl=lbl,
            n=rs_is.get(lbl, {}).get("n_months", 0),
            sh=rs_is.get(lbl, {}).get("share", 0) * 100,
            ms="  ".join(
                "{f}={v:.4f}".format(f=f, v=rs_is.get(lbl, {}).get("mean_" + f, float("nan")))
                for f in features
            ),
        )
        for lbl in LABEL_ORDER
    )

    oos_block = "\n".join(
        "- **{lbl}**: {n} months ({sh:.1f}%)".format(
            lbl=lbl,
            n=rs_oos.get(lbl, {}).get("n_months", 0),
            sh=rs_oos.get(lbl, {}).get("share", 0) * 100,
        )
        for lbl in LABEL_ORDER
    )

    col_headers = " | ".join(LABEL_ORDER)
    transmat_header = "| From / To | " + col_headers + " |"
    sep_cols        = " | ".join("-" * max(len(l), 5) for l in LABEL_ORDER)
    transmat_sep    = "| " + "-" * 9 + " | " + sep_cols + " |"
    transmat_rows   = []
    for from_l in LABEL_ORDER:
        from_s = next(s for s, l in s2l.items() if l == from_l)
        cells  = " | ".join(
            "{:.3f}".format(stats["transmat"][from_s, next(s for s, l in s2l.items() if l == to_l)])
            for to_l in LABEL_ORDER
        )
        transmat_rows.append("| {} | {} |".format(from_l, cells))
    transmat_block = "\n".join([transmat_header, transmat_sep] + transmat_rows)

    if "credit_spread" in features:
        label_note = "Crisis 식별: credit_spread 최고 레짐. 나머지 2개는 S&P 500 유사 샤프 비율(mean/std) 기준 정렬."
    elif "vix_eom" in features:
        label_note = "Crisis 식별: VIX 최고 레짐(credit_spread 없음). 나머지 2개는 S&P 500 유사 샤프 비율 기준 정렬."
    else:
        label_note = "단일 변수 모델: S&P 500 평균 최저 레짐을 Crisis로 지정. 나머지 2개는 샤프 비율 기준 정렬."
    limit_note5 = (
        "단일 변수 모델은 레짐 식별력이 낮을 수 있으며, 경제적 해석이 제한적임."
        if len(features) == 1
        else "입력변수 간 상관관계가 높으면 diagonal covariance 가정이 부적절할 수 있음."
    )

    lines = [
        "# HMM Regime Model Result Summary",
        "## Model: " + model_id,
        "",
        "---",
        "",
        "### 1. 분석 목적",
        "",
        "미국 시장의 월별 거시·시장 변수를 이용해 3-state Gaussian HMM으로 시장 레짐을 분류하고,",
        "입력변수 조합별 분류 결과를 비교한다.",
        "",
        "---",
        "",
        "### 2. 사용 입력변수",
        "",
        "- Model ID: `" + model_id + "`",
        "- Model Name: " + spec["name"],
        "- Features ({}개): {}".format(len(features), ", ".join(features)),
        "",
        "---",
        "",
        "### 3. 전처리 방식",
        "",
        "- 4개 데이터를 월별(Year-Month Period) 기준으로 inner join 병합",
        "- BaaAaa_Spread는 일별 데이터를 월말 마지막 관측값으로 변환",
        "- 결측치 포함 행 제거",
        "- 각 모델별 입력변수에 대해 StandardScaler 적용 (in-sample 기준 fit, OOS는 transform만)",
        "",
        "---",
        "",
        "### 4. In-sample / OOS 기간",
        "",
        "- In-sample  : 1990-01 ~ 2018-12  ({}개월)".format(stats["n_train"]),
        "- OOS        : 2019-01 ~ latest   ({}개월)".format(stats["n_oos"]),
        "- 모델 학습은 in-sample에서만 수행",
        "- OOS 구간에서는 모델 재학습 없이 학습된 파라미터로 Viterbi 분류만 수행",
        "",
        "---",
        "",
        "### 5. HMM 모델 설정",
        "",
        "- n_components = {}".format(N_STATES),
        "- covariance_type = diag",
        "- n_iter = {}".format(N_ITER),
        "- n_inits = {} (최적 log-likelihood 선택)".format(N_INITS),
        "- random_state = {}".format(RANDOM_SEED),
        "- 수렴 여부: {}".format(stats["converged"]),
        "",
        "---",
        "",
        "### 6. 레짐별 경제적 해석 (In-sample 기준)",
        "",
        desc_block,
        "",
        "**레짐 라벨링 기준:**",
        "① crisis_spread/VIX/최저수익률 기준으로 Crisis 레짐 식별 → ② 나머지 2개를 S&P 500 유사 샤프 비율(mean/std)로 정렬 → 최고 Sharpe = Low-Vol Expansion, 차순위 = High-Beta Bull.",
        label_note,
        "",
        "---",
        "",
        "### 7. 전이확률행렬",
        "",
        transmat_block,
        "",
        "**평균 지속기간:**",
    ] + ["- {:<20}: {}".format(lbl, _dur_str(lbl)) for lbl in LABEL_ORDER] + [
        "",
        "---",
        "",
        "### 8. AIC / BIC",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        "| Log Likelihood | {:.4f} |".format(stats["logL"]),
        "| 파라미터 수 (k) | {} |".format(stats["k"]),
        "| AIC | {:.4f} |".format(stats["aic"]),
        "| BIC | {:.4f} |".format(stats["bic"]),
        "| n (IS 관측치) | {} |".format(stats["n_train"]),
        "",
        "AIC/BIC는 낮을수록 좋지만, 단독으로 최종 모델을 선택하면 안 됨.",
        "레짐 해석 가능성, 관측 비중, 전이확률 안정성을 함께 고려해야 함.",
        "",
        "---",
        "",
        "### 9. OOS Viterbi 레짐 분류 결과 요약",
        "",
        oos_block,
        "",
        "**주의:** OOS Viterbi 결과는 OOS 전체 관측값을 이용해 사후적으로 hidden state sequence를 추정한 것이다.",
        "실시간 투자 예측과는 구분해야 한다.",
        "",
        "---",
        "",
        "### 10. 한계점",
        "",
        "1. HMM state 번호는 임의적이며, credit_spread/VIX/수익률 기준으로 Crisis 식별 후 샤프 비율 기준으로 Low-Vol Expansion · High-Beta Bull을 재부여함.",
        "2. OOS Viterbi는 사후적 분류이므로 실시간 투자 신호로 사용 불가.",
        "   - 실시간 투자전략에는 filtered posterior probability 또는 rolling/sequential updating이 필요.",
        "3. Diagonal covariance 가정으로 변수 간 공분산을 무시함.",
        "4. HMM은 변수들이 가우시안 분포를 따른다고 가정하지만, 금융 시계열은 fat tail을 가질 수 있음.",
        "5. " + limit_note5,
        "",
        "---",
        "",
        "### 11. 이후 확장 가능성",
        "",
        "- Filtered posterior를 이용한 실시간 레짐 예측 및 투자전략 구성",
        "- 레짐별 섹터 ETF 성과 분석 및 동적 포트폴리오 백테스트",
        "- Full covariance 또는 tied covariance 모델과 비교",
        "- Regime-switching GARCH 모델과 결합",
        "- BIC 최적 n_states 탐색 (n=2,3,4 비교)",
    ]

    path = os.path.join(OUTPUT_DIR, model_id, "07_hmm_result_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 12. RUN SINGLE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def run_single_model(model_id, spec, df_is, df_oos, master):
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_id}  —  {spec['name']}")
    print(f"{'='*60}")
    features = spec["features"]
    print(f"  Features: {features}")

    # Scale
    X_is, X_oos, scaler = scale_features(df_is, df_oos, features)

    # Fit
    model, converged, logL = fit_hmm(X_is, model_id)

    # Label
    state_to_label, means_orig = label_regimes(model, X_is, df_is, scaler, features)
    print(f"  Regime labels: {state_to_label}")

    # Classify
    raw_is, labels_is, raw_oos, labels_oos = classify_regimes(
        model, X_is, X_oos, state_to_label
    )

    # Stats
    stats = compute_model_stats(
        model, X_is, scaler, features,
        raw_is, labels_is, raw_oos, labels_oos,
        df_is, df_oos, master, state_to_label, converged, logL
    )

    print(f"  logL={stats['logL']:.3f}  AIC={stats['aic']:.2f}  BIC={stats['bic']:.2f}")
    print(f"  IS regime shares: " +
          "  ".join([f"{l}={stats['regime_stats_is'].get(l,{}).get('share',0)*100:.1f}%"
                     for l in LABEL_ORDER]))

    # Save outputs
    save_model_excel_outputs(model_id, spec, stats, df_is, df_oos, X_is, X_oos, scaler)
    _save_model_params_with_model(model_id, model, spec, stats, scaler)
    make_model_figures(model_id, spec, stats)
    write_model_markdown(model_id, spec, stats)

    print(f"  Saved outputs/{model_id}/ and figures/{model_id}/")

    return {
        "model_id":       model_id,
        "name":           spec["name"],
        "n_features":     len(features),
        "feature_list":   ", ".join(features),
        "n_train":        stats["n_train"],
        "n_oos":          stats["n_oos"],
        "converged":      converged,
        "log_likelihood": round(stats["logL"], 4),
        "n_parameters":   stats["k"],
        "AIC":            round(stats["aic"], 4),
        "BIC":            round(stats["bic"], 4),
        "min_regime_share_insample": round(stats["min_share"], 4),
        "max_self_transition_prob":  round(stats["max_self"], 4),
        "mean_duration_by_state":    str({k: (round(v,1) if not np.isnan(v) else "NaN")
                                         for k, v in stats["durations"].items()}),
        "interpretation_note":       stats["interpretation_note"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 13. COMPARISON OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_outputs(records, master, df_is, df_oos):
    print("\n[C] Saving comparison outputs...")

    comp_df = pd.DataFrame(records)
    comp_df = comp_df.sort_values("AIC").reset_index(drop=True)
    comp_df["AIC_rank"] = comp_df["AIC"].rank().astype(int)
    comp_df["BIC_rank"] = comp_df["BIC"].rank().astype(int)

    # ── 00_model_comparison_aic_bic.xlsx ─────────────────────
    comp_df.to_excel(
        os.path.join(OUTPUT_DIR, "00_model_comparison_aic_bic.xlsx"), index=False
    )

    # ── 00_hmm_master_dataset.xlsx ───────────────────────────
    is_end_p = pd.Period(IS_END_YM, "M")
    master_with_sp = master.copy()
    master_with_sp["sample_period"] = np.where(
        master_with_sp["month"] <= is_end_p, "in_sample", "out_of_sample"
    )
    master_out = master_with_sp[["month"] + ALL_FEATURES + ["sample_period"]].copy()
    master_out["month"] = master_out["month"].astype(str)
    master_out.to_excel(
        os.path.join(OUTPUT_DIR, "00_hmm_master_dataset.xlsx"), index=False
    )

    # ── 00_model_specifications.xlsx ─────────────────────────
    spec_rows = []
    for mid, spec in MODEL_SPECS.items():
        spec_rows.append({
            "model_id":        mid,
            "model_name":      spec["name"],
            "n_features":      len(spec["features"]),
            "feature_list":    ", ".join(spec["features"]),
            "n_states":        N_STATES,
            "covariance_type": "diag",
            "in_sample_start": "1990-01",
            "in_sample_end":   IS_END_YM,
            "oos_start":       OOS_START_YM,
            "oos_end":         str(df_oos["month"].max()),
        })
    pd.DataFrame(spec_rows).to_excel(
        os.path.join(OUTPUT_DIR, "00_model_specifications.xlsx"), index=False
    )

    # ── 00_model_comparison_summary.md ───────────────────────
    _write_comparison_markdown(comp_df, df_is, df_oos)

    # ── fig_00 AIC/BIC comparison bar chart ──────────────────
    _make_comparison_figure(comp_df)

    print("  Saved: 00_hmm_master_dataset.xlsx")
    print("  Saved: 00_model_specifications.xlsx")
    print("  Saved: 00_model_comparison_aic_bic.xlsx")
    print("  Saved: 00_model_comparison_summary.md")
    print("  Saved: figures/fig_00_model_comparison_aic_bic.png")


def _write_comparison_markdown(comp_df, df_is, df_oos):
    aic_best = comp_df.loc[comp_df["AIC_rank"] == 1, "model_id"].values[0]
    bic_best = comp_df.loc[comp_df["BIC_rank"] == 1, "model_id"].values[0]
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Pre-build all table rows as plain strings (no backslash in expressions)
    model_tbl_rows = "\n".join(
        "| {} | {} |".format(mid, spc["name"])
        for mid, spc in MODEL_SPECS.items()
    )

    aic_tbl_header = "| Model | n_feat | logL | AIC | BIC | AIC_rank | BIC_rank |"
    aic_tbl_sep    = "|-------|--------|------|-----|-----|----------|----------|"
    aic_tbl_rows   = "\n".join(
        "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {} | {} |".format(
            row["model_id"], row["n_features"],
            row["log_likelihood"], row["AIC"], row["BIC"],
            row["AIC_rank"], row["BIC_rank"],
        )
        for _, row in comp_df.sort_values("AIC_rank").iterrows()
    )
    aic_tbl_block = "\n".join([aic_tbl_header, aic_tbl_sep, aic_tbl_rows])

    lines = [
        "# HMM 입력변수 조합 비교 요약 리포트",
        "",
        "생성일: " + now_str,
        "",
        "---",
        "",
        "## 1. 분석 목적",
        "",
        "미국 시장의 거시·시장 변수를 이용해 3-state Gaussian HMM으로 시장 레짐을 분류하고,",
        "총 8개의 입력변수 조합에 대해 모델 적합도(AIC, BIC) 및 레짐 분류 결과를 비교한다.",
        "",
        "---",
        "",
        "## 2. 비교한 모델 목록",
        "",
        "| Model ID | 변수 구성 |",
        "|----------|---------|",
        model_tbl_rows,
        "",
        "---",
        "",
        "## 3. AIC / BIC 기준 모델 비교",
        "",
        aic_tbl_block,
        "",
        "- **AIC 최저 모델**: `" + aic_best + "`",
        "- **BIC 최저 모델**: `" + bic_best + "`",
        "",
        "> **주의:** AIC/BIC가 낮을수록 상대적으로 선호되지만, 이것만으로 최종 모델을 기계적으로 선택하면 안 된다.",
        "> 입력변수가 줄어들면 파라미터 수가 감소해 BIC에서 유리할 수 있다.",
        "> 반대로 변수가 많으면 적합도는 높아지나 과적합 가능성이 있다.",
        "",
        "---",
        "",
        "## 4. 레짐 해석 가능성 요약",
        "",
        "각 모델의 레짐 라벨은 in-sample 구간에서 다음 기준으로 부여됨:",
        "① credit_spread 최고 레짐 → Crisis (credit_spread 없으면 VIX 최고 or sp500 최저)",
        "② 나머지 2개 레짐을 S&P 500 유사 샤프 비율(mean/std) 기준 정렬 → Low-Vol Expansion / High-Beta Bull.",
        "",
        "- 4변수·3변수 모델: credit_spread로 Crisis를 명확히 식별; Sharpe 기준으로 두 Bull 국면을 구분",
        "- 2변수(sp500+VIX): VIX로 Crisis 식별; 나머지는 Sharpe 기준",
        "- 2변수(sp500+term): 수익률 곡선 기울기 활용",
        "- 2변수(sp500+credit): 신용위험 반영",
        "- **단일변수(M1_sp500)**: 수익률 분포만으로 레짐 분류, 경제적 해석 가장 제한적",
        "",
        "---",
        "",
        "## 5. 전이확률 및 평균 지속기간 비교",
        "",
        "각 모델별 상세 전이확률과 평균 지속기간은",
        "`outputs/{model_id}/05_transition_matrix.xlsx` 및",
        "`outputs/{model_id}/07_hmm_result_summary.md` 참조.",
        "",
        "---",
        "",
        "## 6. OOS Viterbi 결과 비교 요약",
        "",
        "각 모델의 OOS 레짐 분류 결과는 `outputs/{model_id}/03_oos_viterbi_regime_classification.xlsx` 참조.",
        "타임라인 그래프는 `figures/{model_id}/fig_02_oos_viterbi_regime_timeline.png` 참조.",
        "",
        "> **중요:** OOS Viterbi 결과는 OOS 전체 관측값을 이용해 **사후적으로** hidden state sequence를",
        "> 추정한 것이다. 실시간 투자 예측과 구분해야 한다.",
        "",
        "---",
        "",
        "## 7. 주의가 필요한 모델",
        "",
        "- min_regime_share < 5%인 모델: 특정 레짐에 관측치가 너무 적어 통계적 추정이 불안정",
        "- p_ii >= 0.999인 모델: 자기전이확률이 너무 높아 지속기간 계산 불안정",
        "- 상세 경고는 `00_model_comparison_aic_bic.xlsx`의 `interpretation_note` 컬럼 참조",
        "",
        "---",
        "",
        "## 8. 한계점",
        "",
        "1. OOS Viterbi는 사후적 분류이며, 실시간 투자 신호로 직접 사용 불가",
        "   -> 실투에는 filtered posterior 또는 sequential updating 필요",
        "2. 이번 분석은 레짐 분류에만 집중하며, ETF 성과분석·백테스트는 수행하지 않음",
        "3. Diagonal covariance 가정으로 변수 간 공분산 무시",
        "4. 모든 모델은 동일한 master dataset의 교집합 기간을 사용",
        "",
        "---",
        "",
        "## 9. 이후 확장 가능성",
        "",
        "- Filtered posterior 기반 실시간 레짐 예측 및 6월 2026 국면 예측",
        "- 레짐별 섹터 ETF 성과 분석 및 동적 포트폴리오 구성",
        "- 최적 n_states 탐색 (BIC 기준 2/3/4 비교)",
        "- Regime-switching GARCH 모델과의 결합",
    ]

    path = os.path.join(OUTPUT_DIR, "00_model_comparison_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _make_comparison_figure(comp_df):
    """AIC/BIC side-by-side bar chart for all 8 models."""
    ordered = comp_df.sort_values("AIC_rank")
    model_ids = ordered["model_id"].tolist()
    aic_vals  = ordered["AIC"].tolist()
    bic_vals  = ordered["BIC"].tolist()

    x     = np.arange(len(model_ids))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, aic_vals, width, label="AIC", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width/2, bic_vals, width, label="BIC", color="#FF9800", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_ids, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Information Criterion (lower is better)", fontsize=11)
    ax.set_title("HMM Model Comparison: AIC vs BIC (All 8 Combinations)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate min values
    min_aic_idx = np.argmin(aic_vals)
    min_bic_idx = np.argmin(bic_vals)
    ax.bar(x[min_aic_idx] - width/2, aic_vals[min_aic_idx],
           width, color="#0D47A1", alpha=0.9, label="_AIC best")
    ax.bar(x[min_bic_idx] + width/2, bic_vals[min_bic_idx],
           width, color="#E65100", alpha=0.9, label="_BIC best")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig_00_model_comparison_aic_bic.png"), dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 14. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HMM Market Regime Model - 8 Variable Combinations")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Setup directories
    _make_dirs()

    # Load & preprocess
    sp, vx, ts, cs = load_data()
    master = preprocess_data(sp, vx, ts, cs)
    df_is, df_oos = split_sample(master)

    # Run all 8 models
    comparison_records = []
    for model_id, spec in MODEL_SPECS.items():
        try:
            record = run_single_model(model_id, spec, df_is, df_oos, master)
            comparison_records.append(record)
        except Exception as e:
            print(f"\n!!! ERROR in model {model_id}: {e}")
            raise

    # Comparison outputs
    save_comparison_outputs(comparison_records, master, df_is, df_oos)

    print("\n" + "=" * 60)
    print("  DONE: all model outputs and comparison files saved.")
    print(f"  outputs/ → {OUTPUT_DIR}")
    print(f"  figures/ → {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
