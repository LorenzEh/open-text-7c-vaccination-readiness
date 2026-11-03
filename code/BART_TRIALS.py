# ============================================================
# PyMC BART with expanded embeddings (already numeric columns)
# ============================================================

# ensure we always get similar results:
import os

# 1. Disable Python hash randomization (ensures dict/set iteration order is identical)
os.environ['PYTHONHASHSEED'] = '0'

# 2. Disable multithreading in underlying linear algebra libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 3. Restrict Numba (used by UMAP)
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import logging
import warnings

import umap
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

import numpy as np
import random


import paths

# Embeddings of the response text only (no question prepended); used for the
# robustness comparison, not for the reported models.
# data = pd.read_csv(paths.EMBEDDINGS)

data = pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)

data_cleaned = pd.read_csv(paths.MERGED_CLEANED)

# ------------------------------------------------------------
# Create masks from data_cleaned for all text columns
# ------------------------------------------------------------

# Identify all text columns that end with '_text' but not '_questionnair_text'
text_cols = [c for c in data_cleaned.columns if c.startswith('text_')]

MARKED_NUMBERS = [-1, 2, 3, 4, 5, 6, 7]   # -1 appears without parentheses

if MARKED_NUMBERS:
    # Build a list of regex fragments for each number
    terms = []
    for num in MARKED_NUMBERS:
        if num == -1:
            terms.append(r'-1')               # literal -1
        else:
            terms.append(r'\(' + str(num) + r'\)')   # e.g. \(2\), \(3\)
    # Combine them with | and anchor to the end of the string
    pattern = r'(?:' + '|'.join(terms) + r')$'
else:
    pattern = None

marked_masks = {}
for col in text_cols:
    if col in data_cleaned.columns:
        # Align with data's index
        cleaned_vals = data_cleaned[col].reindex(data.index, fill_value="").astype(str)
        if pattern:
            mask = cleaned_vals.str.contains(pattern, regex=True, na=False)
        else:
            mask = pd.Series(False, index=data.index)
        marked_masks[col] = mask
    else:
        print(f"Warning: column {col} not found in data_cleaned. No mask created.")
        marked_masks[col] = pd.Series(False, index=data.index)
        

for col in text_cols:
    if col not in data_cleaned.columns:
        continue
    print(f"\n--- Column: {col} ---")

    # Indices where the mask is True
    masked_idx = marked_masks[col][marked_masks[col]].index

    # Only proceed if there is at least one masked row
    if len(masked_idx) > 0:
        # Ensure indices are present in data_cleaned (optional, but safe)
        common_idx = masked_idx.intersection(data_cleaned.index)
        if len(common_idx) == 0:
            print("No masked rows found in data_cleaned (index mismatch?)")
            continue

        # Sample up to 10 masked rows
        sample_size = min(64, len(common_idx))
        # Use .sample on the index
        sample_idx = pd.Series(common_idx).sample(n=sample_size, random_state=42).values

        # Create a small DataFrame with text and mask
        sample_df = pd.DataFrame({
            'text': data_cleaned.loc[sample_idx, col],
            'masked': marked_masks[col].loc[sample_idx]
        })
        print(sample_df.to_string())
    else:
        print("No masked rows for this column.")

    # Also print counts
    total = len(data_cleaned)
    n_marked = marked_masks[col].sum()
    print(f"Marked rows: {n_marked} / {total} ({100*n_marked/total:.2f}%)")
    
    

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
N_SPLITS = 5
RANDOM_SEED = 42
MIN_WORDS = 1                               # minimum words in text column
REDUCTION_METHOD = 'pca'                    # 'umap' or 'pca'
PCA_VARIANCE_THRESHOLD = 0.95                 # e.g. 0.95; if not None, overrides grid
PCA_DIMS_GRID = [10, 20, 30, 50, 100]         # for PCA (used only if threshold is None)
UMAP_DIMS_GRID = [15, 20, 30]                 # as before
USE_MASK= True

# global random seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# Map target names to expanded embedding prefixes and full names
# ------------------------------------------------------------
target_to_text_map = {
    "conf": "text_confidence",
    "cmpcy": "text_complacency",
    "const": "text_constraints",
    "calc": "text_calculation",
    "colr": "text_coll_resp",
    "compli": "text_compliance",
    "consp": "text_conspiracy",
}

full_names = {
    "conf": "Confidence",
    "cmpcy": "Complacency",
    "const": "Constraints",
    "calc": "Calculation",
    "colr": "Coll. Responsibility",
    "compli": "Compliance",
    "consp": "Conspiracy",
}

selected_dimensions = None          # set to None to run all, or e.g. ["conf", "const"]
# selected_dimensions = ["conf"]

if selected_dimensions is not None:
    target_to_text_map = {k: v for k, v in target_to_text_map.items() if k in selected_dimensions}

# ------------------------------------------------------------
# PyMC BART configuration
# ------------------------------------------------------------
BART_PARAMS = {
    "m": 100,
    "alpha": 0.95,
    "beta": 2.0,
    "response": "mix",
}

SAMPLING_PARAMS = {
    "draws": 1000,
    "tune": 500,
    "chains": 2,
    "progressbar": False,
    "random_seed": RANDOM_SEED
}

# ------------------------------------------------------------
# BART CV 
# ------------------------------------------------------------
def run_pymc_bart(X, y, texts=None, method='umap',
                  umap_grid=UMAP_DIMS_GRID,
                  pca_grid=PCA_DIMS_GRID,
                  pca_variance_threshold=None):
    """
    X: feature matrix (n_samples, n_features)
    y: target vector
    texts: optional array of text strings (same length as y)
    method: 'umap' or 'pca'
    umap_grid: list of dimensions to try for UMAP
    pca_grid: list of components to try for PCA (used only if threshold is None)
    pca_variance_threshold: float (0-1) – if given, PCA uses enough components
                            to explain this fraction of variance; grid is ignored.
    Returns: (mean_model_rmse, mean_null_rmse, chosen_dims_list,
              y_true_all, y_pred_all, texts_all, best_dims_avg, final_idata_bart, final_idata_null)
    """
    outer_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    model_rmses = []
    null_rmses = []
    chosen_dims = []
    y_true_all = []
    y_pred_all = []
    texts_all = [] if texts is not None else None

    # We'll also collect the best dimensions across folds to later fit a final model
    all_best_dims = []

    for train_idx, test_idx in outer_kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if texts is not None:
            texts_test = texts[test_idx]

        # Null model
        null_pred = np.mean(y_train)
        null_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, null_pred)))
        null_rmses.append(null_rmse)

        # ---------------- Dimensionality reduction and hyperparameter selection ----------------
        if method == 'umap':
            dim_grid = umap_grid
            # Inner CV to choose best UMAP dimension
            param_to_rmse = {}
            for dim in dim_grid:
                inner_rmses = []
                for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                    X_tr, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
                    y_tr, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

                    reducer = umap.UMAP(
                        n_components=dim,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric="euclidean",
                        random_state=RANDOM_SEED,
                    )
                    X_tr_red = reducer.fit_transform(X_tr)
                    X_val_red = reducer.transform(X_val)

                    # Fit BART (simplified – we only need predictions for inner CV)
                    with pm.Model():
                        Xd = pm.Data("X", X_tr_red)
                        yd = pm.Data("y", y_tr)

                        mu = pmb.BART("mu", Xd, yd, **BART_PARAMS)
                        sigma = pm.HalfNormal("sigma", 5)
                        pm.Normal("obs", mu=mu, sigma=sigma, observed=yd)

                        idata = pm.sample(**SAMPLING_PARAMS)
                        pm.set_data({"X": X_val_red})
                        pp = pm.sample_posterior_predictive(idata, var_names=["mu"], progressbar=False, random_seed=RANDOM_SEED)

                    y_pred = pp.posterior_predictive["mu"].mean(dim=["chain", "draw"]).values
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    inner_rmses.append(rmse)

                param_to_rmse[dim] = np.mean(inner_rmses)

            best_dim = min(param_to_rmse, key=param_to_rmse.get)

        elif method == 'pca':
            if pca_variance_threshold is not None:
                # Use variance explained – determine components on the whole training set
                pca_temp = PCA(random_state=RANDOM_SEED)
                pca_temp.fit(X_train)
                cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
                best_dim = np.searchsorted(cumvar, pca_variance_threshold) + 1  # +1 because index 0 is first component
                # No inner CV needed
            else:
                # Use grid search via inner CV
                # Limit components to feasible numbers
                max_comp = min(X_train.shape[0], X_train.shape[1])
                dim_grid = [d for d in pca_grid if d <= max_comp]
                if not dim_grid:
                    dim_grid = [min(50, max_comp)]

                param_to_rmse = {}
                for dim in dim_grid:
                    inner_rmses = []
                    for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                        X_tr, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
                        y_tr, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

                        reducer = PCA(n_components=dim, random_state=RANDOM_SEED)
                        X_tr_red = reducer.fit_transform(X_tr)
                        X_val_red = reducer.transform(X_val)

                        # Fit BART
                        with pm.Model():
                            Xd = pm.Data("X", X_tr_red)
                            yd = pm.Data("y", y_tr)

                            mu = pmb.BART("mu", Xd, yd, **BART_PARAMS)
                            sigma = pm.HalfNormal("sigma", 5)
                            pm.Normal("obs", mu=mu, sigma=sigma, observed=yd)

                            idata = pm.sample(**SAMPLING_PARAMS)
                            pm.set_data({"X": X_val_red})
                            pp = pm.sample_posterior_predictive(idata, var_names=["mu"], progressbar=False, random_seed=RANDOM_SEED)

                        y_pred = pp.posterior_predictive["mu"].mean(dim=["chain", "draw"]).values
                        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                        inner_rmses.append(rmse)

                    param_to_rmse[dim] = np.mean(inner_rmses)

                best_dim = min(param_to_rmse, key=param_to_rmse.get)

        else:
            raise ValueError("method must be 'umap' or 'pca'")

        chosen_dims.append(best_dim)
        all_best_dims.append(best_dim)

        # ---------------- REFIT OUTER with best hyperparameter ----------------
        if method == 'umap':
            reducer = umap.UMAP(
                n_components=best_dim,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=RANDOM_SEED,
            )
            X_train_red = reducer.fit_transform(X_train)
            X_test_red = reducer.transform(X_test)
        else:  # PCA
            reducer = PCA(n_components=best_dim, random_state=RANDOM_SEED)
            X_train_red = reducer.fit_transform(X_train)
            X_test_red = reducer.transform(X_test)

        with pm.Model():
            Xd = pm.Data("X", X_train_red)
            yd = pm.Data("y", y_train)

            mu = pmb.BART("mu", Xd, yd, **BART_PARAMS)
            sigma = pm.HalfNormal("sigma", 5)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=yd)

            idata = pm.sample(**SAMPLING_PARAMS)
            pm.set_data({"X": X_test_red})
            pp = pm.sample_posterior_predictive(idata, var_names=["mu"], progressbar=False, random_seed=RANDOM_SEED)

        y_pred = pp.posterior_predictive["mu"].mean(dim=["chain", "draw"]).values
        model_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        # Store fold predictions and texts
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        if texts is not None:
            texts_all.extend(texts_test)

    # ========== FINAL MODEL ON ALL DATA (for LOO comparison) ==========
    # Use the average of chosen dimensions (rounded) as the final hyperparameter
    avg_best_dim = int(np.round(np.mean(all_best_dims)))
    if method == 'umap':
        final_reducer = umap.UMAP(
            n_components=avg_best_dim,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_SEED,
        )
    else:
        final_reducer = PCA(n_components=avg_best_dim, random_state=RANDOM_SEED)

    X_all_red = final_reducer.fit_transform(X)

    # Fit final BART model on all data
    with pm.Model() as final_model_bart:
        Xd = pm.Data("X", X_all_red)
        yd = pm.Data("y", y)

        mu = pmb.BART("mu", Xd, yd, **BART_PARAMS)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=yd)

        idata_bart = pm.sample(**SAMPLING_PARAMS, idata_kwargs={"log_likelihood": True})
        # sample posterior predictive to get log-likelihood for LOO
        pm.sample_posterior_predictive(idata_bart, extend_inferencedata=True, random_seed=RANDOM_SEED)

    # Fit null model (intercept only) on all data
    with pm.Model() as final_model_null:
        mu = pm.Normal("mu", mu=y.mean(), sigma=y.std()*2)   # weak prior
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        idata_null = pm.sample(**SAMPLING_PARAMS, idata_kwargs={"log_likelihood": True})
        pm.sample_posterior_predictive(idata_null, extend_inferencedata=True, random_seed=RANDOM_SEED)

    # Compute LOO comparison
    comparison = az.compare({"BART": idata_bart, "Null": idata_null}, ic="loo")

    # Return appropriate tuple
    if texts is not None:
        return (
            np.mean(model_rmses),
            np.mean(null_rmses),
            chosen_dims,
            np.array(y_true_all),
            np.array(y_pred_all),
            np.array(texts_all),
            avg_best_dim,
            idata_bart,
            idata_null,
            comparison
        )
    else:
        return (
            np.mean(model_rmses),
            np.mean(null_rmses),
            chosen_dims,
            np.array(y_true_all),
            np.array(y_pred_all),
            avg_best_dim,
            idata_bart,
            idata_null,
            comparison
        )

# ------------------------------------------------------------
# Main analysis loop 
# ------------------------------------------------------------
results = []
true_vals = {}
pred_vals = {}
text_vals = {}
loo_comparisons = {}

print(f"\nStarting PyMC-BART analysis with {REDUCTION_METHOD} (min_words={MIN_WORDS})...")
for target, text_col in tqdm(target_to_text_map.items(), desc="Targets"):

    # 1) Get rows with target and text column, drop NA
    temp_df = data[[target, text_col]].dropna()

    # --- Conditionally exclude marked rows ---
    if USE_MASK:
        mask_marked = marked_masks[text_col].loc[temp_df.index]
        n_marked = mask_marked.sum()
        if n_marked > 0:
            print(f"  For target '{target}', excluding {n_marked} rows because they are marked.")
            excluded_texts = temp_df.loc[mask_marked, text_col].head(3).tolist()
            if excluded_texts:
                print(f"  Example excluded texts: {excluded_texts}")
        temp_df = temp_df[~mask_marked]   # keep only unmarked rows
    else:
        # When not using mask, you might still want to know how many marked rows exist
        # but we don't filter them out. You can optionally print a count.
        if text_col in marked_masks:
            n_marked = marked_masks[text_col].loc[temp_df.index].sum()
            if n_marked > 0:
                print(f"  Note: {n_marked} marked rows exist for '{target}' but are being kept (USE_MASK=False).")

    # --- Optional stripping (commented out) ---
    # if target == "conf":
    #     temp_df[text_col] = temp_df[text_col].str.replace(r'\(8\)$', '', regex=True).str.strip()

    # 2) Filter by minimum words (unchanged)
    temp_df = temp_df[temp_df[text_col].astype(str).str.split().str.len() >= MIN_WORDS]

    if len(temp_df) < 10:
        print(f"Skipping {target}: insufficient rows after filtering ({len(temp_df)}).")
        continue

    # 3) Extract embedding columns (unchanged)
    emb_cols = [c for c in data.columns if c.startswith(f"{text_col}_emb_")]
    X = data.loc[temp_df.index, emb_cols].values.astype(np.float32)
    y = temp_df[target].values.astype(np.float32)
    texts = temp_df[text_col].values.astype(str)

    # 4) Run BART (unchanged)
    (true_rmse, null_rmse, chosen_dims, y_true_oof, y_pred_oof, texts_oof,
     avg_dim, idata_bart, idata_null, comp) = run_pymc_bart(
        X, y, texts,
        method=REDUCTION_METHOD,
        pca_variance_threshold=PCA_VARIANCE_THRESHOLD
    )

    # Store for plotting
    true_vals[target] = y_true_oof
    pred_vals[target] = y_pred_oof
    text_vals[target] = texts_oof
    loo_comparisons[target] = comp

    # 4) Compute metrics (as before)
    r2 = r2_score(y_true_oof, y_pred_oof)
    rmse_std = true_rmse / np.std(y_true_oof)
    rmse_null_std = null_rmse / np.std(y_true_oof)
    rmse_pct_range = true_rmse / (y_true_oof.max() - y_true_oof.min()) * 100
    

    # =====================================================
    # Quantile-based performance (on y_true)
    # =====================================================
    
    """
    Calculating R2 for each quantile does not make a lot of sense, since we reduce the variance too strongly
    this leads to extremly large negative R2 values
    
    """
    q20, q40, q60, q80 = np.quantile(y_true_oof, [0.2, 0.4, 0.6, 0.8])
    categories = np.digitize(y_true_oof, bins=[q20, q40, q60, q80])
    
    quantile_rmse = {}
    quantile_counts = {}
    
    for q in range(5):
        mask = categories == q
        n_q = np.sum(mask)
        quantile_counts[f"N_Q{q+1}"] = int(n_q)
        
        if n_q > 5:
            # Calculate MSE first, then take the square root for RMSE
            mse_q = mean_squared_error(y_true_oof[mask], y_pred_oof[mask])
            rmse_q = np.sqrt(mse_q)
        else:
            rmse_q = np.nan
            
        quantile_rmse[f"RMSE_Q{q+1}"] = rmse_q

    # =====================================================
    # Text length diagnostics
    # =====================================================
    
    text_lengths = np.array([len(t.split()) for t in texts_oof])
    abs_error = np.abs(y_true_oof - y_pred_oof)
    
    corr_len_error, p_len_error = pearsonr(text_lengths, abs_error) 
    
    # =====================================================
    # Text length quantile bins
    # =====================================================
    
    l20, l40, l60, l80 = np.quantile(text_lengths, [0.2, 0.4, 0.6, 0.8])
    len_categories = np.digitize(text_lengths, bins=[l20, l40, l60, l80])
    
    length_metrics = {}
    length_counts = {}
    
    for q in range(5):
        mask = len_categories == q
        n_q = np.sum(mask)
        length_counts[f"N_LQ{q+1}"] = int(n_q)
        
        if n_q > 5:
            # Calculate R2 for this length-bin
            r2_q = r2_score(y_true_oof[mask], y_pred_oof[mask])
            # Calculate RMSE for this length-bin
            rmse_q = np.sqrt(mean_squared_error(y_true_oof[mask], y_pred_oof[mask]))
        else:
            r2_q = np.nan
            rmse_q = np.nan
            
        length_metrics[f"R2_LQ{q+1}"] = r2_q
        length_metrics[f"RMSE_LQ{q+1}"] = rmse_q
    
    
    # =====================================================
    # Append to results DataFrame
    # =====================================================
    
    results.append({
        "Target": target,
        "N_after_filter": len(y),
        "Chosen dims per fold": chosen_dims,
        "Avg dims": np.mean(chosen_dims),
        "R2": r2,
        "RMSE_Original_Scale": true_rmse,
        "RMSE_Null_Model": null_rmse,
        "RMSE_Standardized": rmse_std,
        "RMSE_Null_Standardized": rmse_null_std,
        "RMSE_Pct_Range": rmse_pct_range,
        "Improvement_over_Mean": null_rmse - true_rmse,
        "Corr_Length_AbsError": corr_len_error,
        "P_Length_AbsError": p_len_error,
    
        # performance by outcome quantiles
        **quantile_rmse,
        **quantile_counts,
    
        # performance by text length
        **length_metrics,
        **length_counts
    })

df_res = pd.DataFrame(results)

# Optionally print or store the LOO comparisons
print("\nLOO model comparisons:")
with pd.option_context(
    'display.max_columns', None,      # No limit on number of columns
    'display.max_colwidth', None,     # No limit on width (shows full dicts)
    'display.expand_frame_repr', False, # Don't wrap columns to new lines
    'display.width', 1000             # Increase total character width
):
    for target, comp in loo_comparisons.items():
        print(f"\n{target}:")
        print(comp)


# Display nicely
display_cols = ["R2", "RMSE_Original_Scale", "R2_LQ1", "R2_LQ2", "R2_LQ3","R2_LQ4","R2_LQ5", 
                "RMSE_LQ1", "RMSE_LQ2", "RMSE_LQ3", "RMSE_LQ4", "RMSE_LQ5"]
df_res['FullName'] = df_res['Target'].map(full_names)
cols_to_show = ["FullName"] + display_cols
print(df_res[cols_to_show].to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))


# check the min max values for the masked values:
data_masked = data_cleaned.copy()

for var, text_col in target_to_text_map.items():
    if text_col in marked_masks:
        # Get the mask for this specific text column
        mask = marked_masks[text_col]
        
        # Count how many we are about to remove (optional but good for debugging)
        n_removed = mask.sum()
        print(f"Masking {n_removed} rows for {var} based on {text_col}")
        
        # Set values to NaN where the mask is True
        data_masked.loc[mask, var] = np.nan
    else:
        print(f"Warning: {text_col} not found in masks. Skipping {var}.")


""" # shows wrong n (check the dataframe, data has been set to nan based on masking)

from comprehensive_group_stats import comprehensive_group_stats, print_group_stats

variables = ["conf", 
             "cmpcy", 
             "const", 
             "calc", 
             "colr", 
             "compli", 
             "consp"]

stats_dict = comprehensive_group_stats(data_masked, variables, cluster_col=None)

print_group_stats(stats_dict)
"""

# ------------------------------------------------------------
# Plot 1: Null vs BART RMSE barplot
# ------------------------------------------------------------

# ----- Prepare data -----
df = df_res.sort_values('Target').reset_index(drop=True)
targets = df['Target'].values
target_labels = [full_names.get(t, t) for t in targets]

null_rmse = df['RMSE_Null_Model'].values
true_rmse = df['RMSE_Original_Scale'].values

# ----- Bar positions -----
x = np.arange(len(targets))          # the label locations
width = 0.35                          # width of each bar

# ----- Create plot -----
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, null_rmse, width, label='Null Model RMSE',
               color='skyblue', edgecolor='white')
bars2 = ax.bar(x + width/2, true_rmse, width, label='True Model RMSE',
               color='steelblue', edgecolor='white')

# ----- Add some text for labels, title and axes -----
ax.set_xlabel('Target')
ax.set_ylabel('RMSE')
ax.set_title('RMSE: Null Model vs BART')
ax.set_xticks(x)
ax.set_xticklabels(target_labels, rotation=45, ha='right')
ax.legend()

# Optional: Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# add_labels(bars1)
# add_labels(bars2)

# Light grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 2: Actual vs Predicted scatter plots (using stored predictions)
# ------------------------------------------------------------
n_targets = len(true_vals)
n_cols = 3
n_rows = (n_targets + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
# Flatten axes for easy indexing, even if only one subplot
if n_targets > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for i, target in enumerate(true_vals.keys()):
    ax = axes[i]
    y_true = true_vals[target]
    y_pred = pred_vals[target]
    full_name = full_names.get(target, target)

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="white", s=50)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    # Retrieve R2 and RMSE from df_res
    row = df_res[df_res["Target"] == target].iloc[0]
    ax.set_title(f"{full_name}\nR² = {row['R2']:.3f}, RMSE = {row['RMSE_Original_Scale']:.2f}")
    ax.legend(loc='upper left')

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Actual vs. Predicted Values", fontsize=20)
plt.show()


# ------------------------------------------------------------
# Plot 3: Residual plots
# ------------------------------------------------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
if n_targets > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for i, target in enumerate(true_vals.keys()):
    ax = axes[i]
    y_true = true_vals[target]
    y_pred = pred_vals[target]
    residuals = y_true - y_pred
    full_name = full_names.get(target, target)

    ax.scatter(y_true, residuals, alpha=0.6, edgecolor="white", s=50)
    ax.axhline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("True Value")
    ax.set_ylabel("Residual (True - Predicted)")
    ax.set_title(f"{full_name}")
    ax.fill_between([y_true.min(), y_true.max()], -0.5, 0.5, alpha=0.1, color="green", label="±0.5 zone")
    ax.legend(loc='upper right')

for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Residual Analysis (Bias Check)", fontsize=20)
plt.show()

# ------------------------------------------------------------
# Plot 4: Qualitative outlier inspection
# ------------------------------------------------------------
print("\n" + "="*80)
print("QUALITATIVE OUTLIER INSPECTION")
print("="*80)

for target in true_vals.keys():
    y_true = true_vals[target]
    y_pred = pred_vals[target]
    texts = text_vals[target]
    full_name = full_names.get(target, target)

    residuals = np.abs(y_true - y_pred)
    worst_idx = np.argsort(residuals)[-5:][::-1]   # 5 largest errors
    best_idx = np.argsort(residuals)[:5]            # 5 smallest errors

    print(f"\n{full_name}")
    print("-" * 40)
    print("\n  WORST PREDICTIONS (Largest Errors):")
    for j, idx in enumerate(worst_idx, 1):
        print(f"    {j}. True: {y_true[idx]:.1f}, Pred: {y_pred[idx]:.1f}, Error: {residuals[idx]:.2f}")
        print(f"       Text: {texts[idx]}")
    print("\n  BEST PREDICTIONS (Smallest Errors):")
    for j, idx in enumerate(best_idx, 1):
        print(f"    {j}. True: {y_true[idx]:.1f}, Pred: {y_pred[idx]:.1f}, Error: {residuals[idx]:.2f}")
        print(f"       Text: {texts[idx]}")
    print("\n" + "="*80)
    
    
    
