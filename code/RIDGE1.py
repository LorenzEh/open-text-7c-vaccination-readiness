import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, permutation_test_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor  # add this import

import paths

# ============================
# SETTINGS
# ============================

RANDOM_SEED = 42
N_OUTER_SPLITS = 5
N_PERMUTATIONS = 1000  # Set to 1000 for final paper, 100 for testing speed

# Alphas for RidgeCV (Log space is usually better)
# Widened 2026-08-03. The original grid stopped at 100, and under the standardised
# specification RidgeCV selected that boundary value on all seven dimensions -- it
# wanted more regularisation than the grid allowed, under-regularised, and lost about
# 0.12 R2. The unstandardised path selects 0.1-1 and is unaffected (verified identical
# to 1e-9), so no reported number changes; this prevents silent truncation.
ALPHAS = [10.0**k for k in range(-3, 9)]   # 1e-3 ... 1e8

# PCA Variance to keep
PCA_VAR = 0.95 

# Toggle for masking marked rows
USE_MASK = True 

# ============================
# DATA & MAPPINGS
# ============================

# Main dataset
data = pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)

# Cleaned dataset for mask generation
data_cleaned = pd.read_csv(paths.MERGED_CLEANED)

text_target_pairs = {
    "conf": "text_confidence",
    "cmpcy": "text_complacency",
    "const": "text_constraints",
    "calc": "text_calculation",
    "colr": "text_coll_resp",
    "compli": "text_compliance",
    "consp": "text_conspiracy",
}

# ============================
# CREATE MASKS
# ============================

text_cols = [c for c in data_cleaned.columns if c.startswith('text_')]
MARKED_NUMBERS = [-1, 2, 3, 4, 5, 6, 7]   # -1 appears without parentheses

if MARKED_NUMBERS:
    terms = []
    for num in MARKED_NUMBERS:
        if num == -1:
            terms.append(r'-1')               # literal -1
        else:
            terms.append(r'\(' + str(num) + r'\)')   # e.g. \(2\), \(3\)
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

# ============================
# MAIN FUNCTION
# ============================

def run_rigorous_nested_ridgev2(vars_to_analyze, dataframe, mapping_dict, min_words=0, use_mask=False, masks_dict=None):
    """
    Perform rigorous nested ridge regression with optional text length filtering and masking.
    """
    final_results = {}

    for target_var in tqdm(vars_to_analyze, desc="Analyzing Targets"):
        
        # 1. Data Prep
        if target_var not in dataframe.columns:
            print(f"Skipping {target_var}: Not in columns.")
            continue

        text_col = mapping_dict[target_var]
        emb_prefix = f"{text_col}_emb_"
        X_cols = [c for c in dataframe.columns if c.startswith(emb_prefix)]

        if not X_cols:
            print(f"Skipping {target_var}: No embeddings found.")
            continue

        temp_df = dataframe[[target_var, text_col] + X_cols].dropna()

        # ==========================================
        # APPLY MASKING LOGIC
        # ==========================================
        if masks_dict is not None and text_col in masks_dict:
            mask_marked = masks_dict[text_col].loc[temp_df.index]
            n_marked = mask_marked.sum()
            
            if use_mask:
                if n_marked > 0:
                    print(f"\n  For target '{target_var}', excluding {n_marked} rows because they are marked.")
                temp_df = temp_df[~mask_marked]  # keep only unmarked rows
            else:
                if n_marked > 0:
                    print(f"\n  Note: {n_marked} marked rows exist for '{target_var}' but are being kept (USE_MASK=False).")

        if len(temp_df) < 50:
            print(f"Skipping {target_var}: Not enough data after masking (N={len(temp_df)})")
            continue

        # ==========================================
        # TEXT LENGTH FILTERING (based on min_words)
        # ==========================================
        word_counts = temp_df[text_col].str.split().str.len()
        length_mask = word_counts >= min_words
        temp_df = temp_df[length_mask].copy()
        
        if len(temp_df) < 50:
            print(f"Skipping {target_var}: After text-length filtering (min_words={min_words}), only {len(temp_df)} rows remain (<50).")
            continue
        
        X = temp_df[X_cols].values
        y = temp_df[target_var].values

        # ==========================================
        # TEXT LENGTH (for diagnostic, after filtering)
        # ==========================================
        text_lengths = temp_df[text_col].str.split().str.len().values

        # ========== FIXED SCALING – NO LEAKAGE ==========
        inner_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_VAR, random_state=RANDOM_SEED),
            RidgeCV(alphas=ALPHAS, scoring='r2')
        )

        model_pipeline = TransformedTargetRegressor(
            regressor=inner_pipeline,
            transformer=StandardScaler(),
            inverse_func=None
        )
        # =================================================

        outer_cv = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_SEED)

        # ==========================================
        # A. PREDICTION (For RMSE & Quantiles)
        # ==========================================
        
        y_pred = cross_val_predict(
            model_pipeline, X, y, cv=outer_cv, n_jobs=-1
        )

        r2_final = r2_score(y, y_pred)
        rmse_final = np.sqrt(mean_squared_error(y, y_pred))
        
        null_pred = np.mean(y)
        rmse_null = np.sqrt(mean_squared_error(y, np.full_like(y, null_pred)))

        # ==========================================
        # B. QUANTILE ANALYSIS (TARGET QUANTILES)
        # ==========================================
        
        q20, q40, q60, q80 = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
        categories = np.digitize(y, bins=[q20, q40, q60, q80])
        
        quantile_rmse = {}
        quantile_counts = {}

        for q in range(5):
            mask = categories == q
            n_q = np.sum(mask)
            quantile_counts[f"N_Q{q+1}"] = int(n_q)

            if n_q > 5:
                rmse_q = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
            else:
                rmse_q = np.nan
            quantile_rmse[f"RMSE_Q{q+1}"] = rmse_q

        # ==========================================
        # C. TEXT LENGTH DIAGNOSTIC
        # ==========================================
        
        abs_error = np.abs(y - y_pred)
        from scipy.stats import pearsonr
        corr_len_error, p_len_error = pearsonr(text_lengths, abs_error)

        len_q20, len_q40, len_q60, len_q80 = np.quantile(text_lengths, [0.2, 0.4, 0.6, 0.8])
        len_categories = np.digitize(text_lengths, bins=[len_q20, len_q40, len_q60, len_q80])

        length_rmse = {}
        length_counts = {}

        for q in range(5):
            mask = len_categories == q
            n_q = np.sum(mask)
            length_counts[f"N_LQ{q+1}"] = int(n_q)

            if n_q > 5:
                rmse_q = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
            else:
                rmse_q = np.nan

            length_rmse[f"RMSE_LQ{q+1}"] = rmse_q

        # ==========================================
        # D. HYPOTHESIS TESTING (Permutation)
        # ==========================================
        
        actual_score, perm_scores, p_value = permutation_test_score(
            model_pipeline, 
            X, 
            y,
            scoring="r2", 
            cv=outer_cv, 
            n_permutations=N_PERMUTATIONS, 
            n_jobs=-1
        )

        # ==========================================
        # STORE
        # ==========================================
        
        final_results[target_var] = {
            "R2": r2_final,
            "RMSE_Original_Scale": rmse_final,
            "RMSE_Null_Model": rmse_null,
            
            "RMSE_Standardized": rmse_final / np.std(y),
            "RMSE_Null_Standardized": rmse_null / np.std(y),
            "RMSE_Pct_Range": rmse_final / (y.max() - y.min()) * 100,
            
            "Improvement_over_Mean": rmse_null - rmse_final,
            "P_Value_Permutation": p_value,
            "perm_scores": perm_scores,
            "actual_r2_perm": actual_score,
            "Corr_Length_AbsError": corr_len_error,
            "P_Length_AbsError": p_len_error,
            **quantile_rmse,
            **quantile_counts,
            **length_rmse,
            **length_counts
        }

    return final_results

# ============================
# EXECUTION
# ============================

results = run_rigorous_nested_ridgev2(
    vars_to_analyze=list(text_target_pairs.keys()),
    dataframe=data,
    mapping_dict=text_target_pairs,
    min_words=1,
    use_mask=USE_MASK,           # Passed from settings
    masks_dict=marked_masks      # Passed from the global mask generation
)
# ============================
# RESULTS FORMATTING
# ============================

results_df = pd.DataFrame(results).T

# Bonferroni Correction
alpha_level = 0.05
bonferroni_thresh = alpha_level / len(results_df)
results_df["Significant"] = results_df["P_Value_Permutation"] < bonferroni_thresh

print(f"\nBonferroni corrected alpha = {bonferroni_thresh:.5f}\n")

# Display nicely
display_cols = ["R2", "Significant", "RMSE_Original_Scale","RMSE_Null_Model"]
print(results_df.sort_values("R2", ascending=False)[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))

# ============================
# PERMUTATION PLOTS (FIXED – USES fig.axes)
# ============================

import seaborn as sns
import matplotlib.pyplot as plt

variable_mapping = {
    "conf": "Confidence",
    "cmpcy": "Complacency",
    "const": "Constraints",
    "calc": "Calculation",
    "colr": "Collective Responsibility",
    "compli": "Compliance",
    "consp": "Conspiracy",
}

sns.set_theme(style="white")


# adjust how the plots are shown in the single plot (...)
n_vars = len(results_df)                     # number of variables to plot
n_cols = 3
n_rows = (n_vars + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(6*n_cols, 5*n_rows),
                         constrained_layout=True)

axes_list = fig.axes                         # flat list of ALL Axes objects



for i, (abbr, row) in enumerate(results_df.iterrows()):
    ax = axes_list[i]                       # guaranteed to be a single Axes
    full_name = variable_mapping.get(abbr, abbr)

    # --- Retrieve values ---
    perm_scores = row['perm_scores']
    actual_r2   = row['actual_r2_perm']
    p_val       = row['P_Value_Permutation']
    r2          = row['R2']

    # --- Plot ---
    sns.histplot(
        perm_scores,
        bins=30,
        kde=True,
        color="skyblue",
        edgecolor="white",
        linewidth=0.5,
        ax=ax
    )
    ax.axvline(
        actual_r2,
        color="crimson",
        linestyle="--",
        linewidth=3,
        label=f"Observed R² ({r2:.3f})"
    )
    ax.set_title(f"{full_name}\np = {p_val:.4f}", fontsize=14)
    ax.set_xlabel("R² Score")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper left')

# Turn off any remaining unused subplots
for ax in axes_list[len(results_df):]:
    ax.axis("off")

plt.suptitle("Permutation Test Results (Null Distribution vs. Observed Fit)", fontsize=20)
plt.show()



# ----- Diagnostics: 

# Recreate model_pipeline and outer_cv for plotting
inner_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=PCA_VAR, random_state=RANDOM_SEED),
    RidgeCV(alphas=ALPHAS, scoring='r2')
)

model_pipeline = TransformedTargetRegressor(
    regressor=inner_pipeline,
    transformer=StandardScaler(),
    inverse_func=None
)

outer_cv = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_SEED)


# ==========================================
# A. ACTUAL VS. PREDICTED SCATTER PLOTS
# ==========================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
axes_list = fig.axes

for i, (abbr, row) in enumerate(results_df.iterrows()):
    ax = axes_list[i]
    full_name = variable_mapping.get(abbr, abbr)
    
    text_col = text_target_pairs[abbr]
    emb_prefix = f"{text_col}_emb_"
    X_cols = [c for c in data.columns if c.startswith(emb_prefix)]
    
    temp_df = data[[abbr, text_col] + X_cols].dropna()
    X = temp_df[X_cols].values
    y = temp_df[abbr].values
    
    y_pred = cross_val_predict(model_pipeline, X, y, cv=outer_cv, n_jobs=-1)
    
    ax.scatter(y, y_pred, alpha=0.6, edgecolor="white", s=50)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"{full_name}\nR² = {row['R2']:.3f}, RMSE = {row['RMSE_Original_Scale']:.2f}")
    ax.legend(loc='upper left')

for ax in axes_list[len(results_df):]:
    ax.axis("off")

plt.suptitle("Actual vs. Predicted Values", fontsize=20)
plt.show()


# ==========================================
# B. RESIDUAL ANALYSIS
# ==========================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
axes_list = fig.axes

for i, (abbr, row) in enumerate(results_df.iterrows()):
    ax = axes_list[i]
    full_name = variable_mapping.get(abbr, abbr)
    
    text_col = text_target_pairs[abbr]
    emb_prefix = f"{text_col}_emb_"
    X_cols = [c for c in data.columns if c.startswith(emb_prefix)]
    
    temp_df = data[[abbr, text_col] + X_cols].dropna()
    X = temp_df[X_cols].values
    y = temp_df[abbr].values
    
    y_pred = cross_val_predict(model_pipeline, X, y, cv=outer_cv, n_jobs=-1)
    
    residuals = y - y_pred
    ax.scatter(y, residuals, alpha=0.6, edgecolor="white", s=50)
    ax.axhline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("True Value")
    ax.set_ylabel("Residual (True - Predicted)")
    ax.set_title(f"{full_name}")
    ax.fill_between([y.min(), y.max()], [-0.5, -0.5], [0.5, 0.5], alpha=0.1, color="green", label="±0.5 zone")
    ax.legend(loc='upper right')

for ax in axes_list[len(results_df):]:
    ax.axis("off")

plt.suptitle("Residual Analysis (Bias Check)", fontsize=20)
plt.show()

# ==========================================
# C. QUALITATIVE OUTLIER INSPECTION
# ==========================================

def get_outlier_analysis(data, target_var, text_col, X_cols, model_pipeline, outer_cv):
    temp_df = data[[target_var, text_col] + X_cols].dropna()
    if len(temp_df) < 50:
        return None
    
    X = temp_df[X_cols].values
    y = temp_df[target_var].values
    texts = temp_df[text_col].values
    
    y_pred = cross_val_predict(model_pipeline, X, y, cv=outer_cv, n_jobs=-1)
    residuals = y - y_pred
    abs_errors = np.abs(residuals)
    
    worst_idx = np.argsort(abs_errors)[-5:][::-1]
    best_idx = np.argsort(abs_errors)[:5]
    
    return {
        "target": target_var,
        "worst_predictions": [{"text": texts[i], "true": y[i], "pred": y_pred[i], "error": abs_errors[i]} for i in worst_idx],
        "best_predictions": [{"text": texts[i], "true": y[i], "pred": y_pred[i], "error": abs_errors[i]} for i in best_idx]
    }

print("\n" + "="*80)
print("QUALITATIVE OUTLIER INSPECTION")
print("="*80)

for target_var in text_target_pairs.keys():
    text_col = text_target_pairs[target_var]
    emb_prefix = f"{text_col}_emb_"
    X_cols = [c for c in data.columns if c.startswith(emb_prefix)]
    
    if not X_cols:
        continue
    
    result = get_outlier_analysis(data, target_var, text_col, X_cols, model_pipeline, outer_cv)
    if result is None:
        continue
    
    full_name = variable_mapping.get(target_var, target_var)
    print(f"\n{full_name}")
    print("-" * 40)
    
    print("\n  WORST PREDICTIONS (Largest Errors):")
    for j, item in enumerate(result["worst_predictions"], 1):
        print(f"    {j}. True: {item['true']:.1f}, Pred: {item['pred']:.1f}, Error: {item['error']:.2f}")
        print(f"       Text: {item['text']}")
    
    print("\n  BEST PREDICTIONS (Smallest Errors):")
    for j, item in enumerate(result["best_predictions"], 1):
        print(f"    {j}. True: {item['true']:.1f}, Pred: {item['pred']:.1f}, Error: {item['error']:.2f}")
        print(f"       Text: {item['text']}")
    
    print("\n" + "="*80)








