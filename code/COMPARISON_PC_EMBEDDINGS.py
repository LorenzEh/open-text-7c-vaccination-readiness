import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


import paths

data_incl_quest = pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)

data_excl_quest = pd.read_csv(paths.EMBEDDINGS)

data_cleaned = pd.read_csv(paths.MERGED_CLEANED)



# ------------------------------------------------------------
# Create masks from data_cleaned (same logic as in your snippet)
# ------------------------------------------------------------
text_cols = [c for c in data_cleaned.columns if c.startswith('text_')]
MARKED_NUMBERS = [-1, 2, 3, 4, 5, 6, 7]

terms = []
for num in MARKED_NUMBERS:
    if num == -1:
        terms.append(r'-1')
    else:
        terms.append(r'\(' + str(num) + r'\)')
pattern = r'(?:' + '|'.join(terms) + r')$'

# We'll create a mask for each text column, aligned with the index of data_cleaned
marked_masks = {}
for col in text_cols:
    if col in data_cleaned.columns:
        cleaned_vals = data_cleaned[col].astype(str)
        mask = cleaned_vals.str.contains(pattern, regex=True, na=False)
        marked_masks[col] = mask
    else:
        print(f"Warning: {col} not in data_cleaned")
        marked_masks[col] = pd.Series(False, index=data_cleaned.index)

# ------------------------------------------------------------
# Function to perform PCA on a given dataframe + text column + mask
# ------------------------------------------------------------
def get_pca_results(df, text_col, mask_series, variance_threshold=0.95):
    """
    df: DataFrame containing embedding columns (e.g., data_incl_quest)
    text_col: e.g., 'text_1'
    mask_series: boolean Series (same index as df) – True = marked rows to EXCLUDE
    variance_threshold: cumulative explained variance to reach (e.g., 0.95)
    
    Returns:
        - n_components_95: number of PCs needed to reach threshold
        - explained_variance_ratio: array of ratios
        - cumsum: cumulative sum
        - components: first few component loadings (optional, for comparison)
        - X_clean: cleaned embedding matrix (after masking & scaling)
    """
    emb_prefix = f"{text_col}_emb_"
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    
    if not emb_cols:
        print(f"No embedding columns found for {text_col}")
        return None
    
    # Subset to relevant columns and align index
    temp_df = df[emb_cols].copy()
    
    # Apply mask: keep rows where mask is False (i.e., NOT marked)
    # Ensure index alignment
    common_idx = temp_df.index.intersection(mask_series.index)
    temp_df = temp_df.loc[common_idx]
    mask_aligned = mask_series.loc[common_idx]
    temp_df = temp_df[~mask_aligned]
    
    if len(temp_df) < 10:
        print(f"Too few rows after masking for {text_col}: {len(temp_df)}")
        return None
    
    # Standardize embeddings (PCA on correlation matrix)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(temp_df)
    
    # PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum >= variance_threshold) + 1
    
    return {
        'n_components_95': n_components_95,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumsum': cumsum,
        'components': pca.components_,      # shape (n_components, n_features)
        'X_clean': X_scaled,
        'scaler': scaler
    }

# ------------------------------------------------------------
# Loop over each text column and compare incl_quest vs excl_quest
# ------------------------------------------------------------
comparison_results = []

for text_col in text_cols:
    print(f"\n{'='*60}\nAnalyzing {text_col}\n{'='*60}")
    
    # Get masks for this column (from data_cleaned)
    mask = marked_masks.get(text_col)
    if mask is None:
        print(f"No mask for {text_col}, skipping")
        continue
    
    # Run PCA on both datasets
    res_incl = get_pca_results(data_incl_quest, text_col, mask)
    res_excl = get_pca_results(data_excl_quest, text_col, mask)
    
    if res_incl is None or res_excl is None:
        print(f"Skipping {text_col} due to insufficient data")
        continue
    
    # Compare number of PCs for 95% variance
    print(f"\nComponents needed to reach 95% variance:")
    print(f"  Incl_quest: {res_incl['n_components_95']}")
    print(f"  Excl_quest: {res_excl['n_components_95']}")
    
    # Compare explained variance of first 5 PCs (or fewer if less)
    n_compare = min(5, len(res_incl['explained_variance_ratio']), len(res_excl['explained_variance_ratio']))
    print(f"\nExplained variance ratio (first {n_compare} PCs):")
    for i in range(n_compare):
        print(f"  PC{i+1}: Incl={res_incl['explained_variance_ratio'][i]:.4f} , Excl={res_excl['explained_variance_ratio'][i]:.4f} , Diff={res_incl['explained_variance_ratio'][i] - res_excl['explained_variance_ratio'][i]:.4f}")
    
    # Compare cumulative variance for first few PCs
    print(f"\nCumulative variance (first {n_compare} PCs):")
    for i in range(n_compare):
        print(f"  PC1-{i+1}: Incl={res_incl['cumsum'][i]:.4f} , Excl={res_excl['cumsum'][i]:.4f} , Diff={res_incl['cumsum'][i] - res_excl['cumsum'][i]:.4f}")
    
    # Optional: Compare loadings of first PC (absolute correlation between loading vectors)
    from scipy.spatial.distance import cosine
    pc1_incl = res_incl['components'][0]
    pc1_excl = res_excl['components'][0]
    similarity = 1 - cosine(pc1_incl, pc1_excl)
    print(f"\nCosine similarity of first PC loadings: {similarity:.4f} (1 = identical direction)")
    
    # Store results for later visualization
    comparison_results.append({
        'text_col': text_col,
        'n_incl': res_incl['n_components_95'],
        'n_excl': res_excl['n_components_95'],
        'var_pc1_incl': res_incl['explained_variance_ratio'][0],
        'var_pc1_excl': res_excl['explained_variance_ratio'][0],
        'cumvar_pc5_incl': res_incl['cumsum'][min(4, len(res_incl['cumsum'])-1)],
        'cumvar_pc5_excl': res_excl['cumsum'][min(4, len(res_excl['cumsum'])-1)],
        'similarity_pc1': similarity
    })

# ------------------------------------------------------------
# Summary table
# ------------------------------------------------------------
summary_df = pd.DataFrame(comparison_results)
print("\n\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(summary_df.to_string(index=False))

# Optional: Plot explained variance for first 10 PCs for one dimension as example
example_col = text_cols[0]  # change as needed
res_incl = get_pca_results(data_incl_quest, example_col, marked_masks[example_col])
res_excl = get_pca_results(data_excl_quest, example_col, marked_masks[example_col])

if res_incl and res_excl:
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(res_incl['explained_variance_ratio'][:10])+1), 
             res_incl['explained_variance_ratio'][:10], 'o-', label='Incl Quest')
    plt.plot(range(1, len(res_excl['explained_variance_ratio'][:10])+1), 
             res_excl['explained_variance_ratio'][:10], 's-', label='Excl Quest')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA Variance Comparison for {example_col}')
    plt.legend()
    plt.grid(True)
    plt.show()