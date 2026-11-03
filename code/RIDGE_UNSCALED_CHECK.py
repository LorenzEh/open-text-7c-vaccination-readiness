"""Ridge WITHOUT StandardScaler (matching BART's preprocessing), full permutation tests."""
import os
import sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # portable: no absolute paths
import paths
from sklearn.model_selection import KFold, permutation_test_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

SEED, NSPLIT, NPERM = 42, 5, 1000
# Widened 2026-08-03. The original grid stopped at 100, and under the standardised
# specification RidgeCV selected that boundary value on all seven dimensions -- it
# wanted more regularisation than the grid allowed, under-regularised, and lost about
# 0.12 R2. The unstandardised path selects 0.1-1 and is unaffected (verified identical
# to 1e-9), so no reported number changes; this prevents silent truncation.
ALPHAS = [10.0**k for k in range(-3, 9)]   # 1e-3 ... 1e8
data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE); dc=pd.read_csv(paths.MERGED_CLEANED)
pairs={"conf":"text_confidence","cmpcy":"text_complacency","const":"text_constraints",
       "calc":"text_calculation","colr":"text_coll_resp","compli":"text_compliance",
       "consp":"text_conspiracy"}
pat=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'
BONF=0.05/7
out={}
for t,tc in pairs.items():
    t0=time.time()
    Xc=[c for c in data.columns if c.startswith(f"{tc}_emb_")]
    m=dc[tc].reindex(data.index,fill_value="").astype(str).str.contains(pat,regex=True,na=False)
    df=data[[t,tc]+Xc].dropna(); df=df[~m.loc[df.index]]
    df=df[df[tc].str.split().str.len()>=1]
    X,y=df[Xc].values, df[t].values
    # NO StandardScaler -- identical preprocessing to BART_TRIALS.py
    pipe=make_pipeline(PCA(n_components=0.95,random_state=SEED),
                       RidgeCV(alphas=ALPHAS,scoring='r2'))
    cv=KFold(n_splits=NSPLIT,shuffle=True,random_state=SEED)
    yp=cross_val_predict(pipe,X,y,cv=cv,n_jobs=-1)
    r2=r2_score(y,yp); rmse=float(np.sqrt(mean_squared_error(y,yp)))
    _,perm,p=permutation_test_score(pipe,X,y,scoring="r2",cv=cv,
                                    n_permutations=NPERM,n_jobs=-1)
    out[t]=dict(n=len(df),r2=float(r2),rmse=rmse,p=float(p),
                p_bonf=min(1.0,float(p)*7), sig=bool(p<BONF),
                null_mean=float(np.mean(perm)), null_max=float(np.max(perm)),
                null_distribution=[float(v) for v in perm])
    print(f"PERM {t:7} n={len(df):4} R2={r2:+.3f} RMSE={rmse:.3f} p={p:.4f} "
          f"p_bonf={min(1.0,p*7):.4f} {'SIG' if p<BONF else 'ns'} "
          f"nullmean={np.mean(perm):+.3f} [{time.time()-t0:.0f}s]",flush=True)
json.dump(out, open(paths.RESULTS/"results_ridge_unscaled_permutation.json", "w"), indent=1)
print('saved results_ridge_unscaled_permutation.json')
print("DONE",flush=True)
