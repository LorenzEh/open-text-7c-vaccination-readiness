"""Ridge metrics matching BART_FULL_METRICS.py, for both preprocessing specs.

  scaled   -- StandardScaler -> PCA(0.95) -> RidgeCV, y standardised via
              TransformedTargetRegressor. This is RIDGE1.py, i.e. Table 1.
  unscaled -- PCA(0.95) -> RidgeCV, no scaling anywhere. Identical
              preprocessing to BART_TRIALS.py, so the two are comparable.

Reports global R2/RMSE, R2+RMSE by text-length quintile (LQ1-LQ5) and RMSE by
observed-score quintile (Q1-Q5), using the same binning as BART_FULL_METRICS.py.

Ridge has no ELPD -- that is a Bayesian LOO quantity. The analogous evidence
column is the permutation test, carried in from the full 1000-permutation runs
(see PERM_P below and CLAUDE.md); re-running those takes ~35 min per spec and
is not repeated here. RMSE of the mean-only null is included as a baseline.

Fast: no MCMC. Whole script runs in a couple of minutes.
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Alpha grid widened 2026-08-03: the original stopped at 100, which the standardised
# specification hit on all seven dimensions (under-regularised, ~0.12 R2 lost). The
# unstandardised path picks 0.1-1, so no reported number changes.
SEED=42; NSPLIT=5; ALPHAS=[10.0**k for k in range(-3,9)]; PCA_VAR=0.95
PAIRS={"conf":"text_confidence","cmpcy":"text_complacency","const":"text_constraints",
       "calc":"text_calculation","colr":"text_coll_resp","compli":"text_compliance",
       "consp":"text_conspiracy"}
PAT=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'

# From the full 1000-permutation runs. Bonferroni alpha = .05/7 = .00714;
# .001 is the permutation floor (1/1000). All significant in both specs.
PERM_P={"scaled":  {"conf":.001,"cmpcy":.001,"const":.001,"calc":.002,
                    "colr":.001,"compli":.001,"consp":.001},
        "unscaled":{"conf":.001,"cmpcy":.001,"const":.001,"calc":.001,
                    "colr":.001,"compli":.001,"consp":.001}}

OUT=str(paths.RESULTS/"results_ridge_full_metrics.json")
PRED=str(paths.RESULTS/"results_ridge_oof_predictions.npz")

def build(spec):
    if spec=="scaled":
        inner=make_pipeline(StandardScaler(),
                            PCA(n_components=PCA_VAR,random_state=SEED),
                            RidgeCV(alphas=ALPHAS,scoring='r2'))
        return TransformedTargetRegressor(regressor=inner,
                                          transformer=StandardScaler(),inverse_func=None)
    return make_pipeline(PCA(n_components=PCA_VAR,random_state=SEED),
                         RidgeCV(alphas=ALPHAS,scoring='r2'))

def quintile_metrics(values,y_true,y_pred,prefix,with_r2):
    """Bins from np.quantile(.2,.4,.6,.8) + np.digitize; skipped when n<=5.
    Identical to BART_FULL_METRICS.py and BART_TRIALS.py."""
    cat=np.digitize(values,bins=np.quantile(values,[0.2,0.4,0.6,0.8]))
    out={}
    for i in range(5):
        m=cat==i; n=int(m.sum()); out[f"N_{prefix}{i+1}"]=n
        if n>5:
            out[f"RMSE_{prefix}{i+1}"]=float(np.sqrt(mean_squared_error(y_true[m],y_pred[m])))
            if with_r2: out[f"R2_{prefix}{i+1}"]=float(r2_score(y_true[m],y_pred[m]))
        else:
            out[f"RMSE_{prefix}{i+1}"]=None
            if with_r2: out[f"R2_{prefix}{i+1}"]=None
    return out

def main():
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    dc=pd.read_csv(paths.MERGED_CLEANED)
    res={}; preds={}
    for spec in ("scaled","unscaled"):
        res[spec]={}
        for tgt,tcol in PAIRS.items():
            t0=time.time()
            Xc=[c for c in data.columns if c.startswith(f"{tcol}_emb_")]
            m=dc[tcol].reindex(data.index,fill_value="").astype(str).str.contains(PAT,regex=True,na=False)
            df=data[[tgt,tcol]+Xc].dropna(); df=df[~m.loc[df.index]]
            df=df[df[tcol].str.split().str.len()>=1]
            X=df[Xc].values; y=df[tgt].values
            lens=df[tcol].str.split().str.len().values.astype(float)
            cv=KFold(n_splits=NSPLIT,shuffle=True,random_state=SEED)
            yp=cross_val_predict(build(spec),X,y,cv=cv,n_jobs=-1)
            preds[f"{spec}_{tgt}_true"]=y; preds[f"{spec}_{tgt}_pred"]=yp
            preds[f"{spec}_{tgt}_len"]=lens
            row=dict(n=int(len(y)),r2=float(r2_score(y,yp)),
                     rmse=float(np.sqrt(mean_squared_error(y,yp))),
                     rmse_null=float(np.sqrt(mean_squared_error(y,np.full_like(y,y.mean())))))
            row["improvement_over_mean"]=row["rmse_null"]-row["rmse"]
            row.update(quintile_metrics(lens,y,yp,"LQ",with_r2=True))
            row.update(quintile_metrics(y,   y,yp,"Q", with_r2=False))
            p=PERM_P[spec][tgt]
            row["perm_p"]=p; row["perm_p_bonferroni"]=min(1.0,p*7)
            row["perm_significant"]=p < 0.05/7
            row["perm_source"]="1000-permutation run; see CLAUDE.md"
            res[spec][tgt]=row
            print(f"RIDGE {spec:8} {tgt:7} R2={row['r2']:+.4f} RMSE={row['rmse']:.4f} [{time.time()-t0:.0f}s]",flush=True)
    json.dump(res,open(OUT,'w'),indent=1)
    np.savez_compressed(PRED,**preds)
    print("DONE",flush=True)

if __name__=="__main__":
    main()
