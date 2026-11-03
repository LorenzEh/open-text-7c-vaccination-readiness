"""BART on raw embeddings, without PCA -- exploratory robustness check.

Tests whether the preregistered PCA step disadvantages BART specifically.
Motivation: PCA is near-neutral for ridge (mean dR2 = +0.001 across the seven
dimensions), because L2 shrinkage is essentially rotation-invariant. Trees,
by contrast, split on one coordinate at a time, so a rotated basis in which
the signal is spread across many components is a poor fit for them. If BART
improves without PCA while ridge does not, the gap between the two models is
partly an artefact of the dimensionality reduction rather than evidence about
linearity.

Deviation note: the preregistration specifies PCA. This is exploratory and
does not replace the confirmatory analysis.

Everything else matches BART_FULL_METRICS.py: m=100, alpha=0.95, beta=2.0,
response='mix', 4 chains x (500 tune + 1000 draws), 5-fold KFold(seed 42),
same exclusion mask and min-word filter.

All seven dimensions. Resumable: dimensions already present in the output
JSON are skipped, so Confidence and Constraints are not recomputed.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[v]='1'
os.environ['PYTHONHASHSEED']='0'
import sys, json, time, random
import numpy as np, pandas as pd, pymc as pm, pymc_bart as pmb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED=42
BART_PARAMS={"m":100,"alpha":0.95,"beta":2.0,"response":"mix"}
SAMP={"draws":1000,"tune":500,"chains":4,"progressbar":False,"random_seed":SEED}
DIMS={"conf":"text_confidence","cmpcy":"text_complacency","const":"text_constraints",
      "calc":"text_calculation","colr":"text_coll_resp","compli":"text_compliance",
      "consp":"text_conspiracy"}
PAT=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'
OUT=str(paths.RESULTS/"results_bart_no_pca.json")

def main():
    np.random.seed(SEED); random.seed(SEED)
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    dc=pd.read_csv(paths.MERGED_CLEANED)
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo={k:v for k,v in DIMS.items() if k not in res}
    if not todo: print("nothing to do"); return
    print(f"running: {list(todo)}",flush=True)
    for tgt,tcol in todo.items():
        t0=time.time()
        m=dc[tcol].reindex(data.index,fill_value="").astype(str).str.contains(PAT,regex=True,na=False)
        df=data[[tgt,tcol]].dropna(); df=df[~m.loc[df.index]]
        df=df[df[tcol].astype(str).str.split().str.len()>=1]
        emb=[c for c in data.columns if c.startswith(f"{tcol}_emb_")]
        X=data.loc[df.index,emb].values.astype(np.float32)   # RAW, no PCA
        y=df[tgt].values.astype(np.float32)
        print(f"  {tgt}: X={X.shape} (no PCA)",flush=True)
        kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
        yt,yp=[],[]
        for fi,(tr,te) in enumerate(kf.split(X),1):
            ft=time.time()
            with pm.Model():
                Xd=pm.Data("X",X[tr]); yd=pm.Data("y",y[tr])
                mu=pmb.BART("mu",Xd,yd,**BART_PARAMS); s=pm.HalfNormal("sigma",5)
                pm.Normal("obs",mu=mu,sigma=s,observed=yd)
                idata=pm.sample(**SAMP); pm.set_data({"X":X[te]})
                pp=pm.sample_posterior_predictive(idata,var_names=["mu"],
                                                  progressbar=False,random_seed=SEED)
            pred=pp.posterior_predictive["mu"].mean(dim=["chain","draw"]).values
            yt.extend(y[te]); yp.extend(pred)
            print(f"    fold {fi}/5 rmse={np.sqrt(mean_squared_error(y[te],pred)):.4f} "
                  f"[{time.time()-ft:.0f}s]",flush=True)
        yt,yp=np.array(yt),np.array(yp)
        res[tgt]=dict(n=int(len(yt)),n_features=int(X.shape[1]),
                      r2=float(r2_score(yt,yp)),
                      rmse=float(np.sqrt(mean_squared_error(yt,yp))))
        json.dump(res,open(OUT,'w'),indent=1)
        print(f"NOPCA {tgt:7} R2={res[tgt]['r2']:+.4f} RMSE={res[tgt]['rmse']:.4f} "
              f"[{time.time()-t0:.0f}s]",flush=True)
    print("DONE",flush=True)

if __name__=="__main__":
    import multiprocessing as mp; mp.freeze_support(); main()
