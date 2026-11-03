"""ELPD and convergence diagnostics for the no-PCA BART models.

BART_NO_PCA_CHECK.py produced out-of-fold R2/RMSE only. Before those numbers
could be considered for anything beyond an exploratory robustness check they
need the same scrutiny the PCA-based models received: a PSIS-LOO comparison
against an intercept-only null, and R-hat / ESS.

The concern is specific: R-hat was already marginal with ~223 principal
components (sigma above 1.01 on 3 of 7 dimensions, pointwise mu 69-88% above
threshold). With 1024 raw predictors the sampler explores a much larger tree
space and may mix worse, which would count against these models regardless of
their out-of-fold accuracy.

Final model on the full analytic sample + intercept-only null, per dimension.
4 chains, everything else as in BART_NO_PCA_CHECK.py. Resumable.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[v]='1'
os.environ['PYTHONHASHSEED']='0'
import sys, json, time, random
import numpy as np, pandas as pd, pymc as pm, pymc_bart as pmb, arviz as az
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED=42
BART_PARAMS={"m":100,"alpha":0.95,"beta":2.0,"response":"mix"}
SAMP={"draws":1000,"tune":500,"chains":4,"progressbar":False,"random_seed":SEED}
DIMS={"conf":"text_confidence","cmpcy":"text_complacency","const":"text_constraints",
      "calc":"text_calculation","colr":"text_coll_resp","compli":"text_compliance",
      "consp":"text_conspiracy"}
PAT=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'
OUT=str(paths.RESULTS/"results_bart_no_pca_diagnostics.json")

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
        X=data.loc[df.index,emb].values.astype(np.float32)     # raw, no PCA
        y=df[tgt].values.astype(np.float32)
        with pm.Model():
            Xd=pm.Data("X",X); yd=pm.Data("y",y)
            mu=pmb.BART("mu",Xd,yd,**BART_PARAMS); s=pm.HalfNormal("sigma",5)
            pm.Normal("obs",mu=mu,sigma=s,observed=yd)
            ib=pm.sample(**SAMP,idata_kwargs={"log_likelihood":True})
        with pm.Model():
            mu=pm.Normal("mu",mu=y.mean(),sigma=y.std()*2); s=pm.HalfNormal("sigma",5)
            pm.Normal("obs",mu=mu,sigma=s,observed=y)
            inl=pm.sample(**SAMP,idata_kwargs={"log_likelihood":True})
        c=az.compare({"BART":ib,"Null":inl},ic="loo")
        rh=az.rhat(ib); rmu=np.atleast_1d(rh['mu'].values).ravel()
        essmu=np.atleast_1d(az.ess(ib)['mu'].values).ravel()
        res[tgt]=dict(n=int(len(y)),n_features=int(X.shape[1]),
            elpd_diff=float(c['elpd_diff'].max()),dse=float(c['dse'].max()),
            rhat_sigma=float(np.atleast_1d(rh['sigma'].values).ravel()[0]),
            rhat_mu_max=float(np.nanmax(rmu)),
            rhat_mu_frac_gt101=float(np.mean(rmu>1.01)),
            ess_mu_min=float(np.nanmin(essmu)))
        json.dump(res,open(OUT,'w'),indent=1)
        r=res[tgt]
        print(f"NOPCADIAG {tgt:7} dELPD={r['elpd_diff']:.2f} DSE={r['dse']:.2f} "
              f"rhat_sigma={r['rhat_sigma']:.4f} rhat_mu_max={r['rhat_mu_max']:.4f} "
              f"mu>1.01={r['rhat_mu_frac_gt101']:.1%} essmin={r['ess_mu_min']:.0f} "
              f"[{time.time()-t0:.0f}s]",flush=True)
    print("DONE",flush=True)

if __name__=="__main__":
    import multiprocessing as mp; mp.freeze_support(); main()
