"""BART, all 7 dimensions, 4 chains instead of 2.

Everything else matches BART_TRIALS.py exactly: m=100, alpha=0.95, beta=2.0,
response='mix', 500 tune + 1000 draws, unscaled PCA at 95% var fitted within
each training fold, 5-fold KFold(shuffle, seed=42).

Answers two questions at once:
  1. Does the R-hat problem survive a proper chain count? (Methods claims <1.01)
  2. Is BART still >= matched-preprocessing Ridge when fairly configured?
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[v]='1'
os.environ['PYTHONHASHSEED']='0'
import sys, json, time, random
import numpy as np, pandas as pd, pymc as pm, pymc_bart as pmb, arviz as az
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # portable: no absolute paths
import paths

SEED=42
BART_PARAMS={"m":100,"alpha":0.95,"beta":2.0,"response":"mix"}
SAMP={"draws":1000,"tune":500,"chains":4,"progressbar":False,"random_seed":SEED}
PAIRS={"conf":"text_confidence","cmpcy":"text_complacency","const":"text_constraints",
       "calc":"text_calculation","colr":"text_coll_resp","compli":"text_compliance",
       "consp":"text_conspiracy"}
PAT=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'
OUT=str(paths.RESULTS/"results_bart_4chain_partial.json")

def fit(X,y):
    with pm.Model():
        Xd=pm.Data("X",X); yd=pm.Data("y",y)
        mu=pmb.BART("mu",Xd,yd,**BART_PARAMS); s=pm.HalfNormal("sigma",5)
        pm.Normal("obs",mu=mu,sigma=s,observed=yd)
        idata=pm.sample(**SAMP)
    return idata

def main():
    np.random.seed(SEED); random.seed(SEED)
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    dc=pd.read_csv(paths.MERGED_CLEANED)
    # Resumable: reload finished dimensions and skip them. Each dimension is
    # written to OUT as soon as it completes, so an interrupted run loses at
    # most the dimension in flight.
    res={}
    if os.path.exists(OUT):
        res=json.load(open(OUT))
        if res:
            print(f"resuming — already done: {list(res)}",flush=True)
    todo={k:v for k,v in PAIRS.items() if k not in res}
    if not todo:
        print("nothing to do; all 7 dimensions present in",OUT,flush=True); return
    print(f"running: {list(todo)}",flush=True)

    for tgt,tcol in todo.items():
        t0=time.time()
        m=dc[tcol].reindex(data.index,fill_value="").astype(str).str.contains(PAT,regex=True,na=False)
        df=data[[tgt,tcol]].dropna(); df=df[~m.loc[df.index]]
        df=df[df[tcol].astype(str).str.split().str.len()>=1]
        emb=[c for c in data.columns if c.startswith(f"{tcol}_emb_")]
        X=data.loc[df.index,emb].values.astype(np.float32)
        y=df[tgt].values.astype(np.float32)
        kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
        yt,yp,dims=[],[],[]
        for tr,te in kf.split(X):
            p=PCA(random_state=SEED).fit(X[tr])
            d=int(np.searchsorted(np.cumsum(p.explained_variance_ratio_),0.95)+1); dims.append(d)
            r=PCA(n_components=d,random_state=SEED)
            Xtr=r.fit_transform(X[tr]); Xte=r.transform(X[te])
            with pm.Model():
                Xd=pm.Data("X",Xtr); yd=pm.Data("y",y[tr])
                mu=pmb.BART("mu",Xd,yd,**BART_PARAMS); s=pm.HalfNormal("sigma",5)
                pm.Normal("obs",mu=mu,sigma=s,observed=yd)
                idata=pm.sample(**SAMP); pm.set_data({"X":Xte})
                pp=pm.sample_posterior_predictive(idata,var_names=["mu"],
                                                  progressbar=False,random_seed=SEED)
            yt.extend(y[te]); yp.extend(pp.posterior_predictive["mu"].mean(dim=["chain","draw"]).values)
        yt,yp=np.array(yt),np.array(yp)
        r2=r2_score(yt,yp); rmse=float(np.sqrt(mean_squared_error(yt,yp)))

        # final model on all data: R-hat + ELPD vs intercept-only null
        avg=int(np.round(np.mean(dims)))
        Xall=PCA(n_components=avg,random_state=SEED).fit_transform(X)
        with pm.Model():
            Xd=pm.Data("X",Xall); yd=pm.Data("y",y)
            mu=pmb.BART("mu",Xd,yd,**BART_PARAMS); s=pm.HalfNormal("sigma",5)
            pm.Normal("obs",mu=mu,sigma=s,observed=yd)
            ib=pm.sample(**SAMP,idata_kwargs={"log_likelihood":True})
        with pm.Model():
            mu=pm.Normal("mu",mu=y.mean(),sigma=y.std()*2); s=pm.HalfNormal("sigma",5)
            pm.Normal("obs",mu=mu,sigma=s,observed=y)
            inl=pm.sample(**SAMP,idata_kwargs={"log_likelihood":True})
        rh=az.rhat(ib)
        rmu=np.atleast_1d(rh['mu'].values).ravel(); rsig=float(np.atleast_1d(rh['sigma'].values).ravel()[0])
        essmu=np.atleast_1d(az.ess(ib)['mu'].values).ravel()
        c=az.compare({"BART":ib,"Null":inl},ic="loo")
        res[tgt]=dict(n=int(len(y)),r2=float(r2),rmse=rmse,dims=dims,
                      rhat_sigma=rsig,rhat_mu_max=float(np.nanmax(rmu)),
                      rhat_mu_mean=float(np.nanmean(rmu)),
                      rhat_mu_frac_gt101=float(np.mean(rmu>1.01)),
                      ess_mu_min=float(np.nanmin(essmu)),
                      elpd_diff=float(c['elpd_diff'].max()),dse=float(c['dse'].max()))
        print(f"BART4 {tgt:7} n={len(y):4} R2={r2:+.4f} RMSE={rmse:.4f} "
              f"dELPD={c['elpd_diff'].max():.2f} DSE={c['dse'].max():.2f} "
              f"rhat_sigma={rsig:.4f} rhat_mu_max={np.nanmax(rmu):.4f} "
              f"mu>1.01={np.mean(rmu>1.01):.1%} essmin={np.nanmin(essmu):.0f} "
              f"[{time.time()-t0:.0f}s]",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    print("DONE",flush=True)

if __name__=="__main__":
    import multiprocessing as mp; mp.freeze_support(); main()
