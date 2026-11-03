"""BART out-of-fold predictions + full metric breakdown, all 7 dimensions.

Produces everything BART_TRIALS.py reports, plus saves the raw out-of-fold
predictions so any further breakdown can be recomputed without re-sampling:

  global      R2, RMSE
  by text length quintile (LQ1..LQ5)   R2 and RMSE
  by observed score quintile (Q1..Q5)  RMSE
  ELPD difference vs intercept-only null, and its DSE

Quintile logic is copied from BART_TRIALS.py so numbers are comparable:
bins from np.quantile(.2,.4,.6,.8) + np.digitize, metric skipped when n<=5.

4 chains (BART_TRIALS.py uses 2). Point estimates are stable across chain
count -- verified: Confidence R2 0.3640 at 2 chains vs 0.3655 at 4 -- so
these match Table 1 within ~0.006. Chain count was raised because PyMC needs
>=4 chains for reliable R-hat. Resumable: finished dimensions are skipped.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED=42
BART_PARAMS={"m":100,"alpha":0.95,"beta":2.0,"response":"mix"}
SAMP={"draws":1000,"tune":500,"chains":4,"progressbar":False,"random_seed":SEED}
PAIRS={"conf":"text_confidence","cmpcy":"text_complacency","const":"text_constraints",
       "calc":"text_calculation","colr":"text_coll_resp","compli":"text_compliance",
       "consp":"text_conspiracy"}
PAT=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'
OUT=str(paths.RESULTS/"results_bart_full_metrics.json")
PRED=str(paths.RESULTS/"results_bart_oof_predictions.npz")
# ELPD/DSE and R-hat already computed by BART_4CHAIN_CHECK.py under the same
# configuration; reused here so the final+null models are not refitted.
_PRIOR_PATH=paths.RESULTS/"results_bart_4chain_partial.json"
PRIOR=json.load(open(_PRIOR_PATH)) if _PRIOR_PATH.exists() else {}

def quintile_metrics(values, y_true, y_pred, prefix, with_r2):
    """RMSE (and optionally R2) within quintiles of `values`. Mirrors BART_TRIALS.py."""
    q=np.quantile(values,[0.2,0.4,0.6,0.8])
    cat=np.digitize(values,bins=q)
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
    np.random.seed(SEED); random.seed(SEED)
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    dc=pd.read_csv(paths.MERGED_CLEANED)
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    preds=dict(np.load(PRED,allow_pickle=True)) if os.path.exists(PRED) else {}
    todo={k:v for k,v in PAIRS.items() if k not in res}
    if not todo:
        print("all 7 dimensions already present in",OUT,flush=True); return
    if res: print(f"resuming — done: {list(res)}",flush=True)
    print(f"running: {list(todo)}",flush=True)

    for tgt,tcol in todo.items():
        t0=time.time()
        m=dc[tcol].reindex(data.index,fill_value="").astype(str).str.contains(PAT,regex=True,na=False)
        df=data[[tgt,tcol]].dropna(); df=df[~m.loc[df.index]]
        df=df[df[tcol].astype(str).str.split().str.len()>=1]
        emb=[c for c in data.columns if c.startswith(f"{tcol}_emb_")]
        X=data.loc[df.index,emb].values.astype(np.float32)
        y=df[tgt].values.astype(np.float32)
        texts=df[tcol].values.astype(str)

        kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
        yt,yp,tx,dims=[],[],[],[]
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
            tx.extend(texts[te])
        yt=np.array(yt); yp=np.array(yp)
        lens=np.array([len(t.split()) for t in tx],dtype=float)
        preds[f"{tgt}_true"]=yt; preds[f"{tgt}_pred"]=yp; preds[f"{tgt}_len"]=lens
        np.savez_compressed(PRED,**preds)

        row=dict(n=int(len(yt)), dims=dims,
                 r2=float(r2_score(yt,yp)),
                 rmse=float(np.sqrt(mean_squared_error(yt,yp))))
        row.update(quintile_metrics(lens,yt,yp,"LQ",with_r2=True))   # by text length
        row.update(quintile_metrics(yt,  yt,yp,"Q", with_r2=False))  # by observed score

        # ELPD vs intercept-only null.
        # The 4-chain run (BART_4CHAIN_CHECK.py) already fitted the final and
        # null models under an identical configuration, so reuse its numbers
        # rather than re-sampling -- that is ~25% of the per-dimension cost.
        if tgt in PRIOR:
            pr=PRIOR[tgt]
            for f in ("elpd_diff","dse","rhat_sigma","rhat_mu_max","rhat_mu_frac_gt101"):
                row[f]=pr.get(f)
            row["elpd_source"]="reused from results_bart_4chain_partial.json"
        else:
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
            c=az.compare({"BART":ib,"Null":inl},ic="loo")
            row["elpd_diff"]=float(c['elpd_diff'].max()); row["dse"]=float(c['dse'].max())
            rh=az.rhat(ib); rmu=np.atleast_1d(rh['mu'].values).ravel()
            row["rhat_sigma"]=float(np.atleast_1d(rh['sigma'].values).ravel()[0])
            row["rhat_mu_max"]=float(np.nanmax(rmu))
            row["rhat_mu_frac_gt101"]=float(np.mean(rmu>1.01))
            row["elpd_source"]="fitted here"
        row["elpd_ratio"]=(row["elpd_diff"]/row["dse"]) if row.get("dse") else None

        res[tgt]=row; json.dump(res,open(OUT,'w'),indent=1)
        print(f"FULL {tgt:7} R2={row['r2']:+.4f} RMSE={row['rmse']:.4f} "
              f"dELPD={row['elpd_diff']:.2f} DSE={row['dse']:.2f} [{time.time()-t0:.0f}s]",flush=True)
    print("DONE",flush=True)

if __name__=="__main__":
    import multiprocessing as mp; mp.freeze_support(); main()
