"""BART *with* standardisation before PCA -- the missing cell of the 2x2.

Ridge was published with StandardScaler before PCA; BART never had it. We have
measured three of the four preprocessing combinations:

                        standardised+PCA   unstandardised+PCA   no PCA
    Ridge                     yes                yes             yes
    BART                    THIS SCRIPT          yes             yes

Why it matters. The appendix and the methods documents assert a mechanism for
why standardisation hurts: all 1,024 embedding dimensions come from one encoder
and already share a scale, so standardising up-weights near-noise dimensions and
PCA on the resulting correlation matrix spends components describing noise.

That explanation is about what PCA extracts, not about the estimator, so it
predicts standardisation should hurt BART too -- by roughly the amount it hurt
ridge (mean -0.124). If BART is unaffected, the asserted mechanism is wrong or
incomplete and the appendix text needs revising.

Only difference from BART_FULL_METRICS.py: StandardScaler is fitted on the
training fold and applied to both folds before PCA. Same 4 chains, same seeds,
same masks, same 95% variance threshold. Resumable.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[v]='1'
os.environ['PYTHONHASHSEED']='0'
import sys, json, time, random
import numpy as np, pandas as pd, pymc as pm, pymc_bart as pmb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
OUT=str(paths.RESULTS/"results_bart_scaled.json")

def main():
    np.random.seed(SEED); random.seed(SEED)
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    dc=pd.read_csv(paths.MERGED_CLEANED)
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo={k:v for k,v in PAIRS.items() if k not in res}
    if not todo: print("nothing to do"); return
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
        kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
        yt,yp,dims=[],[],[]
        for fi,(tr,te) in enumerate(kf.split(X),1):
            ft=time.time()
            sc=StandardScaler().fit(X[tr])                 # <-- the only change
            Xtr_s, Xte_s = sc.transform(X[tr]), sc.transform(X[te])
            p=PCA(random_state=SEED).fit(Xtr_s)
            d=int(np.searchsorted(np.cumsum(p.explained_variance_ratio_),0.95)+1); dims.append(d)
            r=PCA(n_components=d,random_state=SEED)
            Xtr=r.fit_transform(Xtr_s); Xte=r.transform(Xte_s)
            with pm.Model():
                Xd=pm.Data("X",Xtr); yd=pm.Data("y",y[tr])
                mu=pmb.BART("mu",Xd,yd,**BART_PARAMS); s=pm.HalfNormal("sigma",5)
                pm.Normal("obs",mu=mu,sigma=s,observed=yd)
                idata=pm.sample(**SAMP); pm.set_data({"X":Xte})
                pp=pm.sample_posterior_predictive(idata,var_names=["mu"],
                                                  progressbar=False,random_seed=SEED)
            pred=pp.posterior_predictive["mu"].mean(dim=["chain","draw"]).values
            yt.extend(y[te]); yp.extend(pred)
            print(f"    fold {fi}/5 dims={d} rmse={np.sqrt(mean_squared_error(y[te],pred)):.4f} "
                  f"[{time.time()-ft:.0f}s]",flush=True)
        yt,yp=np.array(yt),np.array(yp)
        res[tgt]=dict(n=int(len(yt)),dims=dims,r2=float(r2_score(yt,yp)),
                      rmse=float(np.sqrt(mean_squared_error(yt,yp))))
        json.dump(res,open(OUT,'w'),indent=1)
        print(f"SCALED {tgt:7} R2={res[tgt]['r2']:+.4f} RMSE={res[tgt]['rmse']:.4f} "
              f"[{time.time()-t0:.0f}s]",flush=True)
    print("DONE",flush=True)

if __name__=="__main__":
    import multiprocessing as mp; mp.freeze_support(); main()
