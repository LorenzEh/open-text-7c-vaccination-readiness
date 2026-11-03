"""Cronbach's alpha per 7C dimension, with item statistics.

Fills the Methods placeholder "all Cronbach's α ≥ .XX" and supplies the
content for the supplementary table of item texts and internal consistency.

Computed on the recoded items in final_data_merged_cleaned.csv, i.e. after the
nine reverse-scored items have been transformed (8 - x) so that higher always
means greater vaccination readiness. That is the scoring the dimension means
use, so alpha describes the scales as analysed.

alpha = k/(k-1) * (1 - sum(item variances) / variance of the total score),
with k = 3 items per dimension. A 95% interval is obtained by bootstrapping
participants (5,000 resamples, percentile method).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED=42
SCALES={'conf':['conf_1','conf_2','conf_3'],'cmpcy':['cmpcy_1','cmpcy_2','cmpcy_3'],
        'const':['const_1','const_2','const_3'],'calc':['calc_1','calc_2','calc_3'],
        'colr':['colr_1','colr_2','colr_3'],'compli':['compli_1','compli_2','compli_3'],
        'consp':['consp_1','consp_2','consp_3']}
NM={"conf":"Confidence","cmpcy":"Complacency","const":"Constraints","calc":"Calculation",
    "colr":"Collective Responsibility","compli":"Compliance","consp":"Conspiracy"}
REVERSED={"cmpcy_1","cmpcy_2","const_3","calc_1","calc_2","calc_3",
          "consp_1","consp_2","consp_3"}

def alpha(X):
    """X: n x k array, listwise complete."""
    k=X.shape[1]
    if k<2: return np.nan
    item_var=X.var(axis=0,ddof=1).sum()
    total_var=X.sum(axis=1).var(ddof=1)
    if total_var==0: return np.nan
    return k/(k-1)*(1-item_var/total_var)

def main():
    d=pd.read_csv(paths.MERGED_CLEANED)
    rng=np.random.default_rng(SEED)
    out={}
    print(f"{'Dimension':26}{'n':>6}{'α':>8}{'95% CI':>18}{'mean r(item-total)':>21}")
    for k,cols in SCALES.items():
        X=d[cols].dropna().values.astype(float)
        a=alpha(X)
        boot=[alpha(X[rng.integers(0,len(X),len(X))]) for _ in range(5000)]
        lo,hi=np.percentile(boot,[2.5,97.5])
        # corrected item-total correlations, and alpha with each item dropped
        its, drop = {}, {}
        for j,c in enumerate(cols):
            rest=np.delete(X,j,axis=1).sum(axis=1)
            its[c]=float(np.corrcoef(X[:,j],rest)[0,1])
            drop[c]=float(alpha(np.delete(X,j,axis=1)))
        out[k]=dict(name=NM[k],n=int(len(X)),alpha=float(a),ci_low=float(lo),ci_high=float(hi),
                    item_total=its, alpha_if_dropped=drop,
                    reversed_items=[c for c in cols if c in REVERSED])
        print(f"{NM[k]:26}{len(X):>6}{a:>8.3f}   [{lo:.3f}, {hi:.3f}]{np.mean(list(its.values())):>21.3f}")
    json.dump(out,open(paths.RESULTS/"results_cronbach_alpha.json","w"),indent=1)
    al=[out[k]['alpha'] for k in SCALES]
    print(f"\nrange: {min(al):.3f} – {max(al):.3f}   |   lowest: "
          f"{NM[min(SCALES,key=lambda k: out[k]['alpha'])]}")
    print("\nAlpha if an item were dropped (flags items that hurt the scale):")
    for k in SCALES:
        o=out[k]
        worse=[(c,v) for c,v in o['alpha_if_dropped'].items() if v>o['alpha']+0.01]
        if worse:
            for c,v in worse:
                print(f"   {NM[k]:26} dropping {c} would RAISE α {o['alpha']:.3f} -> {v:.3f}")
    print("   (nothing listed = no item detracts by more than .01)")

if __name__=="__main__":
    main()
