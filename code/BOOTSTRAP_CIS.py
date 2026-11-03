"""Bootstrap 95% confidence intervals for out-of-fold R2 and RMSE.

Answers the co-author request to add intervals to Table 1.

Method: BCa (bias-corrected and accelerated) percentile bootstrap over the
pooled out-of-fold predictions, 10,000 resamples. Observation pairs
(y_i, yhat_i) are resampled together, so the interval reflects sampling
variability in the evaluation set.

What this interval does and does not cover
------------------------------------------
It covers uncertainty from *which participants happened to be in the sample*.
It does NOT cover uncertainty from the particular cross-validation split, nor
MCMC sampling error in BART. The first of those would need repeated CV with
different seeds; the second is separately bounded -- doubling the chains moved
every R2 by <= 0.007. Both are small relative to the intervals below, but the
interval should be described as what it is rather than as a full accounting.

No model refitting: predictions are read from the saved .npz files.
"""
import os, sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED=42; NBOOT=10000
ORDER=["conf","cmpcy","const","calc","colr","compli","consp"]
NM={"conf":"Confidence","cmpcy":"Complacency","const":"Constraints","calc":"Calculation",
    "colr":"Collective Responsibility","compli":"Compliance","consp":"Conspiracy"}

def r2(y,p):
    y=np.asarray(y); p=np.asarray(p)
    return 1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
def rmse(y,p):
    y=np.asarray(y); p=np.asarray(p)
    return np.sqrt(np.mean((y-p)**2))

def ci(y,p,fn):
    rng=np.random.default_rng(SEED)
    res=stats.bootstrap((y,p), fn, paired=True, vectorized=False,
                        n_resamples=NBOOT, method="BCa",
                        confidence_level=0.95, random_state=rng)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)

def main():
    bart=np.load(paths.result("results_bart_oof_predictions.npz"))
    ridge=np.load(paths.result("results_ridge_oof_predictions.npz"))
    out={}
    for label,get in (("BART",     lambda k:(bart[f"{k}_true"], bart[f"{k}_pred"])),
                      ("Ridge",    lambda k:(ridge[f"unscaled_{k}_true"], ridge[f"unscaled_{k}_pred"]))):
        out[label]={}
        for k in ORDER:
            y,p=get(k)
            pt_r2, pt_rm = r2(y,p), rmse(y,p)
            lo_r2,hi_r2 = ci(y,p,r2)
            lo_rm,hi_rm = ci(y,p,rmse)
            out[label][k]=dict(n=int(len(y)),
                r2=pt_r2, r2_lo=lo_r2, r2_hi=hi_r2,
                rmse=pt_rm, rmse_lo=lo_rm, rmse_hi=hi_rm)
            print(f"{label:6} {NM[k]:26} R2={pt_r2:.3f} [{lo_r2:.3f}, {hi_r2:.3f}]"
                  f"   RMSE={pt_rm:.3f} [{lo_rm:.3f}, {hi_rm:.3f}]",flush=True)
        print()
    json.dump(out,open(paths.RESULTS / "results_bootstrap_cis.json","w"),indent=1)
    print("saved results_bootstrap_cis.json")

if __name__=="__main__":
    main()
