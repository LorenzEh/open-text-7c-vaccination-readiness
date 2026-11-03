"""Sentiment regressions with per-dimension masking, consistent with every
other analysis in the paper.

The published pipeline (paper_2_regression_sentiment_num_scores.py) applies
listwise deletion: any participant marked in ANY of the seven text columns is
dropped from all seven models, leaving 692 of 777. Every other analysis --
BART, ridge, the t-tests, the discriminant matrix -- masks per dimension and
keeps 713-769 per model. The sentiment results therefore rested on a smaller
and differently selected sample than Table 1.

This refits the same OLS models on the per-dimension samples. It reuses the
labels already produced by SENTIMENT_GROUP_TEST.py, which classified under
per-dimension masking, so no re-classification is required.
"""
import os, sys, json
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

ORDER=["conf","cmpcy","const","calc","colr","compli","consp"]
NM={"conf":"Confidence","cmpcy":"Complacency","const":"Constraints","calc":"Calculation",
    "colr":"Collective Responsibility","compli":"Compliance","consp":"Conspiracy"}
BY_NAME={v:k for k,v in NM.items()}

def main():
    lab=pd.read_csv(paths.result("results_sentiment_labels.csv"))
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    old=json.load(open(paths.result("results_sentiment_regression.json")))
    res={}
    print(f"{'Dimension':26}{'n':>6}{'intercept':>11}{'pos':>9}{'p':>8}{'neg':>9}{'p':>8}{'R²':>8}")
    for k in ORDER:
        sub=lab[lab.dim==NM[k]].copy()
        sub[k]=data.loc[sub.row.values, k].values
        sub=sub.dropna(subset=[k,'label'])
        du=pd.get_dummies(sub['label'])
        for c in ('positive','negative','neutral'):
            if c not in du: du[c]=0
        X=sm.add_constant(du[['positive','negative']].astype(float).reset_index(drop=True))
        m=sm.OLS(sub[k].astype(float).reset_index(drop=True),X).fit()
        res[k]=dict(name=NM[k],n=int(len(sub)),
            intercept=float(m.params['const']),
            coef_positive=float(m.params['positive']),p_positive=float(m.pvalues['positive']),
            coef_negative=float(m.params['negative']),p_negative=float(m.pvalues['negative']),
            r_squared=float(m.rsquared))
        r=res[k]
        print(f"{NM[k]:26}{r['n']:>6}{r['intercept']:>11.3f}{r['coef_positive']:>9.3f}"
              f"{r['p_positive']:>8.3f}{r['coef_negative']:>9.3f}{r['p_negative']:>8.3f}{r['r_squared']:>8.3f}")
    json.dump(res,open(paths.RESULTS/"results_sentiment_regression_perdim.json","w"),indent=1)

    print(f"\n{'Dimension':26}{'n listwise':>12}{'n per-dim':>11}{'ΔR²':>8}   coefficient shifts")
    for k in ORDER:
        a,b=old[k],res[k]
        print(f"{NM[k]:26}{a['n']:>12}{b['n']:>11}{b['r_squared']-a['r_squared']:>+8.3f}"
              f"   pos {a['coef_positive']:+.3f}→{b['coef_positive']:+.3f}"
              f"   neg {a['coef_negative']:+.3f}→{b['coef_negative']:+.3f}")
    # do any reported conclusions change sign or significance?
    print("\nSignificance changes at .05:")
    ch=0
    for k in ORDER:
        for f in ('p_positive','p_negative'):
            s1,s2=old[k][f]<.05, res[k][f]<.05
            if s1!=s2:
                ch+=1
                print(f"   {NM[k]} {f}: {old[k][f]:.3f} -> {res[k][f]:.3f}  ({'gained' if s2 else 'lost'})")
    if ch==0: print("   none")

if __name__=="__main__":
    main()
