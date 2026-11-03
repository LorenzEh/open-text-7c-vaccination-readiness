"""Cross-dimension prediction matrix -- discriminant validity.

For every pair (i, j): use dimension i's open-text embeddings to predict
dimension j's 7C score. 7 x 7 = 49 Ridge models per specification.

The question is whether each dimension's text predicts its OWN score better
than any other dimension's score (row-wise), and whether each score is best
predicted by its OWN text (column-wise). The diagonal reproduces the
per-dimension results already reported, which checks the matrix is correct.

Rows for cell (i, j): usable text for dimension i (exclusion mask + >=1 word)
AND non-missing score j. Ns therefore vary slightly across the matrix.
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
SEED=42; ALPHAS=[10.0**k for k in range(-3,9)]; PCA_VAR=0.95
PAIRS={"conf":"text_confidence","cmpcy":"text_complacency","const":"text_constraints",
       "calc":"text_calculation","colr":"text_coll_resp","compli":"text_compliance",
       "consp":"text_conspiracy"}
ORDER=list(PAIRS)
NM={"conf":"Confidence","cmpcy":"Complacency","const":"Constraints","calc":"Calculation",
    "colr":"Coll. Resp.","compli":"Compliance","consp":"Conspiracy"}
PAT=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'
OUT=str(paths.RESULTS/"results_discriminant_validity.json")

def build(spec):
    if spec=="scaled":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(),
                PCA(n_components=PCA_VAR,random_state=SEED),
                RidgeCV(alphas=ALPHAS,scoring='r2')),
            transformer=StandardScaler(), inverse_func=None)
    return make_pipeline(PCA(n_components=PCA_VAR,random_state=SEED),
                         RidgeCV(alphas=ALPHAS,scoring='r2'))

def main():
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    dc=pd.read_csv(paths.MERGED_CLEANED)
    masks={t:dc[c].reindex(data.index,fill_value="").astype(str)
             .str.contains(PAT,regex=True,na=False) for t,c in PAIRS.items()}
    res={}
    for spec in ("scaled","unscaled"):
        t0=time.time(); res[spec]={}
        for ti in ORDER:                     # text dimension (predictor)
            tcol=PAIRS[ti]
            Xc=[c for c in data.columns if c.startswith(f"{tcol}_emb_")]
            base=data[~masks[ti]]
            base=base[base[tcol].fillna("").astype(str).str.split().str.len()>=1]
            res[spec][ti]={}
            for tj in ORDER:                 # target dimension (outcome)
                sub=base[[tj]+Xc].dropna()
                X=sub[Xc].values; y=sub[tj].values
                cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
                yp=cross_val_predict(build(spec),X,y,cv=cv,n_jobs=-1)
                res[spec][ti][tj]=dict(n=int(len(y)),r2=float(r2_score(y,yp)),
                                       rmse=float(np.sqrt(mean_squared_error(y,yp))))
            print(f"  {spec:9} text={ti:7} done",flush=True)
        print(f"{spec} finished in {time.time()-t0:.0f}s",flush=True)
    json.dump(res,open(OUT,'w'),indent=1)

    for spec in ("scaled","unscaled"):
        M=np.array([[res[spec][i][j]["r2"] for j in ORDER] for i in ORDER])
        print(f"\n===== R² matrix, Ridge {spec} =====")
        hdr = "text \\ score"
        print(f"{hdr:14}"+"".join(f"{NM[j][:9]:>11}" for j in ORDER))
        for a,i in enumerate(ORDER):
            cells=""
            for b,j in enumerate(ORDER):
                mark="*" if a==b else " "
                cells+=f"{M[a,b]:>10.3f}{mark}"
            print(f"{NM[i][:13]:14}{cells}")
        rowwin=sum(M[a].argmax()==a for a in range(7))
        colwin=sum(M[:,b].argmax()==b for b in range(7))
        print(f"  diagonal is the best in its ROW    (own text predicts own score best): {rowwin}/7")
        print(f"  diagonal is the best in its COLUMN (own score best predicted by own text): {colwin}/7")
        for a,i in enumerate(ORDER):
            off=np.delete(M[a],a)
            print(f"    {NM[i]:22} own={M[a,a]:+.3f}  best other={off.max():+.3f} "
                  f"({NM[ORDER[[j for j in range(7) if j!=a][off.argmax()]]]})  margin={M[a,a]-off.max():+.3f}")
    print("\nDONE")

if __name__=="__main__":
    main()
