"""Categorical sentiment regression: permutation tests and out-of-fold R².

Fills two gaps in Supplementary Table 16.

1. PERMUTATION TESTS. Methods states "Permutation tests with 1,000 shuffled
   outcomes evaluated the coefficients." The test exists in the original Colab
   script (paper_2_regression_sentiment_num_scores.py, permutation_test_categorical)
   but its results were never reported -- Table 16 carries only the analytic OLS
   p-values. This runs it under the per-dimension masking used everywhere else.

2. OUT-OF-FOLD R². Table 16's R² is in-sample OLS, while every R² it invites
   comparison with (Tables 1, 10, 15) is pooled out-of-fold. In-sample R² is
   biased upward, so quoting the two side by side is not a like-for-like
   comparison -- it happens to understate the embedding advantage, but it is not
   defensible as written. This adds the out-of-fold value from the same folds,
   seed and pooling rule as the embedding models.

Both use KFold(5, shuffle=True, random_state=42) and 1,000 permutations with
Bonferroni correction across the seven dimensions, matching the ridge
permutation tests.

Run with the BART env:
    /opt/anaconda3/envs/BART/bin/python code/SENTIMENT_FINAL.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED = 42
N_PERM = 1000
ORDER = ["conf", "cmpcy", "const", "calc", "colr", "compli", "consp"]
NM = {"conf": "Confidence", "cmpcy": "Complacency", "const": "Constraints",
      "calc": "Calculation", "colr": "Collective Responsibility",
      "compli": "Compliance", "consp": "Conspiracy"}


def design(sub):
    du = pd.get_dummies(sub["label"])
    for c in ("positive", "negative", "neutral"):
        if c not in du:
            du[c] = 0
    X = du[["positive", "negative"]].astype(float).to_numpy()
    return np.column_stack([np.ones(len(X)), X])


def oof_r2(X, y):
    """Pooled out-of-fold R², same folds and pooling as the embedding models."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    yt, yp = [], []
    for tr, te in kf.split(X):
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        yt.extend(y[te]); yp.extend(X[te] @ beta)
    return float(r2_score(np.array(yt), np.array(yp)))


def main():
    rng = np.random.default_rng(SEED)
    lab = pd.read_csv(paths.result("results_sentiment_labels.csv"))
    data = pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    res = {}

    hdr = (f"{'Dimension':26}{'n':>5}{'pos':>8}{'p_ana':>8}{'p_perm':>8}"
           f"{'neg':>8}{'p_ana':>8}{'p_perm':>8}{'R2_in':>8}{'R2_oof':>8}")
    print(hdr); print("-" * len(hdr))

    for k in ORDER:
        sub = lab[lab.dim == NM[k]].copy()
        sub[k] = data.loc[sub.row.values, k].values
        sub = sub.dropna(subset=[k, "label"])
        X = design(sub)
        y = sub[k].astype(float).to_numpy()

        m = sm.OLS(y, X).fit()
        b_pos, b_neg = float(m.params[1]), float(m.params[2])

        # permutation null for both coefficients simultaneously
        ge_pos = ge_neg = 0
        for _ in range(N_PERM):
            yp = rng.permutation(y)
            beta, *_ = np.linalg.lstsq(X, yp, rcond=None)
            if abs(beta[1]) >= abs(b_pos): ge_pos += 1
            if abs(beta[2]) >= abs(b_neg): ge_neg += 1
        p_pos = (ge_pos + 1) / (N_PERM + 1)
        p_neg = (ge_neg + 1) / (N_PERM + 1)

        res[k] = dict(
            name=NM[k], n=int(len(sub)),
            intercept=float(m.params[0]),
            coef_positive=b_pos, p_positive=float(m.pvalues[1]),
            p_positive_perm=p_pos, p_positive_perm_bonf=min(1.0, p_pos * 7),
            coef_negative=b_neg, p_negative=float(m.pvalues[2]),
            p_negative_perm=p_neg, p_negative_perm_bonf=min(1.0, p_neg * 7),
            r_squared_insample=float(m.rsquared),
            r_squared_oof=oof_r2(X, y))
        r = res[k]
        print(f"{NM[k]:26}{r['n']:>5}{b_pos:>8.3f}{r['p_positive']:>8.3f}{p_pos:>8.3f}"
              f"{b_neg:>8.3f}{r['p_negative']:>8.3f}{p_neg:>8.3f}"
              f"{r['r_squared_insample']:>8.3f}{r['r_squared_oof']:>8.3f}", flush=True)

    json.dump(res, open(paths.RESULTS/"results_sentiment_final.json", "w"), indent=1)
    df = pd.DataFrame([res[k] for k in ORDER])
    df.to_csv(paths.OUTPUT / "sentiment_regression_final.csv", index=False)

    # --- how much of the embedding models' performance is sentiment? ---
    ridge = json.load(open(paths.result("results_ridge_unscaled_permutation.json")))
    print(f"\n{'Dimension':26}{'sentiment R2 oof':>18}{'embedding R2 oof':>18}{'ratio':>8}")
    for k in ORDER:
        s, e = res[k]["r_squared_oof"], ridge[k]["r2"]
        ratio = f"{s/e:.2f}" if e > 0.01 else "n/a"
        print(f"{NM[k]:26}{s:>18.3f}{e:>18.3f}{ratio:>8}")
    print("\nwrote", paths.RESULTS/"results_sentiment_final.json")
    print("wrote", paths.OUTPUT / "sentiment_regression_final.csv")


if __name__ == "__main__":
    main()
