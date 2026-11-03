"""Fast end-to-end check: is this installation able to reproduce the paper?

Run this first. It takes seconds, touches no model fitting beyond one small
ridge, and answers three questions in order:

    1. Can the environment import everything the pipeline needs?
    2. Do the stored results reproduce the published numbers?       (no data needed)
    3. Can the modelling pipeline actually run on the data present? (data needed)

Step 2 is the one that matters for verification and it works for everybody,
because `results/` ships with the release. Step 3 is skipped with a clear
message when no dataset is present, and reports whether it ran against the real
data or the synthetic stand-in.

    python code/VERIFY_INSTALL.py
    PAPER2_DATA=data_synthetic python code/VERIFY_INSTALL.py
"""
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

# (R2, RMSE) as printed in Table 1, ridge rows, unstandardised specification.
PUBLISHED_RIDGE = {
    "conf": (0.427, 1.163), "cmpcy": (0.150, 1.206), "const": (0.038, 1.298),
    "calc": (0.083, 1.128), "colr": (0.436, 1.271), "compli": (0.389, 1.246),
    "consp": (0.212, 1.224),
}
PUBLISHED_BART_R2 = {"conf": 0.366, "cmpcy": 0.089, "const": 0.041, "calc": 0.058,
                     "colr": 0.339, "compli": 0.392, "consp": 0.181}

fails = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def main():
    print("1. environment")
    for mod, need in [("numpy", True), ("pandas", True), ("sklearn", True),
                      ("scipy", True), ("matplotlib", True),
                      ("pymc", False), ("pymc_bart", False)]:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            check(f"import {mod}", True, f"v{v}")
        except ImportError:
            if need:
                check(f"import {mod}", False, "required")
            else:
                print(f"  SKIP  import {mod}  (only needed to refit BART)")

    print("\n2. stored results reproduce the published numbers (no data needed)")
    rp = paths.RESULTS_PUBLISHED   # always the real ones, even under PAPER2_DATA
    try:
        ridge = json.load(open(rp / "results_ridge_unscaled_permutation.json"))
        bad = [k for k, (r2, rmse) in PUBLISHED_RIDGE.items()
               if abs(ridge[k]["r2"] - r2) > 5e-4 or abs(ridge[k]["rmse"] - rmse) > 5e-4]
        check("Table 1 ridge R2 and RMSE, all 7 dimensions", not bad,
              "" if not bad else f"mismatched: {bad}")
        perm = all(ridge[k]["p_bonf"] <= .05 for k in PUBLISHED_RIDGE)
        check("all 7 permutation tests clear Bonferroni", perm)
    except FileNotFoundError as e:
        check("ridge results present", False, str(e))

    try:
        import numpy as np
        d = np.load(rp / "results_bart_oof_predictions.npz")
        bad = []
        for k, r2 in PUBLISHED_BART_R2.items():
            y, p = d[f"{k}_true"], d[f"{k}_pred"]
            got = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
            if abs(got - r2) > 5e-4:
                bad.append(f"{k}: {got:.3f} vs {r2}")
        check("Table 1 BART R2 recomputed from stored predictions", not bad,
              "" if not bad else "; ".join(bad))
    except FileNotFoundError as e:
        check("BART predictions present", False, str(e))

    # Cronbach's alpha is included because this file has been silently clobbered
    # once by a run against substitute data. Anything a run can overwrite should
    # be covered here, so corruption is caught by a 10-second check rather than
    # by noticing a wrong number in a table weeks later.
    PUBLISHED_ALPHA = {"colr": .927, "consp": .842, "conf": .833, "compli": .821,
                       "cmpcy": .801, "const": .681, "calc": .513}
    try:
        a = json.load(open(rp / "results_cronbach_alpha.json"))
        bad = [k for k, v in PUBLISHED_ALPHA.items() if abs(a[k]["alpha"] - v) > 5e-4]
        check("Supplementary Table 19 alphas", not bad,
              "" if not bad else f"mismatched: {bad}")
    except FileNotFoundError as e:
        check("Cronbach alpha results present", False, str(e))

    PUBLISHED_SENTIMENT_OOF = {"conf": .067, "colr": .040, "compli": .043,
                               "cmpcy": -.005, "const": -.010, "calc": -.008,
                               "consp": -.007}
    try:
        s = json.load(open(rp / "results_sentiment_final.json"))
        bad = [k for k, v in PUBLISHED_SENTIMENT_OOF.items()
               if abs(s[k]["r_squared_oof"] - v) > 1e-3]
        check("Supplementary Table 16 out-of-fold R2", not bad,
              "" if not bad else f"mismatched: {bad}")
    except (FileNotFoundError, KeyError) as e:
        check("sentiment results present", False, str(e))

    print("\n3. modelling pipeline runs on the data present")
    f = paths.EMBEDDINGS_WITH_QUESTIONNAIRE
    if not f.exists():
        print(f"  SKIP  no dataset at {f}")
        print("        The release ships results/ but not the participant data.")
        print("        To execute the pipeline, generate the synthetic stand-in:")
        print("            python code/MAKE_SYNTHETIC_DATA.py")
        print("            PAPER2_DATA=data_synthetic python code/VERIFY_INSTALL.py")
    else:
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import make_pipeline

        head = pd.read_csv(f, nrows=1)
        synthetic = "synthetic" in head.columns
        print(f"  data: {f.name}  ({'SYNTHETIC' if synthetic else 'real'})")

        cols = [c for c in head.columns if c.startswith("text_confidence_emb_")]
        t0 = time.time()
        df = pd.read_csv(f, usecols=cols + ["conf"]).dropna()
        X = df[cols].to_numpy(np.float32)
        y = df["conf"].to_numpy(float)
        pipe = make_pipeline(PCA(n_components=0.95, random_state=42),
                            RidgeCV(alphas=[10.0 ** k for k in range(-3, 9)]))
        pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42))
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        check("ridge pipeline fits and predicts (Confidence)", True,
              f"out-of-fold R2 = {r2:.3f}  [{time.time()-t0:.0f}s]")
        if synthetic:
            print(f"        Synthetic data: this number is NOT the published 0.427")
            print(f"        and is not meant to be. It confirms the code runs.")
        else:
            close = abs(r2 - 0.427) < 0.02
            check("matches the published Confidence R2 of 0.427", close,
                  f"got {r2:.3f}")

    print("\n" + ("ALL CHECKS PASSED" if not fails
                  else f"{len(fails)} CHECK(S) FAILED: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
