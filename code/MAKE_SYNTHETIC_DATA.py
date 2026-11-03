"""Generate a synthetic dataset with the real schema, so the pipeline can be run.

The problem this solves
-----------------------
Stored results let a reader VERIFY every reported number. They do not let anyone
REFIT the models, because that needs the embeddings, which are not published
(they are invertible back to the response text — see README_RELEASE.md).

Without something to run against, a reader cannot tell a correct pipeline from a
broken one: the code is unexecutable for them. This writes a dataset with the
same shape, column names, dtypes and value ranges as the real one, so every
script runs end to end on a laptop in minutes.

What it is and is not
---------------------
It is **not** a privacy-preserving copy of the data. It is not resampled,
perturbed or synthesised from the real responses, and no real value is used to
seed it. The 7C scores are drawn from the published group means and SDs in
Supplementary Table 4; the embeddings are random vectors with a planted linear
signal of a chosen strength, which is only there so the models return something
other than noise.

So: numbers from this dataset will NOT match the paper, and are not meant to.
Every file it writes is prefixed `SYNTHETIC_` and carries a `synthetic` column
set to 1, so a run against it cannot be mistaken for a run against real data.

Usage
-----
    python code/MAKE_SYNTHETIC_DATA.py --out data_synthetic/

Then point the pipeline at it:

    PAPER2_DATA=data_synthetic python code/RIDGE_UNSCALED_CHECK.py

Writes files named exactly as the real ones so no script needs editing.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED = 20261029          # deliberately not 42: this is not a replication seed
N = 777
N_EMB = 1024
DIMS = ["conf", "cmpcy", "const", "calc", "colr", "compli", "consp"]
TEXT = {"conf": "text_confidence", "cmpcy": "text_complacency",
        "const": "text_constraints", "calc": "text_calculation",
        "colr": "text_coll_resp", "compli": "text_compliance",
        "consp": "text_conspiracy"}
# Published group means/SDs (Supplementary Table 4, readiness coding). Used only
# to put the synthetic scores on a plausible scale.
PUBLISHED = {
    "conf":   ((4.94, 1.23), (2.69, 1.25)),
    "cmpcy":  ((5.80, 1.00), (4.08, 1.51)),
    "const":  ((4.64, 1.23), (3.34, 1.14)),
    "calc":   ((3.13, 1.14), (2.94, 1.30)),
    "colr":   ((5.46, 1.37), (3.21, 1.64)),
    "compli": ((3.84, 1.48), (2.05, 1.13)),
    "consp":  ((5.86, 0.98), (3.84, 1.45)),
}
# Roughly the published out-of-fold R2, so each dimension is about as learnable
# as its real counterpart and a pipeline bug shows up as the wrong *pattern*.
# A first version planted the signal in a single direction, which PCA then lost
# among 1024 noise dimensions -- every R2 came out at zero, so a working
# pipeline and a broken one looked identical. The signal is now spread over
# K_SIGNAL dimensions and its strength is CALIBRATED per dimension by searching
# for the loading that hits the target, because the relationship between loading
# and out-of-fold R2 is not analytic here (777 rows, 1024 columns, PCA at 95%).
SIGNAL = {"conf": .40, "cmpcy": .12, "const": .05, "calc": .07,
          "colr": .42, "compli": .39, "consp": .20}
K_SIGNAL = 200
LOREM = ("Dies ist ein synthetischer Platzhaltertext ohne Bezug zu einer realen "
         "Person. Er existiert nur, damit die Textspalten die richtige Form haben.")


def oof_r2(X, y):
    """Pooled out-of-fold R2 under the same pipeline the paper uses."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(PCA(n_components=0.95, random_state=42),
                         RidgeCV(alphas=[10.0 ** k for k in range(-3, 9)]))
    pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42))
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def calibrate(base, idx, z, target, lo=0.0, hi=0.35, iters=7):
    """Bisect the loading until out-of-fold R2 lands on `target`.

    Monotone in `a`, so bisection is safe. Seven iterations gets within about
    0.01 of the target, which is far closer than this needs to be.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        X = base.copy()
        X[:, idx] += (mid * z[:, None]).astype(np.float32)
        if oof_r2(X, z) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synthetic",
                    help="directory to write into (default: data_synthetic/)")
    args = ap.parse_args()
    out = paths.PROJECT_ROOT / args.out
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    hes = np.zeros(N, dtype=int)
    hes[rng.choice(N, 166, replace=False)] = 1      # the real 166 / 611 split

    frame = {"Unnamed: 0": np.arange(N), "Participant ID": np.arange(1, N + 1),
             "Hesitant": hes, "synthetic": 1,
             "age": rng.integers(18, 75, N),
             "health literacy": rng.integers(1, 8, N)}
    for col, vals in [("gender", ["Male", "Female"]),
                      ("education", ["High", "Medium", "Low"]),
                      ("employment", ["Employed", "Unemployed"]),
                      ("city_size", ["Rural", "Urban"]),
                      ("party", ["Left", "Middle", "Right", "Other"]),
                      ("religious", ["Ja", "Nein", "Keine Angabe"]),
                      ("In welchem Land leben Sie derzeit?",
                       ["Deutschland", "Österreich", "Schweiz"])]:
        frame[col] = rng.choice(vals, N)

    emb_block = {}
    achieved = {}
    print("calibrating signal strength per dimension (a few seconds each)...")
    for d in DIMS:
        (mn, sn), (mh, sh) = PUBLISHED[d]
        score = np.where(hes == 1, rng.normal(mh, sh, N), rng.normal(mn, sn, N))
        score = np.clip(score, 1, 7)
        # Plant a linear signal of the target strength in a handful of directions,
        # then add noise sized to hit roughly that R2.
        z = (score - score.mean()) / score.std()
        base = rng.normal(0, 1, (N, N_EMB)).astype(np.float32)
        idx = rng.choice(N_EMB, K_SIGNAL, replace=False)
        a = calibrate(base, idx, z, SIGNAL[d])
        X = base.copy()
        X[:, idx] += (a * z[:, None]).astype(np.float32)
        achieved[d] = oof_r2(X, z)
        frame[d] = np.round(score * 3) / 3               # multiples of 1/3, as real scores are
        frame[TEXT[d]] = [LOREM] * N
        frame[f"{TEXT[d]}_questionnaire"] = [LOREM] * N
        # Items must be generated FROM the dimension score, not independently:
        # three independent uniform draws are uncorrelated within a dimension, the
        # dimension mean becomes meaningless, and anything checking item alignment
        # (TABLE4_NATIVE_DIRECTION) correctly refuses to run. Noise is sized to
        # give an alpha in the range the real scale shows.
        for i in range(3):
            item = score + rng.normal(0, 0.9, N)
            frame[f"{d}_{i+1}"] = np.clip(np.round(item), 1, 7).astype(int)
        for j in range(N_EMB):
            emb_block[f"{TEXT[d]}_emb_{j}"] = X[:, j]

    df = pd.DataFrame({**frame, **emb_block})
    f1 = out / "final_data_embeddings_with_questionnaire.csv"
    df.to_csv(f1, index=False)

    # response-only variant: same minus the _questionnaire text columns
    df.drop(columns=[c for c in df.columns if c.endswith("_questionnaire")]
            ).to_csv(out / "final_data_embeddings.csv", index=False)

    non_emb = [c for c in df.columns if "_emb_" not in c]
    df[non_emb].to_csv(out / "final_data_merged_cleaned.csv", index=False)
    df[non_emb].to_csv(out / "final_data_merged.csv", index=False)

    (out / "README.txt").write_text(
        "SYNTHETIC DATA — NOT REAL PARTICIPANT RESPONSES.\n\n"
        "Generated by code/MAKE_SYNTHETIC_DATA.py. No real value was used to\n"
        "produce any number here; the response text is a fixed placeholder\n"
        "sentence repeated for every row, and every row carries synthetic=1.\n\n"
        "Purpose: let the analysis pipeline be executed end to end without\n"
        "access to the real data, so the code can be read, run and debugged.\n"
        "Results from this dataset will NOT match the published numbers and\n"
        "are not intended to. To reproduce the paper you need the real\n"
        "embeddings, which are available on request under a data use\n"
        "agreement — see README.md.\n", encoding="utf8")

    print("\nrecoverable signal in the synthetic embeddings:")
    print(f"  {'dimension':10}{'target R2':>11}{'achieved':>10}   (published, for reference)")
    for d in DIMS:
        print(f"  {d:10}{SIGNAL[d]:>11.2f}{achieved[d]:>10.3f}")
    print("  These are properties of the synthetic data, NOT replications of the paper.")

    print(f"\nwrote {f1.name}  {df.shape[0]} x {df.shape[1]}")
    print(f"      final_data_embeddings.csv, final_data_merged{{,_cleaned}}.csv")
    print(f"      into {out}")
    print(f"\ntotal size: {sum(f.stat().st_size for f in out.glob('*.csv'))/1e6:.0f} MB")
    print("\nRun the pipeline against it with:")
    print(f"    PAPER2_DATA={args.out} python code/RIDGE_UNSCALED_CHECK.py")


if __name__ == "__main__":
    main()
