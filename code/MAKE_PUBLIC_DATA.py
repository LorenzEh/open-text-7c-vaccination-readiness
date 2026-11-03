"""Build the participant-level data that can actually be published.

The privacy notice participants were shown permits depositing *anonymised* data
in a public archive, and promises that "Rückschlüsse auf Ihre oder die Identität
anderer Personen sind nicht möglich". That is a claim about re-identification,
so this script measures it rather than asserting it.

Three kinds of variable, three different risks
----------------------------------------------
**7C scale scores** — seven dimension means per participant. Near-unique as a
combination, but that does not make them identifying: re-identification needs a
second source holding the same attributes, and nobody else has these people's 7C
scores. This is the layer psychology papers routinely deposit. Published.

**Demographics** — age, gender, education, employment, place, country, politics,
religion. These ARE linkable to external data, which is what makes them quasi-
identifiers. Uncoarsened they single out 65% of the sample (505 of 777 unique on
the combination), so publishing them as collected would break the promise above.
Coarsened here until every combination covers at least K_MIN people.

**Embeddings** — not published at any coarsening. They are invertible back to the
response text (see README), so the question is disclosure, not linkage, and no
amount of aggregation over participants addresses it.

Outputs
    output/public_7c_scores.csv        777 x 7, scale scores + hesitancy group
    output/public_demographics.csv     coarsened to k >= K_MIN
    output/public_data_risk_report.txt what was coarsened and what k was reached

Run with the BART env:
    /opt/anaconda3/envs/BART/bin/python code/MAKE_PUBLIC_DATA.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

K_MIN = 5          # no combination of quasi-identifiers may describe fewer people
DIMS = ["conf", "cmpcy", "const", "calc", "colr", "compli", "consp"]
COUNTRY_COL = "In welchem Land leben Sie derzeit?"
QI = ["age_band", "gender", "education", "employment", "city_size",
      "country", "party", "religious"]

AGE_BANDS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 120)]
COUNTRY = {"Deutschland": "Germany", "Österreich": "Austria", "Schweiz": "Switzerland"}
RELIGIOUS = {"Ja": "Yes", "Nein": "No", "Keine Angabe": "Not specified"}


def band_age(a):
    if pd.isna(a):
        return "Not reported"
    for lo, hi in AGE_BANDS:
        if lo <= a <= hi:
            return f"{lo}-{hi}" if hi < 120 else f"{lo}+"
    return "Not reported"


def k_profile(df, cols):
    g = df.groupby(cols, dropna=False).size()
    return dict(combinations=len(g), k_min=int(g.min()),
                unique=int((g == 1).sum()), under_k=int(g[g < K_MIN].sum()))


def main():
    d = pd.read_csv(paths.MERGED_CLEANED)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- 7C scores: published as-is -------------------------------------
    scores = d[DIMS].copy()
    scores.insert(0, "participant", np.arange(1, len(d) + 1))   # sequential, not a Prolific ID
    scores["hesitant"] = d["Hesitant"].astype(int)
    scores.to_csv(paths.OUTPUT / "public_7c_scores.csv", index=False)
    say(f"7C scores: {len(scores)} rows x {len(DIMS)} dimensions, readiness-coded "
        f"(higher = greater readiness), no identifiers.")
    say("  Published without coarsening: these are not linkable to any external source.")
    say()

    # ---- demographics: coarsen until k >= K_MIN -------------------------
    dem = pd.DataFrame({
        "participant": scores["participant"],
        "hesitant": scores["hesitant"],
        "age_band": d["age"].map(band_age),
        "gender": d["gender"].fillna("Other / not reported"),
        "education": d["education"].fillna("Not reported"),
        "employment": d["employment"],
        "city_size": d["city_size"],
        "country": d[COUNTRY_COL].map(COUNTRY),
        "party": d["party"],
        "religious": d["religious"].map(RELIGIOUS),
    })

    raw = d.assign(age_band=d["age"], country=d[COUNTRY_COL])
    before = k_profile(raw, ["age", "gender", "education", "employment",
                             "city_size", "country", "party", "religious"])
    say(f"demographics as collected: {before['combinations']} combinations, "
        f"{before['unique']} people unique ({100*before['unique']/len(d):.0f}%), "
        f"{before['under_k']} in groups smaller than {K_MIN}")

    steps = [("age banded into 5 groups", None)]
    after = k_profile(dem, QI)
    say(f"  after banding age: k_min = {after['k_min']}, "
        f"{after['unique']} still unique")

    # Coarsen further, cheapest information loss first, until k >= K_MIN.
    if after["k_min"] < K_MIN:
        dem["country"] = dem["country"].replace({"Switzerland": "Austria/Switzerland",
                                                 "Austria": "Austria/Switzerland"})
        steps.append(("Austria and Switzerland merged (Switzerland n = 22)", None))
        after = k_profile(dem, QI)
        say(f"  after merging AT/CH: k_min = {after['k_min']}, "
            f"{after['unique']} still unique")

    if after["k_min"] < K_MIN:
        dem = dem.drop(columns=["party"])
        QI.remove("party")
        steps.append(("political leaning dropped from the row-level file "
                      "(it remains in the aggregate Supplementary Table 3)", None))
        after = k_profile(dem, QI)
        say(f"  after dropping politics: k_min = {after['k_min']}, "
            f"{after['unique']} still unique")

    if after["k_min"] < K_MIN:
        dem = dem.drop(columns=["religious"])
        QI.remove("religious")
        steps.append(("religion dropped from the row-level file "
                      "(it remains in the aggregate Supplementary Table 3)", None))
        after = k_profile(dem, QI)
        say(f"  after dropping religion: k_min = {after['k_min']}, "
            f"{after['unique']} still unique")

    say()
    if after["k_min"] >= K_MIN:
        dem.to_csv(paths.OUTPUT / "public_demographics.csv", index=False)
        say(f"RESULT: k = {after['k_min']} achieved over {after['combinations']} "
            f"combinations. Written to output/public_demographics.csv")
    else:
        say(f"RESULT: k = {after['k_min']} after all coarsening steps, below the "
            f"K_MIN = {K_MIN} target. Row-level demographics NOT written.")
        say("        Deposit the aggregate table (Supplementary Table 3) instead, or")
        say("        lower the number of variables released at row level.")

    say()
    say("Coarsening applied:")
    for s, _ in steps:
        say(f"  - {s}")
    say()
    say("NOT published at any coarsening: the embeddings "
        "(data/final_data_embeddings*.csv) and the response text. Sentence "
        "embeddings are invertible back to the text, so the issue is direct "
        "disclosure rather than linkage, and coarsening over participants does "
        "not address it. See README_RELEASE.md.")

    (paths.OUTPUT / "public_data_risk_report.txt").write_text("\n".join(lines) + "\n",
                                                              encoding="utf8")
    say()
    print(f"wrote {paths.OUTPUT/'public_data_risk_report.txt'}")


if __name__ == "__main__":
    main()
