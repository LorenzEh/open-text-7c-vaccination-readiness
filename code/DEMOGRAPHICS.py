"""Demographics table for the analytic sample (N = 777), by hesitancy group.

Recomputes every cell of appendix Table 3 from the canonical analytic file, with
explicit denominators. Table 3 as drafted has cells that do not sum to the group
size, so this script reports, for each variable:

    n valid           responses actually available for that variable
    n missing         group size - n valid
    % of valid        percentage within the responses available  (sums to 100)
    % of group        percentage of the whole group              (sums to <100
                      when the variable has missing data)

Reporting both makes the denominator unambiguous, which is what the drafted
table leaves implicit. Percentages that "do not add up" are almost always a
missing-data question, not an arithmetic error.

Outputs
    output/demographics_table.csv     long form, one row per category
    output/demographics_table.xlsx    same, plus a wide sheet formatted like the
                                      appendix table (written only if openpyxl
                                      is importable; the BART env lacks it)
    output/demographics_wide.csv      wide form, publication layout

Run with the BART env:
    /opt/anaconda3/envs/BART/bin/python code/DEMOGRAPHICS.py
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

GROUPS = [(0, "Non-hesitant"), (1, "Hesitant")]

# (variable label, column, category order or None for natural order)
CATEGORICAL = [
    ("Gender",            "gender",       ["Male", "Female"]),
    ("Country",           "country",      ["Germany", "Austria", "Switzerland"]),
    ("Place of residence", "city_size",   ["Rural", "Urban"]),
    ("Education",         "education",    ["High", "Medium", "Low"]),
    ("Employment",        "employment",   ["Employed", "Unemployed"]),
    ("Political leaning", "party",        ["Left", "Middle", "Right", "Other"]),
    ("Religious",         "religious",    ["Ja", "Nein", "Keine Angabe"]),
]
RELIGIOUS_LABEL = {"Ja": "Yes", "Nein": "No", "Keine Angabe": "Not specified"}
COUNTRY_LABEL = {"Deutschland": "Germany", "Österreich": "Austria", "Schweiz": "Switzerland"}
CONTINUOUS = [("Age (years)", "age"), ("Self-rated health literacy", "health literacy")]


def load():
    d = pd.read_csv(paths.MERGED_CLEANED)
    d["country"] = d["In welchem Land leben Sie derzeit?"].map(COUNTRY_LABEL)
    return d


def build(d):
    rows = []
    for hes, gname in GROUPS:
        g = d[d.Hesitant == hes]
        rows.append(dict(variable="Number total", category="", group=gname,
                         n=len(g), n_valid=len(g), n_missing=0,
                         pct_of_valid=np.nan, pct_of_group=np.nan,
                         mean=np.nan, sd=np.nan))
        for label, col, order in CATEGORICAL:
            s = g[col]
            valid = s.dropna()
            cats = order if order else sorted(valid.unique())
            for c in cats:
                n = int((valid == c).sum())
                disp = RELIGIOUS_LABEL.get(c, c) if col == "religious" else c
                rows.append(dict(variable=label, category=disp, group=gname,
                                 n=n, n_valid=len(valid), n_missing=len(s) - len(valid),
                                 pct_of_valid=100 * n / len(valid) if len(valid) else np.nan,
                                 pct_of_group=100 * n / len(g),
                                 mean=np.nan, sd=np.nan))
            # anything not covered by the declared category order
            extra = set(valid.unique()) - set(cats)
            for c in sorted(extra):
                n = int((valid == c).sum())
                rows.append(dict(variable=label, category=f"{c} (uncategorised)", group=gname,
                                 n=n, n_valid=len(valid), n_missing=len(s) - len(valid),
                                 pct_of_valid=100 * n / len(valid), pct_of_group=100 * n / len(g),
                                 mean=np.nan, sd=np.nan))
            miss = len(s) - len(valid)
            if miss:
                rows.append(dict(variable=label, category="Not reported", group=gname,
                                 n=miss, n_valid=len(valid), n_missing=miss,
                                 pct_of_valid=np.nan, pct_of_group=100 * miss / len(g),
                                 mean=np.nan, sd=np.nan))
        for label, col in CONTINUOUS:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            rows.append(dict(variable=label, category="", group=gname,
                             n=len(v), n_valid=len(v), n_missing=len(g) - len(v),
                             pct_of_valid=np.nan, pct_of_group=np.nan,
                             mean=float(v.mean()), sd=float(v.std(ddof=1))))
    return pd.DataFrame(rows)


def wide(long):
    """Publication layout: one row per category, one column pair per group."""
    out = []
    keys = long[["variable", "category"]].drop_duplicates().values.tolist()
    for var, cat in keys:
        row = {"Variable": var, "Category": cat}
        for _, gname in GROUPS:
            r = long[(long.variable == var) & (long.category == cat) & (long.group == gname)]
            if r.empty:
                row[gname] = ""
                continue
            r = r.iloc[0]
            if not np.isnan(r["mean"]):
                row[gname] = f"{r['mean']:.2f} (SD = {r['sd']:.2f})"
            elif var == "Number total":
                row[gname] = f"{int(r['n'])}"
            elif np.isnan(r["pct_of_valid"]):
                row[gname] = f"{r['pct_of_group']:.1f}% ({int(r['n'])})"
            else:
                row[gname] = f"{r['pct_of_valid']:.1f}% ({int(r['n'])})"
        out.append(row)
    return pd.DataFrame(out)


def checks(long):
    """Every categorical variable's percentages must sum to 100 within valid n."""
    print("\nInternal check — % of valid should sum to 100.0 for every variable/group:")
    ok = True
    for (var, gname), sub in long[long.pct_of_valid.notna()].groupby(["variable", "group"], sort=False):
        tot = sub.pct_of_valid.sum()
        ns = sub.n.sum()
        nv = sub.n_valid.iloc[0]
        flag = "" if (abs(tot - 100) < 0.06 and ns == nv) else "   <-- PROBLEM"
        if flag:
            ok = False
        print(f"  {var:26} {gname:13} sum={tot:6.1f}%  counts {ns}/{nv}{flag}")
    print("all variables reconcile" if ok else "SOME VARIABLES DO NOT RECONCILE")


def main():
    d = load()
    long = build(d)
    w = wide(long)
    os.makedirs(paths.OUTPUT, exist_ok=True)
    long.to_csv(paths.OUTPUT / "demographics_table.csv", index=False)
    w.to_csv(paths.OUTPUT / "demographics_wide.csv", index=False)
    print(w.to_string(index=False))
    checks(long)
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(paths.OUTPUT / "demographics_table.xlsx") as xl:
            w.to_csv  # keep linters quiet
            w.to_excel(xl, sheet_name="Table 3 (publication)", index=False)
            long.to_excel(xl, sheet_name="Long form", index=False)
        print(f"\nwrote {paths.OUTPUT/'demographics_table.xlsx'}")
    except ImportError:
        print("\nopenpyxl not in this env - CSV written; run WRITE_XLSX to add the workbook")
    print(f"wrote {paths.OUTPUT/'demographics_table.csv'}")
    print(f"wrote {paths.OUTPUT/'demographics_wide.csv'}")


if __name__ == "__main__":
    main()
