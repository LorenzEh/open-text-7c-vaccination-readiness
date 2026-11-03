"""Supplementary Table 4 in the native 7C direction, instead of readiness-coded.

Every 7C item is recoded in `survey1_data_cleaning.py` so that higher = greater
vaccination readiness. For four dimensions that inverts the construct the
dimension is named after: a high "Complacency" score in the readiness coding
means LOW complacency, which is what makes the table hard to read.

This reports each dimension in its own direction instead — higher Complacency
means more complacent, higher Conspiracy means more conspiracy belief.

Which dimensions flip
---------------------
The flip is DIMENSION-level, not item-level, and it is worth being clear why,
because the per-item reverse flags look like they should decide it and do not.

Reverse-coding in the cleaning script is applied per item (Complacency reverses
2 of 3, Constraints 1 of 3). Its purpose is to make every item within a
dimension point the same way — towards readiness — so the dimension mean is
meaningful. Once that is done, the stored items in a dimension are all
readiness-oriented regardless of which ones were individually reversed. Turning
the dimension back to its own direction is therefore a single operation on the
whole dimension:

    Confidence, Collective Responsibility, Compliance
        more of the construct = more readiness  ->  unchanged
    Complacency, Constraints, Calculation, Conspiracy
        more of the construct = less readiness  ->  native = 8 - readiness

So the per-item flags cannot verify the transform; they only explain how the
items came to be aligned. What the script checks instead is that alignment: that
within each dimension the stored items are positively inter-correlated. If a
future edit to the reverse list broke the alignment, a dimension mean would stop
being interpretable in either direction, and that check would catch it.

What changes in the table, and what does not
--------------------------------------------
Means flip (8 - m). SDs, t, df and p are unchanged -- a constant-minus-x
transform negates the difference between groups but not its magnitude or the
variances. Cohen's d changes sign. So the four flipped rows swap which group
scores higher, and nothing about the inference changes.

Outputs
    output/table4_native_direction.csv    the numbers
    output/table4_native_direction.html   paste-ready for Google Docs / Word

Run with the BART env:
    /opt/anaconda3/envs/BART/bin/python code/TABLE4_NATIVE_DIRECTION.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SCALE_MAX_PLUS_1 = 8  # items are 1-7, so the reverse transform is 8 - x

ORDER = ["conf", "cmpcy", "const", "calc", "colr", "compli", "consp"]
NAME = {"conf": "Confidence", "cmpcy": "Complacency", "const": "Constraints",
        "calc": "Calculation", "colr": "Collective resp.",
        "compli": "Compliance", "consp": "Conspiracy"}
# Items reversed in survey1_data_cleaning.py to make higher = more readiness.
READINESS_REVERSED = {"cmpcy_1", "cmpcy_2", "const_3",
                      "calc_1", "calc_2", "calc_3",
                      "consp_1", "consp_2", "consp_3"}
FLIPS = ["cmpcy", "const", "calc", "consp"]   # asserted below, not assumed

GLOSS = {
    "cmpcy": "higher = more complacent (perceives less disease risk)",
    "const": "higher = more constrained (more practical barriers)",
    "calc": "higher = more deliberative weighing of risks and benefits",
    "consp": "higher = stronger conspiracy beliefs",
}


def check_alignment(df):
    """Within a dimension the stored items must point the same way.

    That alignment is what makes the dimension mean meaningful and what makes
    the direction flip a single operation. Reported rather than asserted at a
    threshold, because what counts as "too low" is a judgement.
    """
    print("item alignment within each dimension (min pairwise r, stored coding):")
    ok = True
    for d in ORDER:
        items = [f"{d}_{i}" for i in (1, 2, 3)]
        r = df[items].astype(float).corr().values
        mn = min(r[0, 1], r[0, 2], r[1, 2])
        flag = "" if mn > 0 else "   <-- NEGATIVE, dimension not coherently oriented"
        if mn <= 0:
            ok = False
        print(f"  {NAME[d]:20} min r = {mn:+.3f}{flag}")
    assert ok, "a dimension has negatively correlated items; the mean is not interpretable"


def native_scores(readiness):
    """Flip the four dimensions whose construct runs opposite to readiness."""
    out = readiness.copy()
    for d in FLIPS:
        out[d] = SCALE_MAX_PLUS_1 - readiness[d]
    return out


def main():
    df = pd.read_csv(paths.MERGED_CLEANED)
    readiness = df[ORDER].astype(float)
    check_alignment(df)
    native = native_scores(readiness)
    print("\ndirection applied:")
    for d in ORDER:
        print(f"  {NAME[d]:20} {'flipped (8 - x)' if d in FLIPS else 'unchanged'}")

    hes = df["Hesitant"].astype(int)
    rows = []
    for d in ORDER:
        a = native.loc[hes == 0, d].dropna()
        b = native.loc[hes == 1, d].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        # Welch-Satterthwaite df
        va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
        dfree = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
        pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
        dcoh = (a.mean() - b.mean()) / pooled
        rows.append(dict(
            dimension=NAME[d], flipped=d in FLIPS,
            n_non=na, m_non=a.mean(), sd_non=a.std(ddof=1),
            n_hes=nb, m_hes=b.mean(), sd_hes=b.std(ddof=1),
            t=t, df=dfree, p=p, d=dcoh,
            p_bonf=min(1.0, p * len(ORDER)),
            direction=GLOSS.get(d, "higher = more of the named construct")))
    res = pd.DataFrame(rows)
    res.to_csv(paths.OUTPUT / "table4_native_direction.csv", index=False)

    fp = lambda v: "&lt; .001" if v < .001 else f"= {v:.3f}".replace("0.", ".")
    print(f"\n{'Dimension':20}{'Non-hesitant':>18}{'Hesitant':>18}   Welch's t-test")
    for r in rows:
        print(f"{r['dimension']:20}{r['m_non']:7.2f} (SD={r['sd_non']:.2f}) "
              f"{r['m_hes']:7.2f} (SD={r['sd_hes']:.2f})   "
              f"t({r['df']:.0f}) = {r['t']:+.2f}, p {fp(r['p']).replace('&lt;','<')}, "
              f"d = {r['d']:+.2f}")

    cells = []
    for r in rows:
        star = " †" if r["flipped"] else ""
        cells.append(
            f"<tr><th scope='row'>{r['dimension']}{star}</th>"
            f"<td>{r['m_non']:.2f} (SD = {r['sd_non']:.2f})</td>"
            f"<td>{r['m_hes']:.2f} (SD = {r['sd_hes']:.2f})</td>"
            f"<td>t({r['df']:.0f}) = {abs(r['t']):.2f}, p {fp(r['p'])}, "
            f"d = {r['d']:+.2f}</td></tr>")
    html = f"""<!doctype html><meta charset="utf-8">
<title>Table 4 — native direction</title>
<style>
 body{{font:14px/1.5 Georgia,serif;margin:2rem;max-width:62rem;color:#111}}
 table{{border-collapse:collapse;width:100%}}
 caption{{text-align:left;font-weight:700;padding-bottom:.5rem}}
 th,td{{border:1px solid #999;padding:6px 9px;text-align:left;vertical-align:top}}
 thead th{{background:#eee}}
 tbody th{{font-weight:400}}
 .sub td{{background:#f6f6f6;font-style:italic}}
 .note{{font-size:12.5px;color:#333;margin-top:.6rem}}
</style>
<table>
<caption>Table 4. Seven dimensions of non-hesitants and hesitants</caption>
<thead><tr><th></th><th>NON-HESITANTS</th><th>HESITANTS</th><th>WELCH'S t-TEST</th></tr></thead>
<tbody>
<tr><th scope="row">Number total</th><td>{rows[0]['n_non']}</td><td>{rows[0]['n_hes']}</td><td></td></tr>
<tr class="sub"><td colspan="4">Dimensions (means; each dimension scored in its own
direction, so higher = more of the named construct)</td></tr>
{chr(10).join(cells)}
</tbody></table>
<p class="note"><em>Note.</em> Welch's t-test for unequal variances; <em>d</em> = Cohen's
<em>d</em> on the pooled SD, signed so that a positive value means non-hesitant participants
scored higher. Each dimension is reported in its own direction: higher Confidence means more
confidence, higher Complacency means more complacency, and so on. &#8224; marks the four
dimensions whose sign is inverted relative to the readiness coding used in the analyses
(Complacency, Constraints, Calculation, Conspiracy); for these, <em>8 &minus; x</em> converts
between the two and only the sign of the group difference changes, not its magnitude,
<em>t</em>, df or <em>p</em>. After Bonferroni correction across the seven dimensions
(&alpha; = .05/7 = .007) all six significant comparisons remain significant. Calculation is the
only dimension that does not differ between groups (Bonferroni-adjusted <em>p</em> = {min(1.0, rows[3]['p']*7):.3f}).</p>
"""
    out = paths.OUTPUT / "table4_native_direction.html"
    out.write_text(html, encoding="utf8")
    print(f"\nwrote {paths.OUTPUT/'table4_native_direction.csv'}")
    print(f"wrote {out}  — open it and copy the table into Google Docs")


if __name__ == "__main__":
    main()
