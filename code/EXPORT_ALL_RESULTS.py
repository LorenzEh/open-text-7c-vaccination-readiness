"""Export every computed result to CSV in output/.

One row per dimension wherever possible, so the files drop straight into
tables. Regenerates from the stored JSON/NPZ results -- no re-sampling.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT=paths.OUTPUT; OUT.mkdir(exist_ok=True)   # redirected under PAPER2_DATA
C=paths.RESULTS   # results/ in the public release, code/ in the working repo
ORDER=["conf","cmpcy","const","calc","colr","compli","consp"]
NM={"conf":"Confidence","cmpcy":"Complacency","const":"Constraints","calc":"Calculation",
    "colr":"Collective Responsibility","compli":"Compliance","consp":"Conspiracy"}
def load(n):
    p=C/n
    return json.load(open(p)) if p.exists() else None
written=[]
def save(df,name,desc):
    df.to_csv(OUT/name,index=False); written.append((name,len(df),desc))

# 1. descriptives + t-tests
tt=load("results_7c_ttests.json")
if tt:
    save(pd.DataFrame([dict(dimension=tt[k]["name"],abbrev=k,
        n_non_hesitant=tt[k]["n_non"],n_hesitant=tt[k]["n_hes"],
        mean_non_hesitant=round(tt[k]["m_non"],4),sd_non_hesitant=round(tt[k]["sd_non"],4),
        mean_hesitant=round(tt[k]["m_hes"],4),sd_hesitant=round(tt[k]["sd_hes"],4),
        welch_t=round(tt[k]["t"],4),df=round(tt[k]["df"],2),
        p=tt[k]["p"],p_bonferroni=tt[k]["p_bonf"],
        significant_bonferroni="yes" if tt[k]["p_bonf"]<.05 else "no",
        cohens_d=round(tt[k]["d"],4),hedges_g=round(tt[k]["hedges_g"],4)) for k in ORDER]),
        "descriptives_7c_ttests.csv","7C means/SDs by group + Welch t-tests")

# 2. BART full metrics
b=load("results_bart_full_metrics.json")
if b:
    rows=[]
    for k in ORDER:
        r=b[k]; d=dict(dimension=NM[k],abbrev=k,n=r["n"],
            pca_components_per_fold=";".join(map(str,r["dims"])) if r.get("dims") else "",
            global_R2=round(r["r2"],4),global_RMSE=round(r["rmse"],4))
        for i in range(1,6): d[f"R2_LQ{i}"]=round(r[f"R2_LQ{i}"],4) if r[f"R2_LQ{i}"] is not None else ""
        for i in range(1,6): d[f"RMSE_LQ{i}"]=round(r[f"RMSE_LQ{i}"],4) if r[f"RMSE_LQ{i}"] is not None else ""
        for i in range(1,6): d[f"RMSE_Q{i}"]=round(r[f"RMSE_Q{i}"],4) if r[f"RMSE_Q{i}"] is not None else ""
        for i in range(1,6): d[f"N_LQ{i}"]=r[f"N_LQ{i}"]
        for i in range(1,6): d[f"N_Q{i}"]=r[f"N_Q{i}"]
        d.update(ELPD_difference=round(r["elpd_diff"],2),DSE=round(r["dse"],2),
                 ELPD_ratio=round(r["elpd_diff"]/r["dse"],2))
        rows.append(d)
    save(pd.DataFrame(rows),"bart_full_metrics.csv","BART 4-chain: global + quintile metrics + ELPD")

# 3. BART convergence
c=load("results_bart_4chain_partial.json")
if c:
    save(pd.DataFrame([dict(dimension=NM[k],abbrev=k,n=c[k]["n"],
        rhat_sigma=round(c[k]["rhat_sigma"],4),
        rhat_sigma_below_1_01="yes" if c[k]["rhat_sigma"]<=1.01 else "no",
        rhat_mu_max=round(c[k]["rhat_mu_max"],4),
        pct_mu_above_1_01=round(c[k]["rhat_mu_frac_gt101"]*100,1),
        ess_mu_min=int(c[k]["ess_mu_min"]),
        ELPD_difference=round(c[k]["elpd_diff"],2),DSE=round(c[k]["dse"],2)) for k in ORDER]),
        "bart_convergence_diagnostics.csv","BART 4-chain convergence diagnostics")

# 4. Ridge, both specifications
r=load("results_ridge_full_metrics.json")
if r:
    rows=[]
    for spec,lab in (("scaled","scaled (published)"),("unscaled","unscaled (matched to BART)")):
        for k in ORDER:
            d=r[spec][k]; row=dict(specification=lab,dimension=NM[k],abbrev=k,n=d["n"],
                global_R2=round(d["r2"],4),global_RMSE=round(d["rmse"],4),
                RMSE_null_model=round(d["rmse_null"],4),
                improvement_over_mean=round(d["improvement_over_mean"],4))
            for i in range(1,6): row[f"R2_LQ{i}"]=round(d[f"R2_LQ{i}"],4) if d[f"R2_LQ{i}"] is not None else ""
            for i in range(1,6): row[f"RMSE_LQ{i}"]=round(d[f"RMSE_LQ{i}"],4) if d[f"RMSE_LQ{i}"] is not None else ""
            for i in range(1,6): row[f"RMSE_Q{i}"]=round(d[f"RMSE_Q{i}"],4) if d[f"RMSE_Q{i}"] is not None else ""
            row.update(permutation_p=d["perm_p"],permutation_p_bonferroni=d["perm_p_bonferroni"],
                       significant="yes" if d["perm_significant"] else "no")
            rows.append(row)
    save(pd.DataFrame(rows),"ridge_full_metrics.csv","Ridge both specs: metrics + permutation tests")

# 5. sentiment tests
s=load("results_sentiment_group_test.json")
if s:
    rows=[]
    for lvl,key in (("response-level (clustered)","response_level"),
                    ("participant-level (modal, independent)","participant_level_modal")):
        t=s[key]
        for g,lab in ((0,"non-hesitant"),(1,"hesitant")):
            cnt=t["table"][str(g)] if str(g) in t["table"] else t["table"][g]
            tot=sum(cnt.values())
            rows.append(dict(level=lvl,group=lab,n=tot,
                **{f"n_{c}":cnt[c] for c in ("negative","neutral","positive")},
                **{f"pct_{c}":round(100*cnt[c]/tot,2) for c in ("negative","neutral","positive")},
                chi2=round(t["chi2"],4),df=t["dof"],p=t["p"],cramers_v=round(t["cramers_v"],4)))
    save(pd.DataFrame(rows),"sentiment_group_chisquare.csv","Sentiment x group chi-square, both units of analysis")
    save(pd.DataFrame([dict(category=k,chi2=round(v["chi2"],4),p=v["p"],
        p_bonferroni=v["p_bonferroni"],
        significant="yes" if v["p_bonferroni"]<.05 else "no") for k,v in s["followups"].items()]),
        "sentiment_category_followups.csv","Per-category 2x2 follow-ups, Bonferroni x3")
    save(pd.DataFrame([dict(dimension=d,**{k2:(round(v2,4) if isinstance(v2,float) else v2)
        for k2,v2 in v.items()}) for d,v in sorted(s["by_dimension"].items(),
        key=lambda kv:-kv[1]["mean_polarity"])]),
        "sentiment_by_dimension.csv","Sentiment per 7C dimension")

# 6. model comparison summary
if b and r and c:
    save(pd.DataFrame([dict(dimension=NM[k],abbrev=k,
        bart_R2=round(b[k]["r2"],4),
        ridge_scaled_R2=round(r["scaled"][k]["r2"],4),
        ridge_unscaled_R2=round(r["unscaled"][k]["r2"],4),
        best_model=max({"BART":b[k]["r2"],"Ridge (scaled)":r["scaled"][k]["r2"],
                        "Ridge (unscaled)":r["unscaled"][k]["r2"]}.items(),key=lambda kv:kv[1])[0],
        bart_RMSE=round(b[k]["rmse"],4),
        ridge_unscaled_RMSE=round(r["unscaled"][k]["rmse"],4)) for k in ORDER]),
        "model_comparison.csv","BART vs both Ridge specifications")

print(f"{'file':44}{'rows':>6}  description")
for n,ln,d in written: print(f"  {n:42}{ln:>6}  {d}")
print(f"\nwritten to {OUT}")
