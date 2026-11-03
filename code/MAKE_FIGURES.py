"""Regenerate Figures 2 and 3 from stored results -- no model refitting.

Figure 2: observed vs out-of-fold BART predictions, from the four-chain run
          (code/results_bart_oof_predictions.npz).
Figure 3: permutation null distributions for ridge, unstandardised
          specification (code/results_ridge_unscaled_permutation.json, which
          stores all 1,000 null R2 values per dimension).

Both previously disagreed with Table 1: Figure 2 carried the two-chain values
and Figure 3 the standardised ridge run, including two negative R2 that the
paper no longer reports.

Two co-author requests are implemented here:
  - Figure 2: the dashed "perfect prediction" diagonal is removed.
  - Figure 3: the in-panel legend showing the exact R2 is removed; the
    vertical line marking the observed value stays.
"""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

ORDER=["conf","cmpcy","const","calc","colr","compli","consp"]
NM={"conf":"Confidence","cmpcy":"Complacency","const":"Constraints","calc":"Calculation",
    "colr":"Coll. Responsibility","compli":"Compliance","consp":"Conspiracy"}
OUT=paths.OUTPUT; OUT.mkdir(exist_ok=True)   # redirected under PAPER2_DATA
DPI=300   # journals generally require >=300 dpi for production figures

def grid(n=7, ncols=3):
    nrows=(n+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(6*ncols,5*nrows),constrained_layout=True)
    return fig, fig.axes

def panel_label(ax, i):
    """Bold lower-case panel letter at the bottom left, as journals require.

    Just outside the axes at the lower left, in axes-fraction coordinates, so
    the letter never sits on top of a point or a bar and needs no patch behind
    it. Checked on the full 3x3 grid rather than a single row: at this offset
    the letters clear both the x-axis label above them and the next row's
    titles below.
    """
    # "a)" not "a": npj Digital Public Health specifies multi-panel figures be
    # "labeled using an a), b), c) convention" (nature.com/npjdigipubhealth/content-types).
    ax.text(-0.14, -0.17, f"{chr(ord('a')+i)})", transform=ax.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="left")

# ---------------- Figure 2 ----------------
def figure2():
    d=np.load(paths.result("results_bart_oof_predictions.npz"))
    fig,axes=grid()
    for i,k in enumerate(ORDER):
        ax=axes[i]; y,p=d[f"{k}_true"], d[f"{k}_pred"]
        r2=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
        rmse=np.sqrt(np.mean((y-p)**2))
        ax.scatter(y,p,alpha=0.6,edgecolor="white",s=50)
        # dashed "perfect prediction" diagonal removed at co-author request
        ax.set_xlabel("Observed score"); ax.set_ylabel("Predicted score")
        ax.set_title(f"{NM[k]}\nR² = {r2:.3f}, RMSE = {rmse:.3f}",fontsize=14)
        # Both axes on the response scale. Without the diagonal, auto-scaling would
        # stretch the prediction range to fill each panel and hide the compression
        # toward the mean that the Results paragraph describes.
        ax.set_xlim(0.8,7.2); ax.set_ylim(0.8,7.2)
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax,i)
    for ax in axes[len(ORDER):]: ax.axis("off")
    fig.suptitle("Observed and cross-validated BART-predicted scores",fontsize=20)
    fig.savefig(OUT/"figure2_bart_observed_vs_predicted.png",dpi=DPI)
    plt.close(fig); print("wrote figure2_bart_observed_vs_predicted.png")

# ---------------- Figure 3 ----------------
def figure3():
    r=json.load(open(paths.result("results_ridge_unscaled_permutation.json")))
    sns.set_theme(style="white")
    fig,axes=grid()
    for i,k in enumerate(ORDER):
        ax=axes[i]; v=r[k]
        sns.histplot(v["null_distribution"],bins=30,kde=True,color="skyblue",
                     edgecolor="white",linewidth=0.5,ax=ax)
        ax.axvline(v["r2"],color="crimson",linestyle="--",linewidth=3)
        # in-panel legend with the exact R2 removed at co-author request
        ax.set_title(f"{NM[k]}\np = {v['p']:.3f}",fontsize=14)
        ax.set_xlabel("Out-of-fold R²"); ax.set_ylabel("Frequency")
        panel_label(ax,i)
    for ax in axes[len(ORDER):]: ax.axis("off")
    fig.suptitle("Permutation tests: null distributions and observed fit",fontsize=20)
    fig.savefig(OUT/"figure3_ridge_permutation.png",dpi=DPI)
    plt.close(fig); print("wrote figure3_ridge_permutation.png")


# ---------------- Appendix Table 12: null model vs BART ----------------
def appendix_null_vs_bart():
    """RMSE of an intercept-only null against BART, per dimension.

    The null predicts the outcome mean for everyone, so its RMSE is the
    population SD -- the same denominator R2 uses, which keeps the bar chart
    consistent with the R2 reported elsewhere.
    """
    d=np.load(paths.result("results_bart_oof_predictions.npz"))
    names=[NM[k] for k in ORDER]
    bart=[float(np.sqrt(np.mean((d[f"{k}_true"]-d[f"{k}_pred"])**2))) for k in ORDER]
    null=[float(np.sqrt(np.mean((d[f"{k}_true"]-d[f"{k}_true"].mean())**2))) for k in ORDER]
    xs=np.arange(len(ORDER)); w=0.35
    fig,ax=plt.subplots(figsize=(10,6))
    ax.bar(xs-w/2,null,w,label="Null model RMSE",color="skyblue",edgecolor="white")
    ax.bar(xs+w/2,bart,w,label="BART RMSE",color="steelblue",edgecolor="white")
    ax.set_xlabel("Dimension"); ax.set_ylabel("RMSE")
    ax.set_title("RMSE: null model vs BART")
    ax.set_xticks(xs); ax.set_xticklabels(names,rotation=45,ha="right")
    ax.yaxis.grid(True,linestyle="--",alpha=0.7); ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"appendix_table12_null_vs_bart.png",dpi=DPI)
    plt.close(fig)
    print("wrote appendix_table12_null_vs_bart.png")
    for n,b_,nl in zip(names,bart,null):
        print(f"   {n:24} null {nl:.3f}   BART {b_:.3f}   improvement {nl-b_:+.3f}")

# ---------------- Appendix Table 6: distribution of the 7C ----------------
def appendix_7c_distributions():
    """Histogram of each 7C dimension score.

    Reproduces plot_dimension_histograms() from EXPLORATORY1.py, which only
    called plt.show() and so could not be regenerated to file. Same bins,
    colour and styling; adds the panel letters and saves at 300 dpi.

    Bin edges sit midway between the attainable values: each score is the mean
    of three 7-point items, so only multiples of 1/3 can occur. Binning on
    anything else merges adjacent spikes and invents structure.
    """
    import pandas as pd
    # The only figure here that needs participant-level scores rather than stored
    # model output. It needs just the seven scale scores, which ARE in the public
    # deposit (output/public_7c_scores.csv, published uncoarsened by
    # MAKE_PUBLIC_DATA.py), so fall back to those. That keeps all five figures
    # regenerable from the public release alone -- no embeddings, no data/.
    pub = paths.OUTPUT / "public_7c_scores.csv"
    if paths.EMBEDDINGS_WITH_QUESTIONNAIRE.exists():
        df=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE, usecols=ORDER)
    elif pub.exists():
        print(f"  reading 7C scores from {pub.name} (the public deposit's copy)")
        df=pd.read_csv(pub, usecols=ORDER)
    else:
        print("skipped appendix_table6_7c_distributions.png "
              "(needs either data/ or output/public_7c_scores.csv)")
        return
    sns.set_theme(style="white", context="notebook")
    vals=np.arange(1, 7+1/3, 1/3)
    edges=np.concatenate([[vals[0]-1/6],(vals[:-1]+vals[1:])/2,[vals[-1]+1/6]])
    fig,axes=grid()
    FULL=dict(NM, colr="Collective Responsibility")
    for i,k in enumerate(ORDER):
        ax=axes[i]
        sns.histplot(df[k].dropna(),bins=edges,kde=False,color="skyblue",
                     edgecolor="white",linewidth=0.5,alpha=1,ax=ax)
        ax.set_xlim(0.8,7.2); ax.set_xticks(range(1,8)); ax.grid(False)
        ax.set_title(FULL[k],fontsize=14,color="dimgrey",weight="bold")
        ax.set_xlabel("Score (1–7)",fontsize=11); ax.set_ylabel("Frequency",fontsize=11)
        panel_label(ax,i)
    for ax in axes[len(ORDER):]: ax.axis("off")
    fig.suptitle("Distribution of 7C Dimensions",fontsize=20)
    fig.savefig(OUT/"appendix_table6_7c_distributions.png",dpi=DPI)
    plt.close(fig); print("wrote appendix_table6_7c_distributions.png")

# ------- Appendix: ridge observed vs predicted, the counterpart to Figure 2 -------
def appendix_ridge_observed_vs_predicted():
    """Same plot as Figure 2 but for ridge, so the two can be read side by side.

    Uses the `unscaled_*` arrays -- the matched specification reported in Table 1
    and appendix Table 15. The file also carries `scaled_*`, the superseded
    standardised run with the two negative R2; those are not plotted.

    Axes, limits and panel letters are identical to figure2() on purpose: the
    point of the figure is the comparison, and any difference in framing would
    show up as an apparent difference in the models.
    """
    d=np.load(paths.result("results_ridge_oof_predictions.npz"))
    sns.set_theme(style="white")
    fig,axes=grid()
    spread=[]
    for i,k in enumerate(ORDER):
        ax=axes[i]; y,p=d[f"unscaled_{k}_true"], d[f"unscaled_{k}_pred"]
        r2=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
        rmse=np.sqrt(np.mean((y-p)**2))
        spread.append((NM[k], p.max()-p.min(), y.max()-y.min()))
        ax.scatter(y,p,alpha=0.6,edgecolor="white",s=50)
        ax.set_xlabel("Observed score"); ax.set_ylabel("Predicted score")
        ax.set_title(f"{NM[k]}\nR² = {r2:.3f}, RMSE = {rmse:.3f}",fontsize=14)
        ax.set_xlim(0.8,7.2); ax.set_ylim(0.8,7.2)
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax,i)
    for ax in axes[len(ORDER):]: ax.axis("off")
    fig.suptitle("Observed and cross-validated ridge-predicted scores",fontsize=20)
    fig.savefig(OUT/"appendix_ridge_observed_vs_predicted.png",dpi=DPI)
    plt.close(fig); print("wrote appendix_ridge_observed_vs_predicted.png")
    return spread

if __name__=="__main__":
    figure2(); figure3(); appendix_null_vs_bart(); appendix_7c_distributions()
    sp=appendix_ridge_observed_vs_predicted()
    # Both models compress predictions toward the mean; report by how much, so the
    # Results sentence about attenuation is not asserted from eyeballing a scatter.
    b=np.load(paths.result("results_bart_oof_predictions.npz"))
    print(f"\n{'Dimension':24}{'observed range':>16}{'BART range':>12}{'ridge range':>13}")
    for (nm,rr,orng),k in zip(sp,ORDER):
        bp=b[f"{k}_pred"]
        print(f"{nm:24}{orng:>16.2f}{bp.max()-bp.min():>12.2f}{rr:>13.2f}")
    # report what the panels now say, so it can be checked against Table 1
    d=np.load(paths.result("results_bart_oof_predictions.npz"))
    r=json.load(open(paths.result("results_ridge_unscaled_permutation.json")))
    print(f"\n{'Dimension':22}{'Fig2 R²':>9}{'Fig2 RMSE':>11}{'Fig3 R²':>9}{'Fig3 p':>9}")
    for k in ORDER:
        y,p=d[f"{k}_true"],d[f"{k}_pred"]
        r2=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2); rmse=np.sqrt(np.mean((y-p)**2))
        print(f"{NM[k]:22}{r2:>9.3f}{rmse:>11.3f}{r[k]['r2']:>9.3f}{r[k]['p']:>9.3f}")
