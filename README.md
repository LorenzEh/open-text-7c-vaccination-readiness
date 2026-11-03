# Open text captures psychological antecedents of vaccination readiness — code and results

Analysis code for the paper, together with every stored model output needed to
re-derive the reported tables and figures.

<!--AUTHORS-->
Helena Pires\*, Lorenz Eckhard\*, Ursula Wiedermann-Schmidt, Robert Böhm
(\*equal contribution). Medical University of Vienna; University of Vienna;
University of Innsbruck.
<!--/AUTHORS-->

Preregistration: <https://osf.io/k3hgn/>

---

## What this study did

777 German-speaking adults each answered **seven open questions**, one written to
match each dimension of the **7C Vaccination Readiness Scale**, and also completed
the **21-item 7C scale**. We embedded each open response and tried to predict the
matching dimension's scale score, to test whether free text carries signal about a
validated psychological construct.

The finding is deliberately hedged: correspondence is **partial and
construct-dependent**. Text predicts Compliance, Confidence and Collective
Responsibility reasonably (R² ≈ .34–.39) and Constraints and Calculation barely
(R² ≈ .04–.06). Open text **complements** psychometric assessment; it does not
replace it.

---

## What is not in this repository, and why

**The participant data are not here.** The raw exports and every derived file
containing response text hold Prolific IDs and German free-text answers about
health behaviour and institutional trust — personal data under the GDPR.

### What participants were told

The privacy notice shown to participants (Universität Wien, *Informationen zum
Datenschutz*) explicitly covers public deposit:

> Die **anonymisierten** Daten werden auf dem Open Science Framework (einem
> öffentlich zugänglichen Archiv) zur zukünftigen Verwendung in Forschung und
> Lehre hinterlegt.

So an OSF deposit is consented. But the permission is conditional on
anonymisation, and the same notice makes two specific factual promises:

> Rückschlüsse auf Ihre oder die Identität anderer Personen sind **nicht möglich**.

> … Ihr persönlicher Datensatz nach der Anonymisierung für uns nicht mehr
> identifizierbar ist und einzelne Datensätze daher **nicht mehr gelöscht werden
> können**.

That is a higher bar than "we removed the names". It is a claim that re-identification
is impossible, and it is the reason participants were told their individual records
can no longer be withdrawn.

### Can we just publish the embeddings, or the principal components?

**No — and the consent notice is the reason, not merely the absence of one.**
Sentence embeddings are not anonymised text:

- Morris et al. (2023), *Text Embeddings Reveal (Almost) As Much As Text*
  (EMNLP), recover **92% of 32-token inputs exactly** from embedding vectors,
  and on clinical notes recover **full names at 89%**. The attack needs query
  access to the same embedding model — but ours is a **public Hugging Face
  checkpoint**, so that access is free to anyone.
- Song & Raghunathan (2020), *Information Leakage in Embedding Models* (CCS),
  recover **50–70% of input words**, and separately show **authorship can be
  inferred from 10–50 sentences per author**. This design has seven responses per
  participant, which is inside that range.
- Recovery does degrade with length — exact-match falls to ~8% at ~81 tokens —
  and these responses are longer than 32 tokens, so the headline figure does not
  transfer directly. But "harder than the benchmark" is not impossibility, and
  impossibility is what participants were promised.

**Principal components do not rescue it.** Neither paper tests dimensionality
reduction, so there is no published basis for claiming PCA prevents inversion.
Retaining 95% of the variance keeps most of the geometry an inversion model uses.
Publishing PCs would mean asserting *Rückschlüsse sind nicht möglich* on the
strength of a mitigation nobody has measured — the one claim the notice does not
permit us to guess at.

The same logic cuts the other way for the second promise: telling participants
their records can no longer be deleted is defensible only if the data really are
anonymous. Publishing invertible data after saying that compounds the problem
rather than sharing it.

Both papers conclude that **embeddings should be handled with the same care as the
raw text**. That is the position taken here.

### What we do provide

| Tier | What you get | Published? |
|---|---|---|
| **1. Stored results** | Every reported number, re-derivable from `results/` | **Yes, here** |
| **2. 7C scale scores** | `public_7c_scores.csv` — 777 × 7 dimension means + group | **Yes** |
| **3. Demographics** | Aggregate only, in Supplementary Table 3 | **Aggregate only** |
| **4. Embeddings** | On request, under a data use agreement | No |
| **5. Raw response text** | Not released | No |

Tier 3 is aggregate-only for a measured reason rather than a cautious one.
`code/MAKE_PUBLIC_DATA.py` tests how identifying the row-level demographics are:
as collected, **505 of 777 participants (65%) are unique** on the combination of
age, gender, education, employment, place, country, politics and religion. After
banding age into five groups, merging Austria with Switzerland, and dropping both
politics and religion, **38 people are still unique** — k never reaches 2, let
alone the k ≥ 5 the script targets. Row-level demographics are therefore not
published, and the script refuses to write the file rather than shipping it.

The 7C scores are published without coarsening, because the risk is different in
kind. Re-identification through quasi-identifiers needs a second source holding
the same attributes; no such source holds these people's 7C scores. That is why
demographics are the dangerous half of a dataset like this and scale scores are
the routine half.

Tier 1 is the default and covers most verification: out-of-fold predictions for
both models, all 1,000 permutation null distributions per dimension, convergence
diagnostics, per-response sentiment labels, and every summary statistic. You can
recompute the R², RMSE, confidence intervals, permutation p-values and almost
every figure without touching a single participant response.

Be aware what tier 1 is: some of these files are **participant-level derived
values** — one predicted score per participant per dimension, one sentiment label
per response. They contain no response text, no Prolific ID and no demographics,
and they cannot be joined to anything we withhold. We list this rather than bundle
it quietly, because "derived" is not a synonym for "non-personal" and the reader
should be able to judge for themselves.

**Venue note.** The privacy notice names OSF, but COS is phasing out the OSF
Projects workflow: no new projects after 16 November 2026, and existing projects
become read-only on 19 February 2027. Registrations are unaffected, so the
preregistration keeps resolving. The data deposit therefore goes to **Zenodo** —
COS's own first-named alternative, which issues a DOI and supports the
restricted-access tier below. The notice's operative wording is "einem öffentlich
zugänglichen Archiv", which Zenodo satisfies.

Tier 2 is the deposit the privacy notice promised: scale scores, coarsened
demographics and the study materials. Tier 3 requests go to the corresponding
author under a data use agreement forbidding inversion attempts and
redistribution.

---

## Setup

Two environments, deliberately separate — the topic-modelling stack pins a
different NumPy and Python from the modelling stack, and mixing them silently
changes results.

```bash
conda env create -f env/environment-bart.yml   # BART, ridge, all reported models
conda env create -f env/environment-text.yml   # BERTopic and sentiment only
```

`env/requirements-*.txt` are pip fallbacks listing the exact versions named in the
manuscript Methods.

> **macOS note.** `pymc_bart` starts a multiprocessing `Manager`, and macOS spawns
> rather than forks, so a script with code at module level re-executes itself in the
> child. `BART_TRIALS.py` is affected: run it in IPython/Spyder, or wrap its body in
> `if __name__ == "__main__":`. Every script written for this release is already
> guarded.

---

## Reproducing the paper without the data

One command. It detects that there is no `data/`, runs the stages that rebuild
reported numbers from `results/`, and verifies the outcome against the published
values:

```bash
conda activate BART
python code/RUN_ALL.py
```

Under a minute. It rebuilds **every figure and every reported table**, then
prints a pass/fail line per published quantity. Nothing in it refits a model —
that is the point: the out-of-fold predictions are stored, so the numbers can be
recomputed from them without the features they were computed from.

The individual stages, if you want them one at a time:

```bash
python code/MAKE_FIGURES.py            # Figures 2, 3 and appendix Tables 6, 12, 21
python code/BOOTSTRAP_CIS.py           # the 95% CIs in Table 1
python code/EXPORT_ALL_RESULTS.py      # every reported table as CSV
python code/SENTIMENT_BY_DIM_GROUP.py  # sentiment by dimension and group
python code/VERIFY_INSTALL.py          # the verification on its own, ~10 seconds
```

Set `PAPER2_RESULTS=…/results` if you move that directory.

This was tested by copying the release to an empty directory with no `data/` and
running it there. All five figures come out **byte-identical** to the ones in the
paper, and `BOOTSTRAP_CIS.py` reproduces `results_bootstrap_cis.json` exactly.

`appendix_table6_7c_distributions.png` is the one figure that plots
participant-level values rather than model output — a histogram of the 7C scale
scores. Those scores are themselves in the deposit
(`output/public_7c_scores.csv`, published uncoarsened), so it reads them from
there and regenerates identically. No figure depends on the embeddings.

| Reported object | Script | Reads |
|---|---|---|
| Table 1, BART rows | `BART_FULL_METRICS.py` | data |
| Table 1, ridge rows | `RIDGE_UNSCALED_CHECK.py` | data |
| Table 1, 95% CIs | `BOOTSTRAP_CIS.py` | stored predictions |
| Figure 2, appendix Table 21 | `MAKE_FIGURES.py` | stored predictions |
| Appendix Table 6 | `MAKE_FIGURES.py` | published 7C scores |
| Figure 3 | `MAKE_FIGURES.py` | stored null distributions |
| Appendix Tables 10, 11, 13 | `BART_4CHAIN_CHECK.py` | data |
| Appendix Tables 14, 15 | `RIDGE_FULL_METRICS.py` | data |
| Appendix Table 16, 20 | `SENTIMENT_FINAL.py`, `SENTIMENT_GROUP_TEST.py` | data |
| Appendix Table 17 | `DISCRIMINANT_VALIDITY.py` | data |
| Appendix Table 18 | `BART_NO_PCA_CHECK.py` | data |
| Appendix Table 19 | `CRONBACH_ALPHA.py` | data |
| Appendix Tables 8, 9 | `COMPARISON_PC_EMBEDDINGS.py` | data |
| Figure 1, appendix topic log | `TOPIC_FINAL.py` | data (text env) |

Full re-run from raw data costs roughly **45 minutes for BART** (seven dimensions
× five folds) and a few minutes for everything else.

---

## Can you replicate the models, not just the numbers?

Two different things, and the release supports them differently.

**Verifying the reported numbers needs nothing from us.** `results/` ships the
out-of-fold predictions, all 1,000 permutation nulls per dimension and every
summary statistic, so R², RMSE, confidence intervals, permutation p-values and
four of five figures recompute exactly. Start here:

```bash
python code/VERIFY_INSTALL.py
```

It checks the environment, confirms the stored results reproduce the published
Table 1 to three decimals, and then tries to fit a model on whatever data is
present. Seconds, not minutes.

**Reproducing the whole paper** — every table and figure, refitted from the
data rather than read from storage — needs the embeddings. If you have them:

```bash
python code/RUN_ALL.py          # 18 stages, ~2 hours, mostly the BART fits
python code/RUN_ALL.py --fast   # skip the slow refits, ~6 minutes
python code/RUN_ALL.py --list   # what runs, in what order, how long
```

It runs the stages in dependency order, reports each one, carries on past a
failure rather than halting, and finishes by checking the result against the
published numbers. The embeddings are available to researchers on request —
terms and what to send are in `DATA_USE_AGREEMENT.md`.

**Without them**, the embeddings are not published.
So that the code is not simply unrunnable for you, there is a synthetic dataset:

```bash
python code/MAKE_SYNTHETIC_DATA.py
PAPER2_DATA=data_synthetic python code/VERIFY_INSTALL.py
```

`PAPER2_DATA` repoints the data directory, so **every script runs unmodified**.

The synthetic set has the real schema — 777 rows, the same column names, dtypes
and value ranges — and contains no real responses: the text columns are one
fixed placeholder sentence and every row carries `synthetic = 1`. It is not a
perturbed or resampled copy of the real data; no real value is used to produce
any number in it.

It does carry **recoverable signal**, calibrated per dimension so each is about
as learnable as its real counterpart:

| | conf | cmpcy | const | calc | colr | compli | consp |
|---|---|---|---|---|---|---|---|
| target | .40 | .12 | .05 | .07 | .42 | .39 | .20 |
| achieved | .391 | .122 | .054 | .089 | .420 | .395 | .203 |

That calibration matters. An earlier version planted the signal in a single
direction, PCA lost it among 1,024 noise dimensions, and every R² came out at
zero — at which point a working pipeline and a broken one are indistinguishable.
With signal present you can tell that the pipeline recovers what is there, in
roughly the right order across dimensions.

**Numbers from the synthetic set will not match the paper and are not meant to.**
To reproduce the published values you need the real embeddings, which go out on
request under a data use agreement. With those in place, the same commands
produce the published numbers, because `PAPER2_DATA` is the only thing that
changes.

## Running the text analyses

These are the stages that touch the responses themselves rather than stored model
output, so they need tier 3 access. **They use the second environment** — the
topic stack pins a different NumPy and Python, and running it in the modelling env
silently changes results.

### The chain

```
complete_dataset/*.csv                 raw exports
   │  survey1_data_cleaning.py  ── data_prep()            [BART env]
   ▼
data/final_data_merged.csv             cleaned + scored
   │  ⚠ manual annotation of unusable responses — see below
   ▼
data/final_data_merged_cleaned.csv     the exclusion-mask source
   │  generate_embeddings_for_bart_1.py   response only   [text env]
   │  generate_embedding_for_bart_2.py    question + response  ★ canonical
   ▼
data/final_data_embeddings*.csv        1,024-dim vectors
   │
   ├── topic modelling and sentiment    [text env]
   └── BART / ridge / everything else   [BART env]
```

### Commands

```bash
conda activate sentence_transformers

python code/TOPIC_FINAL.py             # BERTopic per group → Figure 1  (~5 min)
python code/TOPIC_COUNTS_HESITANT.py   # hesitant topic sizes          (~1 min)
python code/SENTIMENT_GROUP_TEST.py    # classify every response       (~5 min)
```

`SENTIMENT_GROUP_TEST.py` writes `results_sentiment_labels.csv`, which everything
downstream consumes. Once it exists the rest runs in the modelling env and needs no
GPU:

```bash
conda activate BART

python code/SENTIMENT_FINAL.py           # Table 16: permutation tests, out-of-fold R²
python code/SENTIMENT_REGRESSION_PERDIM.py
python code/SENTIMENT_BY_DIM_GROUP.py    # sentiment by dimension × group
```

`TOPIC_FINAL.py` saves Figure 1 instead of opening a window. Set
`PAPER2_INTERACTIVE=1` to get the interactive behaviour back.

### Two things that will bite you

**The manual annotation step is not scripted.** `final_data_merged_cleaned.csv`
differs from `final_data_merged.csv` by hand-coded markers appended to unusable
responses, and that work exists in no script. Regenerating the merged file from raw
would silently discard it. Start from `_cleaned` if you have it; do not rebuild it.

The markers are matched at the **end** of the response string:
`-1` (empty), `(2)` no opinion, `(3)` misunderstood, `(4)`–`(7)` off-topic — all
excluded. **`(8)` is retained on purpose** — short but meaningful answers such as
*"Gar nicht."* Any rewrite of the masking logic must keep it in.

**The embedding scripts were written on Colab.** They carry a shim that mounts Drive
there and falls back to local paths otherwise, so they run in both places, but the
embeddings behind the reported results came from the Colab runs. Re-running locally
reproduces them only if the model checkpoint and `sentence-transformers` version
match.

## Conventions that matter if you modify anything

- `RANDOM_SEED = 42` everywhere; 5 outer folds, `KFold(shuffle=True, random_state=42)`.
- PCA retains **95% of variance, fitted inside the training fold only**. This is a
  stated no-leakage guarantee — do not hoist it out of the cross-validation loop.
- Reported metrics are **pooled out-of-fold predictions**, not fold means.
- Ridge is reported on **unstandardised** inputs, matching BART's preprocessing, so
  the two models differ only in the estimator. The ridge penalty grid runs to 1e8;
  an earlier grid capped at 100 was selected at its upper bound on all seven
  dimensions, which under-regularised the model and produced two negative R².
- **Ridge is exploratory.** The preregistration specifies BART; ridge is a benchmark.

### Known limitations, stated plainly

- **Convergence.** R-hat for σ exceeds 1.01 on three of seven dimensions
  (max 1.020), and pointwise μ exceeds 1.01 for 69–88% of observations. The
  sampler is particle-Gibbs, not NUTS. Point estimates are stable — doubling the
  chains moves R² by ≤ 0.007 — but the diagnostics do not support a clean
  convergence claim, and the appendix reports the full distribution rather than a
  summary.
- **Nested cross-validation** was preregistered; BART in fact runs single-loop
  5-fold, because the variance-threshold branch bypasses the inner loop. Ridge is
  nested via `RidgeCV`.
- **Sample composition** deviates from the preregistration, which specified 300 per
  group. The achieved sample is 166 hesitant / 611 non-hesitant.
- **The `(8)` exclusion code is retained on purpose** — short but meaningful answers
  such as "Gar nicht." Any rewrite of the masking logic must keep it.

---

## Citation

Preregistration: <https://osf.io/k3hgn/>. Please cite the paper; see `CITATION.cff`
once the DOI is assigned.

## Licence

Code: MIT. Model outputs in `results/` and `output/`: CC BY 4.0.
The 7C Vaccination Readiness Scale is the work of its original authors and is
reproduced in the appendix under fair academic use.
