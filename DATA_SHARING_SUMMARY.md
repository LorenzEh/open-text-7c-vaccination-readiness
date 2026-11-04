# Why the embeddings stay closed, and why replication is unaffected

One page, written to be forwarded. The long version is in the repository README;
the request procedure is in [DATA_USE_AGREEMENT.md](DATA_USE_AGREEMENT.md).

---

## The short version

We cannot publish the sentence embeddings of participants' open-text responses,
because an embedding of a short text is not an anonymous object: the published
inversion literature recovers substantial portions of the original wording from
exactly this kind of vector (Morris et al., EMNLP 2023, report 92% exact
reconstruction of 32-token inputs; Song & Raghunathan, CCS 2020, recover 50–70%
of words and attribute authorship from a few dozen sentences). The privacy
notice participants read promises them that "Rückschlüsse auf Ihre oder die
Identität anderer Personen sind nicht möglich" and, on the strength of that
promise, that individual records can no longer be deleted — so publishing a
representation that is partially invertible would breach a specific commitment
at the point where participants have already given up their right to withdraw.
Principal components do not solve this, and they fail for a second, independent
reason: the PCA is fitted *inside each cross-validation training fold*, so there
are five different bases and no single component matrix that corresponds to the
reported models — publishing one would be a different quantity from the one
analysed. What we therefore deposit publicly is the 7C scale scores, coarsened
demographics, the full study materials, the complete analysis code, and every
stored model output including the out-of-fold predictions; and because those
predictions are stored, **every reported table and every figure rebuilds from the
public deposit with a single command** (`python code/RUN_ALL.py`, under a
minute), with a built-in check that prints pass or fail against each published
value. Refitting the models from features is the one thing that needs the
embeddings, and those are available to named researchers on request under a data
use agreement.

---

## What is published, and what is not

| | Published openly | Why |
|---|---|---|
| 7C scale scores, all 777 × 7 | yes | not linkable to any external source; published uncoarsened |
| Hesitancy group | yes | operational, from the screening item |
| Demographics | aggregated only | 505 of 777 participants are unique on the combination of age, gender, country, education and politics; row-level release is not anonymous |
| Study materials (all seven open questions, 21 scale items) | yes | — |
| Analysis code, all stages | yes | — |
| Stored model output: out-of-fold predictions, permutation nulls, bootstrap draws, diagnostics | yes | derived quantities, not reconstructible into text |
| Figures and tables, and the scripts that build them | yes | — |
| **Sentence embeddings (1,024-dim)** | **no** | partially invertible; contradicts the consent promise |
| **Principal components** | **no** | same invertibility concern, *and* fold-local — no global basis exists |
| **Raw response text** | **no** | free text written by identifiable people |
| Prolific IDs | no | direct identifiers |

## What a reader can do, at each level of access

| | Access needed | Command | Time | Reproduces |
|---|---|---|---|---|
| **1. Verify** | the public deposit only | `python code/RUN_ALL.py` | < 1 min | every reported table and figure, plus a pass/fail check per published number |
| **2. Execute** | public deposit + synthetic stand-in | `python code/MAKE_SYNTHETIC_DATA.py` then `PAPER2_DATA=data_synthetic python code/RUN_ALL.py --fast` | ~3 min | nothing numerically — it confirms the pipeline runs end to end on data with the real schema |
| **3. Refit** | embeddings, under the data use agreement | `python code/RUN_ALL.py` | ~2 h | every number from the features upward |

Tier 1 is the one that matters for peer review, and it is complete: a reviewer
with nothing but the public files can regenerate Table 1, every supplementary
table, and all five figures, and be told by the script itself whether what they
got matches what we reported. Tier 3 exists because "I would like to try a
different model on the same features" is a legitimate request, and a data use
agreement is the right instrument for it rather than open publication.

## Verification of the claim above

Not asserted — tested. The release was copied to an empty directory with no
`data/` present and run there:

- all five figures regenerate, **byte-identical** to the versions in the paper
  (checked by md5, including `appendix_table6_7c_distributions.png`, which reads
  the published 7C scores);
- `BOOTSTRAP_CIS.py` reproduces `results_bootstrap_cis.json` exactly;
- `VERIFY_INSTALL.py` passes all five published-value checks — Table 1 ridge R²
  and RMSE on all seven dimensions, all seven permutation tests against
  Bonferroni, Table 1 BART R² recomputed from the stored predictions,
  Supplementary Table 19 alphas, Supplementary Table 16 out-of-fold R².

## If formal confirmation is wanted

Queries go to the University of Vienna data protection officer at the functional
address given in the participant privacy notice,
verarbeitungsverzeichnis@univie.ac.at. The notice cites §2f(5) FOG and
Art. 89(1) DSGVO and names the Open Science Framework as the intended archive;
"ein öffentlich zugängliches Archiv" describes Zenodo equally well, which
matters because OSF is winding down its Projects workflow (no new projects after
16 November 2026).

## References for the invertibility claim

- Morris, J. X., Kuleshov, V., Shmatikov, V., & Rush, A. M. (2023). Text
  embeddings reveal (almost) as much as text. *EMNLP 2023*.
- Song, C., & Raghunathan, A. (2020). Information leakage in embedding models.
  *ACM CCS 2020*.
