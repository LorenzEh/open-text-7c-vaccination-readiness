"""Project paths.

Every script resolves data locations through here instead of hardcoding an
absolute path, so the project can be moved without editing 13 scripts.

Usage:

    from paths import DATA, RAW
    data = pd.read_csv(DATA / "final_data_embeddings_with_questionnaire.csv")

Works when run as a script, imported, or executed cell-by-cell in an IDE
(Spyder/VS Code), where __file__ may be undefined -- in that case the project
root is found by walking up from the working directory.

On Google Colab (generate_embedding_for_bart_*.py,
paper_2_regression_sentiment_num_scores.py) DATA points at the mounted Drive
folder instead; see COLAB below.
"""

import os
from pathlib import Path

COLAB = Path("/content/drive/My Drive")

# Directories that must exist directly under the project root for it to be
# recognised. A root must have code/ plus one sibling that pins it down, so a
# bare parent directory containing an unrelated code/ does not match.
#
# Two layouts are supported and both must keep working:
#   working repo  code/ + data/ + CLAUDE.md, results sit in code/
#   public release  code/ + results/ + README.md, no data/ at all
# The release has no data/, so requiring it here made the release unrunnable.
_ANCHORS = ("data", "results", "CLAUDE.md", "README.md")


def _looks_like_root(p: Path) -> bool:
    return (p / "code").is_dir() and any((p / a).exists() for a in _ANCHORS)


def _find_root() -> Path:
    candidates = []

    # Preferred: location of this file -> parent of code/
    try:
        candidates.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass  # cell-by-cell execution: no __file__

    # Fallback: walk up from the working directory
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    for c in candidates:
        if _looks_like_root(c):
            return c

    raise RuntimeError(
        "Could not locate the paper_2 project root. Expected a directory "
        f"containing {_MARKERS}. Searched from {Path.cwd()}. "
        "Run from inside the project, or set PROJECT_ROOT manually."
    )


if COLAB.exists():
    # Running on Colab: the CSVs sit flat in Drive, there is no project tree.
    PROJECT_ROOT = COLAB
    DATA = COLAB
    RAW = COLAB
    CODE = COLAB
    OUTPUT = COLAB
else:
    PROJECT_ROOT = _find_root()
    # PAPER2_DATA repoints the dataset directory without editing any script.
    # Used to run the pipeline against the synthetic dataset
    # (code/MAKE_SYNTHETIC_DATA.py), which has the same schema but no real
    # responses, so the code is executable without access to the real data.
    DATA = (Path(os.environ["PAPER2_DATA"]).expanduser()
            if os.environ.get("PAPER2_DATA") else PROJECT_ROOT / "data")
    if not DATA.is_absolute():
        DATA = PROJECT_ROOT / DATA
    RAW = PROJECT_ROOT / "complete_dataset"   # raw REDCap exports (2026-03-02)
    CODE = PROJECT_ROOT / "code"
    OUTPUT = PROJECT_ROOT / "output"          # figures/tables; created on demand

# Stored model outputs. They sit beside the scripts in the working repo and in a
# separate results/ directory in the public release; scripts refer to them as
# paths.RESULTS so both layouts work without edits. PAPER2_RESULTS overrides.
RESULTS_PUBLISHED = Path(os.environ["PAPER2_RESULTS"]).expanduser() \
    if os.environ.get("PAPER2_RESULTS") \
    else (PROJECT_ROOT / "results" if (PROJECT_ROOT / "results").is_dir() else CODE)

# A run against a substitute dataset must NOT write into the published results.
# This is not hypothetical: a full ridge permutation run against the synthetic
# data overwrote all seven published R2 with values near zero, and was only
# caught because the file is tracked in git. When PAPER2_DATA is set, results
# and figures are redirected into that dataset's own directory, so synthetic
# runs are self-contained and the published numbers cannot be clobbered.
# Reading the published results stays available as RESULTS_PUBLISHED.
_SUBSTITUTE_DATA = bool(os.environ.get("PAPER2_DATA"))
if _SUBSTITUTE_DATA:
    RESULTS = DATA / "results"
    OUTPUT = DATA / "output"
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
else:
    RESULTS = RESULTS_PUBLISHED


_ANNOUNCED_FALLBACKS = set()


def result(name):
    """Locate a stored result for READING, in either mode.

    Under PAPER2_DATA, prefer a file the substitute run produced itself; fall
    back to the published one so verification tools (figures, bootstrap CIs)
    still work before the refitting scripts have been run. Writers should use
    paths.RESULTS directly, which is redirected and never falls back.
    """
    here = RESULTS / name
    if here.exists():
        return here
    if _SUBSTITUTE_DATA and name not in _ANNOUNCED_FALLBACKS:
        # Say so. A stage that silently reads a real-data artefact while being
        # fed substitute data would otherwise look like it had reproduced
        # something it has not.
        _ANNOUNCED_FALLBACKS.add(name)
        print(f"  note: reading published {name} — this run has not produced "
              f"its own copy", flush=True)
    return RESULTS_PUBLISHED / name


# --- Canonical datasets -----------------------------------------------------
# See CLAUDE.md for the full lineage.

EMBEDDINGS_WITH_QUESTIONNAIRE = DATA / "final_data_embeddings_with_questionnaire.csv"
EMBEDDINGS = DATA / "final_data_embeddings.csv"
MERGED_CLEANED = DATA / "final_data_merged_cleaned.csv"
MERGED = DATA / "final_data_merged.csv"
TRANSLATED = DATA / "final_translated_data.csv"

# Raw REDCap exports, split by Prolific listing.
RAW_NON_HESITANT = RAW / "ExploringAdultExperi_DATA_LABELS_2026-03-02_1003.csv"
RAW_HESITANT = RAW / "ExploringAdultExperi_DATA_LABELS_2026-03-02_1004.csv"


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA:         {DATA}")
    print(f"RESULTS:      {RESULTS}" + ("   (redirected: PAPER2_DATA is set)"
                                        if _SUBSTITUTE_DATA else ""))
    print(f"OUTPUT:       {OUTPUT}")
    for name in (
        "EMBEDDINGS_WITH_QUESTIONNAIRE",
        "EMBEDDINGS",
        "MERGED_CLEANED",
        "MERGED",
        "TRANSLATED",
        "RAW_NON_HESITANT",
        "RAW_HESITANT",
    ):
        p = globals()[name]
        print(f"  {'ok ' if p.exists() else 'MISSING'}  {name}: {p}")
