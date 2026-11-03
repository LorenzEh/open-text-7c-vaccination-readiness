"""Reproduce every reported number, in order, and say whether it matched.

This is the entry point for someone who has the embeddings — a co-author, or a
researcher who has received them under the data use agreement. One command takes
the dataset to the full set of reported tables and figures, then checks the
result against the published values and prints a verdict per stage.

    python code/RUN_ALL.py                # everything, ~50 minutes
    python code/RUN_ALL.py --fast         # skip BART, ~6 minutes
    python code/RUN_ALL.py --list         # show the stages and exit

Design notes
------------
Stages are ordered by dependency, not by the paper's section order: the sentiment
classifier has to run before the regressions that consume its labels, and the
BART fits have to run before the figures that plot their predictions.

Each stage names the environment it needs. Two are required and they are not
interchangeable — the topic/sentiment stack pins a different NumPy and Python
from the modelling stack. The script does not switch environments for you; it
checks whether the current one can run the stage and skips with a clear message
if not, so a partial run is informative rather than a crash.

A stage that fails does not stop the run. The point is a complete report of what
reproduced and what did not, which is more useful than halting on the first
discrepancy.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

CODE = os.path.dirname(os.path.abspath(__file__))

# (script, environment, approximate minutes, needs the dataset?, what it produces)
#
# `needs_data` is the difference between refitting a model and rebuilding a
# reported number from the stored out-of-fold predictions. The four stages
# marked False are why the public release reproduces every table and figure
# without the embeddings: they read results/, not data/.
STAGES = [
    ("SENTIMENT_GROUP_TEST.py",      "text", 5,  True,  "sentiment labels; Table 7 and the chi-square"),
    ("TOPIC_FINAL.py",               "text", 5,  True,  "Figure 1 and the topic tables"),
    ("TOPIC_COUNTS_HESITANT.py",     "text", 1,  True,  "hesitant topic sizes"),
    ("SENTIMENT_FINAL.py",           "bart", 1,  True,  "Table 16"),
    ("SENTIMENT_REGRESSION_PERDIM.py", "bart", 1, True, "per-dimension sentiment regressions"),
    ("SENTIMENT_BY_DIM_GROUP.py",    "bart", 1,  False, "sentiment by dimension and group"),
    ("CRONBACH_ALPHA.py",            "bart", 1,  True,  "Table 19"),
    ("DEMOGRAPHICS.py",              "bart", 1,  True,  "Table 3"),
    ("TABLE4_NATIVE_DIRECTION.py",   "bart", 1,  True,  "Table 4"),
    ("COMPARISON_PC_EMBEDDINGS.py",  "bart", 2,  True,  "Tables 8 and 9"),
    ("RIDGE_UNSCALED_CHECK.py",      "bart", 25, True,  "Table 1 ridge rows, Figure 3 nulls"),
    ("RIDGE_FULL_METRICS.py",        "bart", 3,  True,  "Tables 14 and 15"),
    ("DISCRIMINANT_VALIDITY.py",     "bart", 5,  True,  "Table 17"),
    ("BART_FULL_METRICS.py",         "bart", 35, True,  "Table 1 BART rows, Tables 10-13"),
    ("BART_NO_PCA_CHECK.py",         "bart", 30, True,  "Table 18"),
    ("BOOTSTRAP_CIS.py",             "bart", 1,  False, "Table 1 confidence intervals"),
    ("MAKE_FIGURES.py",              "bart", 1,  False, "Figures 2 and 3, appendix figures"),
    ("EXPORT_ALL_RESULTS.py",        "bart", 1,  False, "every table as CSV"),
]
SLOW = {"RIDGE_UNSCALED_CHECK.py", "BART_FULL_METRICS.py", "BART_NO_PCA_CHECK.py",
        "DISCRIMINANT_VALIDITY.py"}


def env_available(which):
    """Can the interpreter running this script execute a `which` stage?"""
    if which == "text":
        return importlib.util.find_spec("bertopic") is not None
    return importlib.util.find_spec("pymc_bart") is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the slow refits (BART, permutations, discriminant matrix)")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        total = sum(m for _, _, m, _, _ in STAGES)
        print(f"{'stage':34}{'env':6}{'min':>5}{'data':>6}  produces")
        for s, e, m, nd, w in STAGES:
            print(f"{s:34}{e:6}{m:>5}{'yes' if nd else 'no':>6}  {w}")
        print(f"\n{len(STAGES)} stages, roughly {total} minutes in total.")
        print(f"{sum(1 for _, _, _, nd, _ in STAGES if not nd)} of them need no "
              f"participant data: they rebuild reported numbers from results/.")
        return

    have_data = paths.EMBEDDINGS_WITH_QUESTIONNAIRE.exists()
    synthetic = False
    if have_data:
        import pandas as pd
        synthetic = "synthetic" in pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE,
                                               nrows=1).columns
        print(f"data:    {paths.EMBEDDINGS_WITH_QUESTIONNAIRE}")
        print(f"         {'SYNTHETIC — numbers will not match the paper' if synthetic else 'real'}")
    else:
        # Not an error. The public release deliberately ships results/ without
        # data/, and every table and figure is rebuildable from the stored
        # out-of-fold predictions. Run that, rather than refusing to start.
        print(f"data:    none at {paths.EMBEDDINGS_WITH_QUESTIONNAIRE}")
        print("         Running the stages that rebuild reported numbers from")
        print("         results/ instead. Refitting needs the embeddings, which")
        print("         are available on request — see docs/DATA_USE_AGREEMENT.md.")
    print(f"results: {paths.RESULTS}")
    if synthetic:
        print("         (redirected, so the published results cannot be overwritten)")

    stages = [s for s in STAGES if not (args.fast and s[0] in SLOW)]
    if not have_data:
        stages = [s for s in stages if not s[3]]
    print(f"\nrunning {len(stages)} of {len(STAGES)} stages\n")

    done, failed, skipped, blocked = [], [], [], []
    t_all = time.time()
    for script, env, mins, _needs_data, what in stages:
        if not env_available(env):
            print(f"SKIP  {script:32} needs the '{env}' environment")
            skipped.append((script, env))
            continue
        print(f"RUN   {script:32} ~{mins} min  ({what})", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(CODE, script)],
                           capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"  ok    {dt/60:.1f} min")
            done.append(script)
        else:
            tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            # Distinguish "this stage is broken" from "an input it needed was
            # never produced because an earlier stage was skipped".
            missing_input = "FileNotFoundError" in tail and skipped
            label = "DEPS" if missing_input else "FAIL"
            note = ("  (an earlier stage was skipped; re-run after it)"
                    if missing_input else "")
            print(f"  {label}  {dt/60:.1f} min  {tail[:100]}{note}")
            (blocked if missing_input else failed).append((script, tail))

    print(f"\n{'='*70}")
    print(f"{len(done)} ok, {len(failed)} failed, {len(blocked)} blocked by a "
          f"skipped stage, {len(skipped)} skipped, {(time.time()-t_all)/60:.0f} min total")
    if skipped:
        envs = sorted({e for _, e in skipped})
        print(f"\nskipped stages need: {', '.join(envs)}. Activate that environment and")
        print("re-run; completed stages are not redone from scratch where the script")
        print("supports resuming (BART_FULL_METRICS and BART_4CHAIN_CHECK do).")
    for s, why in failed:
        print(f"  FAILED {s}: {why[:140]}")

    if synthetic:
        print("\nSynthetic data: no comparison against published values is meaningful.")
        return
    if not have_data:
        n_refit = sum(1 for _, _, _, nd, _ in STAGES if nd)
        print(f"\n{n_refit} refitting stages were not run — they need the embeddings.")
        print("Everything above was rebuilt from the stored predictions, so the")
        print("tables and figures in output/ are the published ones.")
    print("\nchecking the result against the published numbers:")
    r = subprocess.run([sys.executable, os.path.join(CODE, "VERIFY_INSTALL.py")],
                       capture_output=True, text=True)
    print("\n".join(r.stdout.splitlines()[-12:]))
    sys.exit(1 if (failed or r.returncode) else 0)


if __name__ == "__main__":
    main()
