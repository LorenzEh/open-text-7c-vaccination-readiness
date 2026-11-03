# Requesting the embeddings

Everything needed to **verify** the published numbers is in the public deposit.
This document is for the one thing that is not: **refitting the models from
features**, which needs the sentence embeddings of participants' open-text
responses.

They are not published openly because embeddings of short texts can be partially
inverted to recover the original wording — see the README for the evidence — and
participants were told that inferences about their identity would not be
possible. They are released to researchers under the terms below.

---

## What you get

| File | Size | What it is |
|---|---|---|
| `final_data_embeddings_with_questionnaire.csv` | 48 MB | 1,024-dim embeddings, question text prepended. **The canonical file**; every reported model uses this. |
| `final_data_embeddings.csv` | 47 MB | the same responses embedded without the question prefix; used only for the prompt-dominance robustness check |
| `final_data_merged_cleaned.csv` | 1.1 MB | scale items, dimension scores, demographics, and the exclusion markers the masks are built from |

Not included even under this agreement: **the raw response text** and the
Prolific IDs.

With these three files plus the public repository, one command reproduces every
reported table and figure:

```bash
python code/RUN_ALL.py
```

Roughly two hours end to end, most of it the BART fits. `--fast` skips those and
takes about six minutes.

---

## Terms

By requesting the files you agree to the following. These are not boilerplate;
each one follows from what participants were told.

1. **No re-identification.** You will not attempt to recover the original
   response text from the embeddings, to link records to any other dataset, or
   to identify any participant. This includes embedding-inversion methods, and
   it is the single term that matters most.
2. **No redistribution.** You will not pass the files to anyone outside the
   named recipients, publish them, or deposit them elsewhere. Direct onward
   requests back to the corresponding author.
3. **Research use only**, for the purpose stated in your request.
4. **Secure storage.** Hold the files on access-controlled storage, not in a
   public repository, a shared drive with open links, or a git repository — git
   history makes deletion effectively impossible.
5. **Deletion.** Delete the files when the stated purpose is complete, or within
   three years, whichever is sooner.
6. **Citation.** Cite the paper and the deposit DOI in any work that uses them.
7. **Report problems.** If you find that the embeddings leak more than this
   document claims, tell the corresponding author before publishing it. This is
   a request for coordinated disclosure, not for silence — the finding should be
   published, but participants deserve the chance to be informed first.

---

## How to request

Email the corresponding author with:

- who you are and your institutional affiliation
- what you intend to do with the files
- confirmation that you accept the terms above
- whether your institution requires its own data transfer agreement

A short description is enough; this is not a grant application. Expect a reply
within two weeks.

---

## For the authors: handling a request

- Keep a log of who received the files, when, and for what stated purpose. Under
  Art. 30 GDPR this is a processing activity.
- Send over access-controlled transfer, not email attachments.
- A request from outside the EU/EEA needs a transfer basis — check with the data
  protection officer named in the participant privacy notice before sending.
- If the request is for the **raw text** rather than the embeddings, that is a
  different decision and should not be handled under this agreement.
