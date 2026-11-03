"""Regenerate the hesitant group's topic-size table.

The appendix's raw topic-modelling log contains the "Topic Count Name" table for
the non-hesitant group but not for the hesitant one, so the five hesitant topic
counts quoted in the Results (235, 90, 62, 57, 32) have no appendix counterpart.
This script refits only the hesitant model and writes that table.

Configuration is copied verbatim from TOPIC_FINAL.py -- same seeds, stopwords,
vectorizer, c-TF-IDF, KeyBERTInspired representation, and the hesitant-specific
UMAP/HDBSCAN settings -- so the output is the same model, not a similar one.
Sentiment is not recomputed here; it is unchanged and already in Table 7.

Run with the sentence_transformers env (TOPIC_FINAL.py's env):
    /opt/anaconda3/envs/sentence_transformers/bin/python code/TOPIC_COUNTS_HESITANT.py
"""
import os
import random
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

EMBEDDING_MODEL_NAME = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
TEXT_COLUMNS = ["text_confidence", "text_complacency", "text_constraints",
                "text_calculation", "text_coll_resp", "text_compliance",
                "text_conspiracy"]
MARKED_NUMBERS = [-1, 2, 3, 4, 5, 6, 7]

data = pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
data_cleaned = pd.read_csv(paths.MERGED_CLEANED)

terms = [r"-1" if n == -1 else r"\(" + str(n) + r"\)" for n in MARKED_NUMBERS]
pattern = r"(?:" + "|".join(terms) + r")$"
marked = {c: data_cleaned[c].reindex(data.index, fill_value="").astype(str)
                            .str.contains(pattern, regex=True, na=False)
          for c in TEXT_COLUMNS}


def docs_for(group_value):
    g = data[data["Hesitant"] == group_value]
    out = []
    for col in TEXT_COLUMNS:
        s = g[col]
        s = s[~marked[col].loc[s.index]].dropna().astype(str).str.strip()
        out.extend(s[s != ""].tolist())
    return out


def main():
    nltk.download("stopwords", quiet=True)
    stop = sorted(set(stopwords.words("german") + [
        "impfen", "impfstoffe", "geimpft", "impfungen",
        "impfung", "impfstoffen", "impfstoff"]))

    docs = docs_for(1)
    print(f"Number of documents: {len(docs)}", flush=True)

    model = BERTopic(
        embedding_model=SentenceTransformer(EMBEDDING_MODEL_NAME),
        vectorizer_model=CountVectorizer(ngram_range=(1, 3), stop_words=stop),
        hdbscan_model=HDBSCAN(min_cluster_size=30, metric="euclidean",
                              cluster_selection_method="leaf", prediction_data=True,
                              approx_min_span_tree=False, core_dist_n_jobs=1),
        umap_model=UMAP(n_neighbors=10, n_components=5, min_dist=0.3,
                        metric="cosine", random_state=SEED),
        ctfidf_model=ClassTfidfTransformer(bm25_weighting=True),
        representation_model=KeyBERTInspired(),
        verbose=True)
    model.fit_transform(docs)

    reduced = model.umap_model.embedding_.astype(np.float64)
    labels = np.array(model.topics_)
    dbcv = validity_index(reduced, labels) if len(set(labels) - {-1}) >= 2 else float("nan")
    print(f"DBCV score: {dbcv:.4f}", flush=True)

    info = model.get_topic_info().sort_values("Count", ascending=False)
    info = info[["Topic", "Count", "Name"]].reset_index(drop=True)
    print(info.to_string())

    out = paths.OUTPUT / "topic_counts_hesitant.csv"
    info.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"DBCV {dbcv:.4f}   n_docs {len(docs)}   n_topics {len(info)-1}")


if __name__ == "__main__":
    main()
