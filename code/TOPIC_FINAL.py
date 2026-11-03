# =============================================================================
# 0. Imports & global seeds (100% reproducibility)
# =============================================================================
import os
import random
import re
import warnings
import numpy as np
import pandas as pd
import torch
import nltk
import matplotlib
if not os.environ.get('PAPER2_INTERACTIVE'):
    matplotlib.use('Agg')   # headless: plt.show() would block a plain run
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from transformers import pipeline

# ------------------------------ reproducibility seeds ------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Uncomment and add these strict determinism settings:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Required if using CUDA >= 10.2 with deterministic algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Disable Hugging Face tokenizer multithreading to avoid race conditions
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# 1. Data loading
# =============================================================================
import paths

data = pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
data_cleaned = pd.read_csv(paths.MERGED_CLEANED)

# =============================================================================
# 2. Configuration
# =============================================================================
EMBEDDING_MODEL_NAME = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
SENTIMENT_MIN_TOKENS = 5
BATCH_SIZE = 32

text_columns = [
    'text_confidence',
    'text_complacency',
    'text_constraints',
    'text_calculation',
    'text_coll_resp',
    'text_compliance',
    'text_conspiracy'
]

# =============================================================================
# 3. Stopwords & vectorizer
# =============================================================================
nltk.download('stopwords', quiet=True)
german_stopwords = stopwords.words('german')
custom_stopwords = ['impfen', 'impfstoffe', 'geimpft', 'impfungen',
                    'impfung', 'impfstoffen', 'impfstoff']

stopword_list = sorted(list(set(german_stopwords + custom_stopwords)))

vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=stopword_list)
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
representation_model = KeyBERTInspired()

# =============================================================================
# 4. Marked‑answer masks (exclude specific codes)
# =============================================================================
MARKED_NUMBERS = [-1, 2, 3, 4, 5, 6, 7]
terms = [r'-1' if num == -1 else r'\(' + str(num) + r'\)' for num in MARKED_NUMBERS]
pattern = r'(?:' + '|'.join(terms) + r')$'

marked_masks = {}
for col in text_columns:
    cleaned_vals = data_cleaned[col].reindex(data.index, fill_value="").astype(str)
    marked_masks[col] = cleaned_vals.str.contains(pattern, regex=True, na=False)

# (Optional) print masking summary – omitted for brevity, you can keep it if needed

# =============================================================================
# 5. Helper functions
# =============================================================================
def get_docs_by_group(df, group_value, combine=True):
    """Return list of cleaned, non‑empty, non‑masked text chunks for a group."""
    group_df = df[df['Hesitant'] == group_value]
    docs = []
    for col in text_columns:
        mask = marked_masks[col]
        col_series = group_df[col]
        valid_mask = ~mask.loc[col_series.index]
        col_docs = (col_series[valid_mask].dropna().astype(str).str.strip())
        col_docs = col_docs[col_docs != ""]
        docs.extend(col_docs.tolist())
    return docs


def compute_combined_sentiment(texts, pipe, batch_size=32, min_tokens=5):
    """
    Returns two lists: continuous sentiment scores (pos‑neg) and top label (str).
    """
    if hasattr(texts, 'fillna'):
        texts = texts.fillna("").astype(str).tolist()
    else:
        texts = [str(t) if t is not None else "" for t in texts]

    scores = [np.nan] * len(texts)
    labels = [np.nan] * len(texts)

    valid_indices = []
    valid_texts = []
    for i, txt in enumerate(texts):
        if len(txt.split()) >= min_tokens:
            valid_indices.append(i)
            valid_texts.append(txt)

    if valid_texts:
        for start in range(0, len(valid_texts), batch_size):
            batch = valid_texts[start:start + batch_size]
            outputs = pipe(batch, top_k=None, truncation=True, max_length=512)
            for out, orig_idx in zip(outputs, valid_indices[start:start + batch_size]):
                probs = {pred['label'].lower(): pred['score'] for pred in out}
                scores[orig_idx] = probs.get('positive', 0.0) - probs.get('negative', 0.0)
                labels[orig_idx] = max(probs, key=probs.get)
    return scores, labels


def build_topic_sentiment_df(doc_info):
    """Create a wide DataFrame: Topic, Count, positive, neutral, negative, pct_*."""
    topic_counts = doc_info.groupby('Topic').size().rename('Count')
    cross = pd.crosstab(doc_info['Topic'], doc_info['sentiment_label'])
    for col in ['positive', 'neutral', 'negative']:
        if col not in cross.columns:
            cross[col] = 0
    df = cross.join(topic_counts)
    df['positive_pct'] = df['positive'] / df['Count'] * 100
    df['neutral_pct']  = df['neutral']  / df['Count'] * 100
    df['negative_pct'] = df['negative'] / df['Count'] * 100
    return df.reset_index()


def create_topic_label(topic_model, topic_id, top_n_words=5, max_label_len=50):
    """Return a readable label from the topic model; fallback for outliers."""
    if topic_id == -1:
        return "Outliers / Unassigned"
    words = topic_model.get_topic(topic_id)
    if not words:
        return f"Topic {topic_id} (no words)"
    top_terms = [w[0] for w in words[:top_n_words]]
    label = " / ".join(top_terms)
    if len(label) > max_label_len:
        label = label[:max_label_len - 3] + "..."
    return label


def inspect_topic_examples(doc_info, topic_model=None, n_examples=3, random_state=42):
    """Print random example documents for each topic."""
    np.random.seed(random_state)               # local seed for sampling
    for topic in sorted(doc_info['Topic'].unique()):
        subset = doc_info[doc_info['Topic'] == topic]
        count = len(subset)
        if topic_model and topic != -1:
            words = topic_model.get_topic(topic)
            name = ", ".join([w[0] for w in words[:5]]) if words else "No words"
        elif topic == -1:
            name = "Outliers"
        else:
            name = f"Topic {topic}"
        print(f"\n{'─'*60}")
        print(f"Topic {topic}: {name}  (n={count})")
        print(f"{'─'*60}")
        sample = subset.sample(min(n_examples, count), random_state=random_state)
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            preview = str(row['Document'])[:200]
            print(f"  Example {i}: {preview}{'...' if len(str(row['Document']))>200 else ''}")


def plot_topic_sentiment_on_ax(ax, df, topic_labels, title, min_docs=10,
                               show_percentages=False, pct_threshold=8):
    """Horizontal stacked bars with largest at top, sorted by total doc count."""
    df = df[df['Topic'].isin(topic_labels.keys()) & (df['Topic'] != -1)]
    df = df[df['Count'] >= min_docs]
    if df.empty:
        print(f"No topics to plot for {title}.")
        return

    df = df.sort_values('Count', ascending=True).reset_index(drop=True)  # smallest→bottom, largest→top
    topics = df['Topic'].values
    labels = [topic_labels[t] for t in topics]
    y_pos = np.arange(len(topics))

    colors = {'positive': '#2E8B57', 'neutral': '#808080', 'negative': '#CD5C5C'}
    left = np.zeros(len(topics))
    for sentiment, color in colors.items():
        counts = df[sentiment].values
        ax.barh(y_pos, counts, left=left, color=color,
                label=sentiment.capitalize(), edgecolor='white', linewidth=0.5)
        left += counts

    if show_percentages:
        for pos_idx, (_, row) in enumerate(df.iterrows()):
            total = row['Count']
            if total == 0:
                continue
            cum = 0
            for sentiment, color in colors.items():
                count = row[sentiment]
                pct = count / total * 100
                if pct >= pct_threshold:
                    ax.text(cum + count / 2, pos_idx, f'{pct:.0f}%',
                            ha='center', va='center', color='white',
                            fontsize=10, fontweight='bold')
                cum += count

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_title(title, loc='left', fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(False)

# =============================================================================
# 6. Load models (embedding & sentiment – done once)
# =============================================================================
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Loading sentiment pipeline...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="oliverguhr/german-sentiment-bert",
    tokenizer="oliverguhr/german-sentiment-bert",
    top_k=None
)

# =============================================================================
# 7. Main analysis – loop over two groups (standard hyphen used everywhere)
# =============================================================================
doc_info_dict = {}
group_sentiments = {}
topic_models = {}

umap_settings = {
    'Hesitant':     {'n_neighbors': 10, 'n_components': 5, 'min_dist': 0.3, 
                     'metric': 'cosine', 'random_state': 42},
    
    'Non-hesitant': {'n_neighbors': 10, 'n_components': 5, 'min_dist': 0.02, 
                     'metric': 'cosine', 'random_state': 42}
}

hdb_settings = {
    'Hesitant':     {'min_cluster_size': 30, 'metric': 'euclidean', 'cluster_selection_method': 'leaf', 
                     'prediction_data': True, 'approx_min_span_tree': False, 'core_dist_n_jobs': 1},
    
    'Non-hesitant': {'min_cluster_size': 60, 'metric': 'euclidean', 'cluster_selection_method': 'leaf', 
                     'prediction_data': True, 'approx_min_span_tree': False, 'core_dist_n_jobs': 1}
}





for group_name, group_val in [('Hesitant', 1), ('Non-hesitant', 0)]:
    print(f"\n{'='*60}")
    print(f"Processing combined texts – {group_name} group")
    print('='*60)

    docs = get_docs_by_group(data, group_val)
    if len(docs) < 10:
        print(f"  Skipped – only {len(docs)} documents.")
        continue
    print(f"Number of documents: {len(docs)}")

    # Sentiment
    scores, labels = compute_combined_sentiment(docs, sentiment_pipeline,
                                                batch_size=BATCH_SIZE,
                                                min_tokens=SENTIMENT_MIN_TOKENS)
    valid_mask = ~np.isnan(np.array(scores))
    valid_scores = np.array(scores)[valid_mask]
    valid_labels = np.array(labels)[valid_mask]

    if len(valid_scores) > 0:
        avg_sent = valid_scores.mean()
        pos_r = (valid_labels == 'positive').mean()
        neg_r = (valid_labels == 'negative').mean()
        neu_r = (valid_labels == 'neutral').mean()
    else:
        avg_sent = pos_r = neg_r = neu_r = 0.0

    group_sentiments[group_name] = {
        'avg_sentiment': avg_sent, 'positive_ratio': pos_r,
        'negative_ratio': neg_r, 'neutral_ratio': neu_r,
        'n_valid': len(valid_scores), 'n_total': len(docs)
    }

    print(f"\nSentiment summary for {group_name}:")
    print(f"  Valid docs: {len(valid_scores)}/{len(docs)}")
    print(f"  Avg polarity: {avg_sent:.3f}")
    print(f"  Pos: {pos_r*100:.1f}%  Neg: {neg_r*100:.1f}%  Neu: {neu_r*100:.1f}%")

    # UMAP + HDBSCAN + BERTopic
    umap_model = UMAP(**umap_settings[group_name])
    """
    hdbscan_model = HDBSCAN(min_cluster_size=25, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True,
                            approx_min_span_tree=False, core_dist_n_jobs=1)
    """
    hdbscan_model = HDBSCAN(**hdb_settings[group_name])
    topic_model = BERTopic(embedding_model=embedding_model,
                           vectorizer_model=vectorizer_model,
                           hdbscan_model=hdbscan_model,
                           umap_model=umap_model,
                           ctfidf_model=ctfidf_model,
                           representation_model=representation_model,
                           verbose=True)
    
    topics, probs = topic_model.fit_transform(docs)

    # DBCV
    reduced = topic_model.umap_model.embedding_.astype(np.float64)
    dbcv_labels = np.array(topic_model.topics_)
    if len(set(dbcv_labels) - {-1}) >= 2:
        dbcv = validity_index(reduced, dbcv_labels)
        print(f"\n  DBCV score: {dbcv:.4f}")
    else:
        print("\n  DBCV score: Not available (<2 clusters)")

    topic_models[group_name] = topic_model

    # Topic info
    info = topic_model.get_topic_info().sort_values('Count', ascending=False)
    print(info[['Topic', 'Count', 'Name']].head(15))

    print("\nTop words per topic:")
    for tid in info['Topic'].values:
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        print(f"Topic {tid}: " + ", ".join([f"{w[0]} ({w[1]:.3f})" for w in words[:5]]))

    # Store doc info
    doc_info = topic_model.get_document_info(docs)
    doc_info['sentiment_score'] = scores
    doc_info['sentiment_label'] = labels
    doc_info_dict[group_name] = doc_info

# =============================================================================
# 8. Post‑processing: topic examples & sentiment per topic
# =============================================================================

for grp in ['Hesitant', 'Non-hesitant']:
    print(f"\n--- Topic examples for {grp} ---")
    inspect_topic_examples(doc_info_dict[grp], topic_models[grp], n_examples=10)

print("\n" + "="*60)
print("SENTIMENT PER TOPIC (all groups)")
print("="*60)
for grp, doc_info in doc_info_dict.items():
    print(f"\n>>> {grp}")
    stats = doc_info.groupby(['Topic', 'sentiment_label']).size().unstack(fill_value=0)
    for col in ['positive', 'neutral', 'negative']:
        if col not in stats: stats[col] = 0
    pct = stats.div(stats.sum(axis=1), axis=0) * 100
    names = doc_info.groupby('Topic')['Name'].first()
    for tid in stats.sum(axis=1).sort_values(ascending=False).index:
        name = names[tid] if tid != -1 else "Outliers"
        print(f"Topic {tid} ({name}) [n={int(stats.loc[tid].sum())}]: "
              f"Pos:{pct.loc[tid, 'positive']:5.1f}% | "
              f"Neu:{pct.loc[tid, 'neutral']:5.1f}% | "
              f"Neg:{pct.loc[tid, 'negative']:5.1f}%")



# =============================================================================
# 9. Build sentiment DataFrames for plotting
# =============================================================================
hesitant_sent_df = build_topic_sentiment_df(doc_info_dict['Hesitant'])
nonhesitant_sent_df = build_topic_sentiment_df(doc_info_dict['Non-hesitant'])

# Generate auto‑labels and apply manual overrides (only for IDs that exist)
hesitant_auto_labels = {tid: create_topic_label(topic_models['Hesitant'], tid)
                        for tid in hesitant_sent_df['Topic']}
nonhesitant_auto_labels = {tid: create_topic_label(topic_models['Non-hesitant'], tid)
                           for tid in nonhesitant_sent_df['Topic']}


# sp those need to be overhowled
# Manual labels – adjust IDs to the ones actually present (check printed output above)

hesitant_overrides = {
    0: "Side Effects & Risk Weighting",
    1: "Pharmaceutical Skepticism ",
    2: "Decisional Autonomy",
    3: "Vaccine Injury & COVID",
    4: "Categorical Opposition & Social Divide"
}

nonhesitant_overrides = {
    0: "Institutional & Scientific Trust",
    1: "Side Effects & Infection Risk",
    2: "Vaccine Access & Logistics",
    3: "Flu Vaccine & Health",
    4: "Science & Medicine"
}



# Update labels (overrides take precedence)
hesitant_labels = {**hesitant_auto_labels, **hesitant_overrides}
nonhesitant_labels = {**nonhesitant_auto_labels, **nonhesitant_overrides}

print("\n--- Hesitant topic labels ---")
for tid, lbl in hesitant_labels.items():
    print(f"Topic {tid}: {lbl}")
print("\n--- Non‑hesitant topic labels ---")
for tid, lbl in nonhesitant_labels.items():
    print(f"Topic {tid}: {lbl}")


# =============================================================================
# 10. Final plot – sorted with largest bar at top
# =============================================================================

def clean_hyphen(text):
    """Replace non-breaking hyphen (U+2011) with regular hyphen."""
    return text.replace('\u2011', '-')

def plot_topic_sentiment_on_ax(
    ax,
    df,
    topic_labels,
    title,
    min_docs=10,
    show_percentages=False,
    pct_threshold=8,
    show_sentiment=True,  
    x_max = None
):

    """Plot stacked (or single) horizontal bar chart on the given axes."""
    plot_df = df[df['Topic'].isin(topic_labels.keys())].copy()
    plot_df = plot_df[plot_df['Topic'] != -1]
    
    plot_df['Total_Docs'] = plot_df['positive'] + plot_df['neutral'] + plot_df['negative']
    plot_df = plot_df[plot_df['Total_Docs'] >= min_docs]
    if plot_df.empty:
        print(f"No topics to plot for {title}.")
        return

    plot_df = plot_df.sort_values('Total_Docs', ascending=True).reset_index(drop=True)
    topics = plot_df['Topic'].values
    labels = [clean_hyphen(topic_labels[t]) for t in topics]
    y_pos = np.arange(len(topics))

    if show_sentiment:
        # Original stacked sentiment bars
        colors = {
            'positive': '#1338BE',
            'neutral':  '#808080',
            'negative': '#CD5C5C'
        }
        left_pos = np.zeros(len(topics))
        for sentiment, color in colors.items():
            counts = plot_df[sentiment].values
            ax.barh(y_pos, counts, left=left_pos, color=color,
                    label=sentiment.capitalize(), edgecolor='white', linewidth=0.5)
            left_pos += counts

        if show_percentages:
            for pos_idx, row in plot_df.iterrows():
                total = row['Total_Docs']
                if total == 0:
                    continue
                cum = 0
                for sentiment, color in colors.items():
                    count = row[sentiment]
                    pct = count / total * 100
                    if pct >= pct_threshold:
                        ax.text(cum + count/2, pos_idx, f'{pct:.0f}%',
                                ha='center', va='center', color='white',
                                fontsize=10, fontweight='bold')
                    cum += count
    else:
        # Single solid bar for total documents
        ax.barh(y_pos, plot_df['Total_Docs'], color='#4C72B0',  # choose any neutral color
                edgecolor='white', linewidth=0.5)
        # No percentages shown when show_sentiment=False (or you could show total count)
        # If you still want percentages, you could print the total count instead,
        # but the original percentage logic is sentiment-specific.

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_title(clean_hyphen(title), loc='left', fontsize=16, fontweight='bold')
    
    # Set x-axis limit if requested
    if x_max is not None:
        ax.set_xlim(right=x_max)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(False)
    
    
# Create figure with two subplots, independent x-axes
fig, axes = plt.subplots(
    nrows=2, ncols=1,
    figsize=(12, 12),
    sharex=False,
    gridspec_kw={'hspace': 0.3}
)

# Plot hesitant group (top)
plot_topic_sentiment_on_ax(
    ax=axes[0],
    df=hesitant_sent_df,
    topic_labels=hesitant_overrides,
    title='Hesitant Respondents',
    min_docs=5,
    show_sentiment=False,
    x_max = 250
)

# Plot non-hesitant group (bottom)
plot_topic_sentiment_on_ax(
    ax=axes[1],
    df=nonhesitant_sent_df,
    topic_labels=nonhesitant_overrides,
    title='Non-Hesitant Respondents',
    min_docs=5,
    show_sentiment=False
)

# X-axis label only on the bottom subplot (close to its axis)
axes[1].set_xlabel('Number of Documents', fontsize=18, labelpad=8)

# Collect legend handles/labels from the top subplot (both have same categories)
handles, labels = axes[0].get_legend_handles_labels()

"""
# Place a single legend inside the bottom-right corner of the lower plot,
# with a title "Sentiment"
axes[1].legend(
    handles, labels,
    loc='lower right',
    frameon=True,
    fancybox=False,
    edgecolor='lightgray',
    title='Sentiment',
    title_fontsize=16,
    prop={'size': 16}
)
"""

plt.tight_layout()
# Interactive runs pop the window; headless runs save instead, so the script
# can be executed end to end without a display. Set PAPER2_INTERACTIVE=1 for
# the old behaviour.
if os.environ.get('PAPER2_INTERACTIVE'):
    plt.show()
else:
    paths.OUTPUT.mkdir(exist_ok=True)
    plt.savefig(paths.OUTPUT/'figure1_topic_distributions.png', dpi=300, bbox_inches='tight')
    print('wrote', paths.OUTPUT/'figure1_topic_distributions.png')

