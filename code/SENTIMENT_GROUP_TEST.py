"""Sentiment category x hesitancy group: chi-square test, plus per-dimension sentiment.

Addresses two open co-author comments on v3:
  (a) the sentiment-distribution difference between groups is asserted but untested;
  (b) which dimension elicits the most positive / most negative sentiment.

Classification follows paper_2_regression_sentiment_num_scores.py exactly:
oliverguhr/german-sentiment-bert, marked rows excluded, responses under five
tokens excluded, prompt text NOT included.

Note on the unit of analysis: each participant contributes up to seven
responses, so a chi-square over responses treats clustered observations as
independent. A participant-level test is therefore also reported.
"""
import os, sys, json
os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, pandas as pd, torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

TEXT_COLS=['text_confidence','text_complacency','text_constraints','text_calculation',
           'text_coll_resp','text_compliance','text_conspiracy']
DIM={'text_confidence':'Confidence','text_complacency':'Complacency',
     'text_constraints':'Constraints','text_calculation':'Calculation',
     'text_coll_resp':'Collective Responsibility','text_compliance':'Compliance',
     'text_conspiracy':'Conspiracy'}
PAT=r'(?:-1|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\))$'
MIN_TOKENS=5
OUT=str(paths.RESULTS/"results_sentiment_group_test.json")

def main():
    data=pd.read_csv(paths.EMBEDDINGS_WITH_QUESTIONNAIRE)
    dc=pd.read_csv(paths.MERGED_CLEANED)
    mdl="oliverguhr/german-sentiment-bert"
    pipe=pipeline("text-classification",
                  model=AutoModelForSequenceClassification.from_pretrained(mdl),
                  tokenizer=AutoTokenizer.from_pretrained(mdl),
                  device=0 if torch.cuda.is_available() else -1,
                  batch_size=32, truncation=True)
    recs=[]
    for col in TEXT_COLS:
        marked=dc[col].reindex(data.index,fill_value="").astype(str).str.contains(PAT,regex=True,na=False)
        txt=data[col].fillna("").astype(str)
        keep=(~marked) & (txt.str.split().str.len()>=MIN_TOKENS)
        idx=data.index[keep]
        if len(idx)==0: continue
        outs=pipe(txt.loc[idx].tolist(), top_k=None)
        for i,o in zip(idx,outs):
            probs={p['label'].lower():p['score'] for p in o}
            recs.append(dict(row=int(i), dim=DIM[col],
                             hesitant=int(data.at[i,'Hesitant']),
                             label=max(probs,key=probs.get),
                             polarity=probs.get('positive',0)-probs.get('negative',0)))
        print(f"  {col}: {len(idx)} responses classified",flush=True)
    df=pd.DataFrame(recs)
    df.to_csv(paths.RESULTS/"results_sentiment_labels.csv",index=False)

    cats=["negative","neutral","positive"]
    res={}

    # ---- (1) response-level chi-square (what the co-author comment asks for)
    tab=pd.crosstab(df.hesitant, df.label).reindex(columns=cats).fillna(0).astype(int)
    chi2,p,dof,exp=stats.chi2_contingency(tab.values)
    v=float(np.sqrt(chi2/(tab.values.sum()*(min(tab.shape)-1))))
    res["response_level"]=dict(table={int(k):dict(zip(cats,map(int,r)))
                                      for k,r in zip(tab.index,tab.values)},
        n=int(tab.values.sum()), chi2=float(chi2), dof=int(dof), p=float(p), cramers_v=v,
        n_participants=int(df.row.nunique()),
        responses_per_participant=float(len(df)/df.row.nunique()))
    print(f"\nRESPONSE-LEVEL chi2({dof}) = {chi2:.2f}, p = {p:.3e}, V = {v:.3f}, N = {tab.values.sum()}")
    print(tab.to_string())

    # ---- (2) participant-level: each participant's modal sentiment (independent units)
    mode=(df.groupby(['row','hesitant']).label
            .agg(lambda s: s.value_counts().idxmax()).reset_index())
    tab2=pd.crosstab(mode.hesitant, mode.label).reindex(columns=cats).fillna(0).astype(int)
    chi2b,pb,dofb,_=stats.chi2_contingency(tab2.values)
    vb=float(np.sqrt(chi2b/(tab2.values.sum()*(min(tab2.shape)-1))))
    res["participant_level_modal"]=dict(table={int(k):dict(zip(cats,map(int,r)))
                                               for k,r in zip(tab2.index,tab2.values)},
        n=int(tab2.values.sum()), chi2=float(chi2b), dof=int(dofb), p=float(pb), cramers_v=vb)
    print(f"\nPARTICIPANT-LEVEL (modal) chi2({dofb}) = {chi2b:.2f}, p = {pb:.3e}, V = {vb:.3f}, N = {tab2.values.sum()}")
    print(tab2.to_string())

    # ---- (3) per-category follow-ups, Bonferroni x3
    res["followups"]={}
    for j,c in enumerate(cats):
        sub=np.array([[tab.values[i,j], tab.values[i].sum()-tab.values[i,j]] for i in range(2)])
        c2,pp,_,_=stats.chi2_contingency(sub)
        res["followups"][c]=dict(chi2=float(c2),p=float(pp),p_bonferroni=float(min(1,pp*3)))

    # ---- (4) sentiment by dimension
    res["by_dimension"]={}
    for d,g in df.groupby('dim'):
        vc=g.label.value_counts()
        res["by_dimension"][d]=dict(n=int(len(g)), mean_polarity=float(g.polarity.mean()),
            **{f"pct_{c}": float(100*vc.get(c,0)/len(g)) for c in cats})
    print("\nBY DIMENSION (mean polarity, +1 = positive):")
    for d,v_ in sorted(res["by_dimension"].items(), key=lambda kv:-kv[1]['mean_polarity']):
        print(f"  {d:26} n={v_['n']:4}  polarity={v_['mean_polarity']:+.3f}  "
              f"neg={v_['pct_negative']:4.1f}%  neu={v_['pct_neutral']:4.1f}%  pos={v_['pct_positive']:4.1f}%")
    json.dump(res,open(OUT,'w'),indent=1)
    print("\nDONE")

if __name__=="__main__":
    main()
