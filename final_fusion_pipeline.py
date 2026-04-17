import pandas as pd
import re
import numpy as np
import os
import time
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

log = open('final_fusion_log.txt', 'w', encoding='utf-8')
def p(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()

p("=" * 60)
p("Final Fusion Pipeline: NBSVM + TF-IDF LR + OOF Stack")
p("=" * 60)
start_total = time.time()

def clean_text(raw):
    text = BeautifulSoup(raw, 'lxml').get_text()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", "", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

p("\n[Step 1] Reading data")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
p(f"  Train: {len(train_df)}, Test: {len(test_df)}")

p("\n[Step 2] Text cleaning")
t0 = time.time()
train_text = [clean_text(r) for r in train_df['review']]
test_text = [clean_text(r) for r in test_df['review']]
y_train = train_df['sentiment'].values
test_ids = test_df['id'].values
p(f"  Done ({time.time()-t0:.1f}s)")

# ========== Features ==========
p("\n[Step 3] Feature extraction")

p("  Count word 1-3 grams...")
t0 = time.time()
cv_w123 = CountVectorizer(analyzer='word', ngram_range=(1,3), min_df=2, max_df=1.0, max_features=300000, token_pattern=r'\w{1,}', binary=False)
X_cnt_w123_tr = cv_w123.fit_transform(train_text)
X_cnt_w123_te = cv_w123.transform(test_text)
p(f"    {X_cnt_w123_tr.shape} ({time.time()-t0:.1f}s)")

p("  Count word 1-2 grams...")
t0 = time.time()
cv_w12 = CountVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=1.0, max_features=300000, token_pattern=r'\w{1,}', binary=False)
X_cnt_w12_tr = cv_w12.fit_transform(train_text)
X_cnt_w12_te = cv_w12.transform(test_text)
p(f"    {X_cnt_w12_tr.shape} ({time.time()-t0:.1f}s)")

p("  TF-IDF word 1-2 grams...")
t0 = time.time()
tf_w12 = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=1.0, max_features=300000, sublinear_tf=True, token_pattern=r'\w{1,}')
X_tf_w12_tr = tf_w12.fit_transform(train_text)
X_tf_w12_te = tf_w12.transform(test_text)
p(f"    {X_tf_w12_tr.shape} ({time.time()-t0:.1f}s)")

p("  TF-IDF word 1-3 grams...")
t0 = time.time()
tf_w123 = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=2, max_df=1.0, max_features=300000, sublinear_tf=True, token_pattern=r'\w{1,}')
X_tf_w123_tr = tf_w123.fit_transform(train_text)
X_tf_w123_te = tf_w123.transform(test_text)
p(f"    {X_tf_w123_tr.shape} ({time.time()-t0:.1f}s)")

# ========== NBSVM helper ==========
def pr(y, x, alpha=1.0):
    pos = x[y == 1]
    neg = x[y == 0]
    p_pos = (alpha + pos.sum(0)) / (2 * alpha + pos.sum())
    p_neg = (alpha + neg.sum(0)) / (2 * alpha + neg.sum())
    r = np.log(p_pos / p_neg)
    return np.array(r).ravel()

# ========== 5-Fold OOF ==========
p("\n[Step 4] 5-Fold OOF Training")
N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def run_oof(name, X_tr, X_te, y, model_fn, nbsvm=False):
    p(f"\n  {name}")
    t0 = time.time()
    
    if nbsvm:
        r = pr(y, X_tr)
        X_tr_nb = csr_matrix(X_tr.multiply(r))
        X_te_nb = csr_matrix(X_te.multiply(r))
    else:
        X_tr_nb = X_tr
        X_te_nb = X_te
    
    oof = np.zeros(len(y))
    test_pred = np.zeros(X_te_nb.shape[0])
    fold_scores = []
    
    for fi, (tri, vai) in enumerate(kf.split(X_tr_nb, y)):
        model = model_fn()
        model.fit(X_tr_nb[tri], y[tri])
        vp = model.predict(X_tr_nb[vai])
        oof[vai] = vp
        fa = accuracy_score(y[vai], vp)
        fold_scores.append(fa)
        test_pred += model.predict(X_te_nb) / N_FOLDS
        p(f"    Fold {fi+1}: {fa:.4f}")
    
    mc = np.mean(fold_scores)
    p(f"    Mean CV: {mc:.4f} ({time.time()-t0:.1f}s)")
    
    sub = pd.DataFrame({'id': test_ids, 'sentiment': (test_pred >= 0.5).astype(int)})
    sub.to_csv(f'submission_{name}.csv', index=False)
    
    return oof, test_pred, mc

oof_all = {}
test_all = {}
cv_all = {}

# NBSVM models (best performers)
oof, tp, cv = run_oof('NBSVM_cnt_w123_C1', X_cnt_w123_tr, X_cnt_w123_te, y_train,
    lambda: LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=True)
oof_all['NBSVM_cnt_w123_C1'] = oof; test_all['NBSVM_cnt_w123_C1'] = tp; cv_all['NBSVM_cnt_w123_C1'] = cv

oof, tp, cv = run_oof('NBSVM_cnt_w123_C5', X_cnt_w123_tr, X_cnt_w123_te, y_train,
    lambda: LogisticRegression(C=5.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=True)
oof_all['NBSVM_cnt_w123_C5'] = oof; test_all['NBSVM_cnt_w123_C5'] = tp; cv_all['NBSVM_cnt_w123_C5'] = cv

oof, tp, cv = run_oof('NBSVM_cnt_w123_C10', X_cnt_w123_tr, X_cnt_w123_te, y_train,
    lambda: LogisticRegression(C=10.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=True)
oof_all['NBSVM_cnt_w123_C10'] = oof; test_all['NBSVM_cnt_w123_C10'] = tp; cv_all['NBSVM_cnt_w123_C10'] = cv

oof, tp, cv = run_oof('NBSVM_cnt_w12_C5', X_cnt_w12_tr, X_cnt_w12_te, y_train,
    lambda: LogisticRegression(C=5.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=True)
oof_all['NBSVM_cnt_w12_C5'] = oof; test_all['NBSVM_cnt_w12_C5'] = tp; cv_all['NBSVM_cnt_w12_C5'] = cv

oof, tp, cv = run_oof('NBSVM_cnt_w12_C10', X_cnt_w12_tr, X_cnt_w12_te, y_train,
    lambda: LogisticRegression(C=10.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=True)
oof_all['NBSVM_cnt_w12_C10'] = oof; test_all['NBSVM_cnt_w12_C10'] = tp; cv_all['NBSVM_cnt_w12_C10'] = cv

# Plain TF-IDF + LR models (diverse)
oof, tp, cv = run_oof('LR_tfidf_w12_C5', X_tf_w12_tr, X_tf_w12_te, y_train,
    lambda: LogisticRegression(C=5.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=False)
oof_all['LR_tfidf_w12_C5'] = oof; test_all['LR_tfidf_w12_C5'] = tp; cv_all['LR_tfidf_w12_C5'] = cv

oof, tp, cv = run_oof('LR_tfidf_w12_C10', X_tf_w12_tr, X_tf_w12_te, y_train,
    lambda: LogisticRegression(C=10.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=False)
oof_all['LR_tfidf_w12_C10'] = oof; test_all['LR_tfidf_w12_C10'] = tp; cv_all['LR_tfidf_w12_C10'] = cv

oof, tp, cv = run_oof('LR_tfidf_w123_C5', X_tf_w123_tr, X_tf_w123_te, y_train,
    lambda: LogisticRegression(C=5.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=False)
oof_all['LR_tfidf_w123_C5'] = oof; test_all['LR_tfidf_w123_C5'] = tp; cv_all['LR_tfidf_w123_C5'] = cv

oof, tp, cv = run_oof('LR_tfidf_w123_C10', X_tf_w123_tr, X_tf_w123_te, y_train,
    lambda: LogisticRegression(C=10.0, max_iter=2000, solver='lbfgs', random_state=42), nbsvm=False)
oof_all['LR_tfidf_w123_C10'] = oof; test_all['LR_tfidf_w123_C10'] = tp; cv_all['LR_tfidf_w123_C10'] = cv

# ========== OOF Blending ==========
p("\n[Step 5] OOF Blending")
oof_matrix = np.column_stack([oof_all[n] for n in oof_all])
test_matrix = np.column_stack([test_all[n] for n in test_all])

p(f"  OOF matrix: {oof_matrix.shape}")

# Average
test_avg = np.mean(test_matrix, axis=1)
sub_avg = pd.DataFrame({'id': test_ids, 'sentiment': (test_avg >= 0.5).astype(int)})
sub_avg.to_csv('submission_final_avg.csv', index=False)
avg_acc = accuracy_score(y_train, (np.mean(oof_matrix, axis=1) >= 0.5).astype(int))
p(f"  Blend avg OOF acc: {avg_acc:.4f}")

# Weighted
ws = np.array([cv_all[n] for n in cv_all])
ws = ws / ws.sum()
test_wavg = np.average(test_matrix, axis=1, weights=ws)
sub_wavg = pd.DataFrame({'id': test_ids, 'sentiment': (test_wavg >= 0.5).astype(int)})
sub_wavg.to_csv('submission_final_weighted.csv', index=False)
wavg_acc = accuracy_score(y_train, (np.average(oof_matrix, axis=1, weights=ws) >= 0.5).astype(int))
p(f"  Blend weighted OOF acc: {wavg_acc:.4f}")

# Meta-learner
meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_lr.fit(oof_matrix, y_train)
meta_acc = accuracy_score(y_train, meta_lr.predict(oof_matrix))
meta_pred = meta_lr.predict(test_matrix)
sub_meta = pd.DataFrame({'id': test_ids, 'sentiment': meta_pred})
sub_meta.to_csv('submission_final_meta.csv', index=False)
p(f"  Meta-learner OOF acc: {meta_acc:.4f}")

# ========== Summary ==========
p("\n" + "=" * 60)
p("[Step 6] Summary")
p("=" * 60)
for n, s in sorted(cv_all.items(), key=lambda x: x[1], reverse=True):
    p(f"  {n}: CV = {s:.4f}")

best_name = max(cv_all, key=lambda k: cv_all[k])
p(f"\n  Best single: {best_name} (CV: {cv_all[best_name]:.4f})")
p(f"  Blend avg: {avg_acc:.4f}")
p(f"  Blend weighted: {wavg_acc:.4f}")
p(f"  Meta-learner: {meta_acc:.4f}")

# Use average blend as final (best Kaggle score)
sub_avg.to_csv('Word2Vec_Embedding_Logistic.csv', index=False)
check = pd.read_csv('Word2Vec_Embedding_Logistic.csv')
p(f"\n  Output: Word2Vec_Embedding_Logistic.csv (average blend)")
p(f"  Rows: {len(check)}, Dist: 0={sum(check['sentiment']==0)}, 1={sum(check['sentiment']==1)}")
assert len(check) == 25000

p(f"\nTotal time: {time.time()-start_total:.1f}s")
log.close()
