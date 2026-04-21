import pandas as pd
import re
import numpy as np
import os
import time
import gc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix

log = open('optimized_pipeline_log.txt', 'w', encoding='utf-8')
def p(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()

p("=" * 60)
p("Optimized Pipeline v5: TF-IDF + NBSVM + Stack (memory-safe)")
p("Key fix: output PROBABILITIES not hard labels!")
p("=" * 60)
start_total = time.time()

def clean_text(raw):
    text = re.sub(r'<[^>]+>', ' ', raw)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = " ".join(text.lower().split())
    return text

p("\n[Step 1] Reading data")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
p(f"  Train: {len(train_df)}, Test: {len(test_df)}")

p("\n[Step 2] Text cleaning")
t0 = time.time()
train_text = [clean_text(r) for r in train_df['review']]
test_text = [clean_text(r) for r in test_df['review']]
y_train = train_df['sentiment'].values.astype(int)
test_ids = test_df['id'].values
p(f"  Done ({time.time()-t0:.1f}s)")

p("\n[Step 3] Feature extraction")

p("  TF-IDF word (1,2)...")
t0 = time.time()
tf_word12 = TfidfVectorizer(
    analyzer='word', ngram_range=(1,2), min_df=3, max_df=0.95,
    sublinear_tf=True, strip_accents='unicode', max_features=80000,
    dtype=np.float32
)
X_tf_word12_tr = tf_word12.fit_transform(train_text)
X_tf_word12_te = tf_word12.transform(test_text)
p(f"    {X_tf_word12_tr.shape} ({time.time()-t0:.1f}s)")

p("  TF-IDF word (1,3)...")
t0 = time.time()
tf_word13 = TfidfVectorizer(
    analyzer='word', ngram_range=(1,3), min_df=3, max_df=0.95,
    sublinear_tf=True, strip_accents='unicode', max_features=80000,
    dtype=np.float32
)
X_tf_word13_tr = tf_word13.fit_transform(train_text)
X_tf_word13_te = tf_word13.transform(test_text)
p(f"    {X_tf_word13_tr.shape} ({time.time()-t0:.1f}s)")

p("  Count word (1,2) binary for NBSVM...")
t0 = time.time()
cv_word12 = CountVectorizer(
    analyzer='word', ngram_range=(1,2), min_df=3, max_df=0.95,
    strip_accents='unicode', max_features=80000, binary=True
)
X_cv_word12_tr = cv_word12.fit_transform(train_text)
X_cv_word12_te = cv_word12.transform(test_text)
p(f"    {X_cv_word12_tr.shape} ({time.time()-t0:.1f}s)")

p("  Count word (1,3) binary for NBSVM...")
t0 = time.time()
cv_word13 = CountVectorizer(
    analyzer='word', ngram_range=(1,3), min_df=3, max_df=0.95,
    strip_accents='unicode', max_features=80000, binary=True
)
X_cv_word13_tr = cv_word13.fit_transform(train_text)
X_cv_word13_te = cv_word13.transform(test_text)
p(f"    {X_cv_word13_tr.shape} ({time.time()-t0:.1f}s)")

del tf_word12, tf_word13, cv_word12, cv_word13, train_text, test_text
gc.collect()

def pr(y, x, alpha=1.0):
    pos = x[y == 1]
    neg = x[y == 0]
    p_pos = (alpha + np.asarray(pos.sum(0)).ravel()) / (2 * alpha + np.asarray(pos.sum()))
    p_neg = (alpha + np.asarray(neg.sum(0)).ravel()) / (2 * alpha + np.asarray(neg.sum()))
    r = np.log(p_pos / p_neg)
    return r

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def score_1d(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        return sigmoid(np.asarray(model.decision_function(X), dtype=np.float64))
    return sigmoid(np.asarray(model.predict(X), dtype=np.float64))

N_FOLDS = 5
SEED = 42
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def run_oof_proba(name, X_tr, X_te, y, model_fn, nbsvm=False):
    p(f"\n  {name}")
    t0 = time.time()

    if nbsvm:
        r = pr(y, X_tr)
        X_tr_nb = csr_matrix(X_tr.multiply(r))
        X_te_nb = csr_matrix(X_te.multiply(r))
    else:
        X_tr_nb = X_tr
        X_te_nb = X_te

    oof_proba = np.zeros(len(y))
    test_proba = np.zeros(X_te_nb.shape[0])
    fold_scores = []

    for fi, (tri, vai) in enumerate(kf.split(X_tr_nb, y)):
        model = model_fn()
        model.fit(X_tr_nb[tri], y[tri])
        vp = score_1d(model, X_tr_nb[vai])
        oof_proba[vai] = vp
        tp = score_1d(model, X_te_nb)
        test_proba += tp / N_FOLDS
        va_auc = roc_auc_score(y[vai], vp)
        fold_scores.append(va_auc)
        p(f"    Fold {fi+1}: AUC={va_auc:.5f}")
        del model
        gc.collect()

    mc = np.mean(fold_scores)
    p(f"    Mean AUC: {mc:.5f} ({time.time()-t0:.1f}s)")

    sub = pd.DataFrame({'id': test_ids, 'sentiment': test_proba})
    sub.to_csv(f'submission_{name}.csv', index=False)

    if nbsvm:
        del X_tr_nb, X_te_nb
        gc.collect()

    return oof_proba, test_proba, mc

p("\n[Step 4] 5-Fold OOF Training")

oof_all = {}
test_all = {}
auc_all = {}

p("\n  --- TF-IDF + LR models ---")

oof, tp, auc = run_oof_proba('tfidf_word12_C4', X_tf_word12_tr, X_tf_word12_te, y_train,
    lambda: LogisticRegression(C=4.0, max_iter=4000, solver='saga', random_state=SEED))
oof_all['tfidf_word12_C4'] = oof; test_all['tfidf_word12_C4'] = tp; auc_all['tfidf_word12_C4'] = auc

oof, tp, auc = run_oof_proba('tfidf_word12_C8', X_tf_word12_tr, X_tf_word12_te, y_train,
    lambda: LogisticRegression(C=8.0, max_iter=4000, solver='saga', random_state=SEED))
oof_all['tfidf_word12_C8'] = oof; test_all['tfidf_word12_C8'] = tp; auc_all['tfidf_word12_C8'] = auc

oof, tp, auc = run_oof_proba('tfidf_word13_C4', X_tf_word13_tr, X_tf_word13_te, y_train,
    lambda: LogisticRegression(C=4.0, max_iter=4000, solver='saga', random_state=SEED))
oof_all['tfidf_word13_C4'] = oof; test_all['tfidf_word13_C4'] = tp; auc_all['tfidf_word13_C4'] = auc

oof, tp, auc = run_oof_proba('tfidf_word13_C8', X_tf_word13_tr, X_tf_word13_te, y_train,
    lambda: LogisticRegression(C=8.0, max_iter=4000, solver='saga', random_state=SEED))
oof_all['tfidf_word13_C8'] = oof; test_all['tfidf_word13_C8'] = tp; auc_all['tfidf_word13_C8'] = auc

p("\n  --- NBSVM models ---")

oof, tp, auc = run_oof_proba('nbsvm_word12_sgd', X_cv_word12_tr, X_cv_word12_te, y_train,
    lambda: SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, max_iter=20, tol=1e-3, random_state=SEED),
    nbsvm=True)
oof_all['nbsvm_word12_sgd'] = oof; test_all['nbsvm_word12_sgd'] = tp; auc_all['nbsvm_word12_sgd'] = auc

oof, tp, auc = run_oof_proba('nbsvm_word13_sgd', X_cv_word13_tr, X_cv_word13_te, y_train,
    lambda: SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, max_iter=20, tol=1e-3, random_state=SEED),
    nbsvm=True)
oof_all['nbsvm_word13_sgd'] = oof; test_all['nbsvm_word13_sgd'] = tp; auc_all['nbsvm_word13_sgd'] = auc

p("\n  --- SVM models ---")

oof, tp, auc = run_oof_proba('svm_word12', X_tf_word12_tr, X_tf_word12_te, y_train,
    lambda: SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5, max_iter=30, tol=1e-3, random_state=SEED))
oof_all['svm_word12'] = oof; test_all['svm_word12'] = tp; auc_all['svm_word12'] = auc

oof, tp, auc = run_oof_proba('svm_word13', X_tf_word13_tr, X_tf_word13_te, y_train,
    lambda: SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5, max_iter=30, tol=1e-3, random_state=SEED))
oof_all['svm_word13'] = oof; test_all['svm_word13'] = tp; auc_all['svm_word13'] = auc

del X_tf_word12_tr, X_tf_word12_te, X_tf_word13_tr, X_tf_word13_te
del X_cv_word12_tr, X_cv_word12_te, X_cv_word13_tr, X_cv_word13_te
gc.collect()

p("\n[Step 5] OOF Stacking")
oof_matrix = np.column_stack([oof_all[n] for n in oof_all])
test_matrix = np.column_stack([test_all[n] for n in test_all])
p(f"  OOF matrix: {oof_matrix.shape}")

oof_auc = roc_auc_score(y_train, np.mean(oof_matrix, axis=1))
p(f"  Simple avg OOF AUC: {oof_auc:.5f}")

ws = np.array([auc_all[n] for n in auc_all])
ws = ws / ws.sum()
test_wavg = np.average(test_matrix, axis=1, weights=ws)
wavg_auc = roc_auc_score(y_train, np.average(oof_matrix, axis=1, weights=ws))
p(f"  Weighted avg OOF AUC: {wavg_auc:.5f}")

meta_lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=SEED)
meta_lr.fit(oof_matrix, y_train)
meta_oof = meta_lr.predict_proba(oof_matrix)[:, 1]
meta_auc = roc_auc_score(y_train, meta_oof)
p(f"  Meta-learner OOF AUC: {meta_auc:.5f}")

meta_test = meta_lr.predict_proba(test_matrix)[:, 1]

p("\n[Step 6] Rank Mean Blend with DistilBERT")
if os.path.exists('submission_distilbert.csv'):
    bert_sub = pd.read_csv('submission_distilbert.csv')
    bert_proba = bert_sub['sentiment'].values.astype(float)
    p(f"  DistilBERT submission found")

    def rank_mean(probs_a, probs_b):
        n = len(probs_a)
        rank_a = np.argsort(np.argsort(probs_a)) / max(1, n - 1)
        rank_b = np.argsort(np.argsort(probs_b)) / max(1, n - 1)
        return (rank_a + rank_b) / 2

    blended_ranks = rank_mean(meta_test, bert_proba)
    sub_stack_bert = pd.DataFrame({'id': test_ids, 'sentiment': blended_ranks})
    sub_stack_bert.to_csv('submission_stack_bert.csv', index=False)
    p(f"  Saved: submission_stack_bert.csv")
else:
    p("  No DistilBERT submission found, skip blend")

p("\n[Step 7] Output final submissions")

sub_avg = pd.DataFrame({'id': test_ids, 'sentiment': np.mean(test_matrix, axis=1)})
sub_avg.to_csv('submission_final_avg.csv', index=False)
p(f"  submission_final_avg.csv (simple avg, proba)")

sub_wavg_out = pd.DataFrame({'id': test_ids, 'sentiment': test_wavg})
sub_wavg_out.to_csv('submission_final_weighted.csv', index=False)
p(f"  submission_final_weighted.csv (weighted avg, proba)")

sub_meta = pd.DataFrame({'id': test_ids, 'sentiment': meta_test})
sub_meta.to_csv('submission_final_meta.csv', index=False)
p(f"  submission_final_meta.csv (meta-learner, proba)")

best_method = max(
    ('avg', oof_auc),
    ('weighted', wavg_auc),
    ('meta', meta_auc),
    key=lambda x: x[1]
)
p(f"\n  Best method: {best_method[0]} (OOF AUC: {best_method[1]:.5f})")

if best_method[0] == 'avg':
    final_proba = np.mean(test_matrix, axis=1)
elif best_method[0] == 'weighted':
    final_proba = test_wavg
else:
    final_proba = meta_test

sub_final = pd.DataFrame({'id': test_ids, 'sentiment': final_proba})
sub_final.to_csv('Word2Vec_Embedding_Logistic.csv', index=False)
check = pd.read_csv('Word2Vec_Embedding_Logistic.csv')
p(f"\n  Final output: Word2Vec_Embedding_Logistic.csv")
p(f"  Rows: {len(check)}")
p(f"  Sentiment range: [{check['sentiment'].min():.4f}, {check['sentiment'].max():.4f}]")
p(f"  Sentiment mean: {check['sentiment'].mean():.4f}")
assert len(check) == 25000

p("\n" + "=" * 60)
p("[Summary]")
p("=" * 60)
for n, s in sorted(auc_all.items(), key=lambda x: x[1], reverse=True):
    p(f"  {n}: AUC = {s:.5f}")
p(f"\n  Simple avg AUC: {oof_auc:.5f}")
p(f"  Weighted avg AUC: {wavg_auc:.5f}")
p(f"  Meta-learner AUC: {meta_auc:.5f}")
p(f"  Best: {best_method[0]} = {best_method[1]:.5f}")

p(f"\nTotal time: {time.time()-start_total:.1f}s")
log.close()
