import pickle
import time
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Step 3: Embedding + LR + Predict - START", flush=True)
t0 = time.time()

print("Loading cleaned data...", flush=True)
with open('cleaned_data.pkl', 'rb') as f:
    data = pickle.load(f)
train_clean = data['train']
test_clean = data['test']
print(f"  Train: {len(train_clean)}, Test: {len(test_clean)}", flush=True)

print("Loading labels and IDs from TSV...", flush=True)
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
y_train = train_df['sentiment'].values
test_ids = test_df['id'].values
print(f"  y_train: {y_train.shape}, test_ids: {test_ids.shape}", flush=True)

print("Loading Word2Vec model...", flush=True)
model = Word2Vec.load('word2vec_model.bin')
print(f"  Vocab: {len(model.wv)}, Dim: {model.wv.vector_size}", flush=True)

def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros(num_features, dtype=np.float32)
    n_words = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def get_avg_feature_vecs(reviews, model, num_features, desc=""):
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype=np.float32)
    counter = 0
    t1 = time.time()
    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter += 1
        if counter % 5000 == 0 or counter == len(reviews):
            print(f"  [{desc}] {counter}/{len(reviews)} ({time.time()-t1:.1f}s)", flush=True)
    return review_feature_vecs

print("Generating train Embedding...", flush=True)
X_train = get_avg_feature_vecs(train_clean, model, 300, desc="Train")
print(f"  X_train: {X_train.shape}", flush=True)

print("Generating test Embedding...", flush=True)
X_test = get_avg_feature_vecs(test_clean, model, 300, desc="Test")
print(f"  X_test: {X_test.shape}", flush=True)

print("Training Logistic Regression...", flush=True)
t1 = time.time()
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs')
lr.fit(X_train, y_train)
print(f"  LR training: {time.time()-t1:.1f}s", flush=True)

train_pred = lr.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"  Train acc: {train_acc:.4f}", flush=True)
print(f"  Train pred: 0={sum(train_pred==0)}, 1={sum(train_pred==1)}", flush=True)

print("Predicting test set...", flush=True)
predictions = lr.predict(X_test)
print(f"  Pred dist: 0={sum(predictions==0)}, 1={sum(predictions==1)}", flush=True)

print("Generating Word2Vec_Embedding_Logistic.csv...", flush=True)
submission = pd.DataFrame({'id': test_ids, 'sentiment': predictions})
submission.to_csv('Word2Vec_Embedding_Logistic.csv', index=False)

print("Verifying...", flush=True)
check = pd.read_csv('Word2Vec_Embedding_Logistic.csv')
print(f"  Rows: {len(check)}", flush=True)
print(f"  Cols: {list(check.columns)}", flush=True)
print(f"  Dist: 0={sum(check['sentiment']==0)}, 1={sum(check['sentiment']==1)}", flush=True)
print(f"  Size: {os.path.getsize('Word2Vec_Embedding_Logistic.csv')} bytes", flush=True)
print(f"  First 5:\n{check.head()}", flush=True)
assert len(check) == 25000, f"ERROR: expected 25000, got {len(check)}"

print(f"\nStep 3 DONE! Total: {time.time()-t0:.1f}s", flush=True)
print(f"Output: Word2Vec_Embedding_Logistic.csv (25000 rows)", flush=True)
