import os
import sys
import time
import pickle

log_file = open('pipeline_log.txt', 'w', encoding='utf-8')

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

log("All imports successful!")

def review_to_words(raw_review):
    soup = BeautifulSoup(raw_review, 'lxml')
    review_text = soup.get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    lowercase_text = letters_only.lower()
    words = lowercase_text.split()
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never', 'nor'}
    meaningful_words = [w for w in words if w not in stop_words or w in negation_words]
    return meaningful_words


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
    t0 = time.time()
    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter += 1
        if counter % 5000 == 0 or counter == len(reviews):
            log(f"  [{desc}] {counter}/{len(reviews)} reviews  ({time.time()-t0:.1f}s)")
    return review_feature_vecs


start_total = time.time()

# ========== Step 1: Read data ==========
log("=" * 60)
log("Step 1: Reading all data")
log("=" * 60)

train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
log(f"  labeledTrainData: {len(train_df)}")

unlabeled_df = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', on_bad_lines='skip')
log(f"  unlabeledTrainData: {len(unlabeled_df)}")

test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
log(f"  testData: {len(test_df)}")

# ========== Step 2: Clean text ==========
log("\n" + "=" * 60)
log("Step 2: Text cleaning")
log("=" * 60)

t0 = time.time()
train_clean = [review_to_words(r) for r in train_df['review']]
log(f"  Train cleaned: {len(train_clean)} ({time.time()-t0:.1f}s)")

t0 = time.time()
unlabeled_clean = [review_to_words(r) for r in unlabeled_df['review']]
log(f"  Unlabeled cleaned: {len(unlabeled_clean)} ({time.time()-t0:.1f}s)")

t0 = time.time()
test_clean = [review_to_words(r) for r in test_df['review']]
log(f"  Test cleaned: {len(test_clean)} ({time.time()-t0:.1f}s)")

# Save cleaned data
log("  Saving cleaned data...")
with open('cleaned_data.pkl', 'wb') as f:
    pickle.dump({'train': train_clean, 'unlabeled': unlabeled_clean, 'test': test_clean}, f)
log("  Cleaned data saved!")

# ========== Step 3: Train Word2Vec ==========
log("\n" + "=" * 60)
log("Step 3: Training Word2Vec")
log("=" * 60)

all_sentences = train_clean + unlabeled_clean
log(f"  Total sentences: {len(all_sentences)}")

t0 = time.time()
model = Word2Vec(
    sentences=all_sentences,
    vector_size=300,
    window=10,
    min_count=40,
    workers=4,
    epochs=10,
    sg=1
)
log(f"  Vocab: {len(model.wv)}, Dim: {model.wv.vector_size}")
log(f"  W2V training: {time.time()-t0:.1f}s")

model.save('word2vec_model.bin')
log("  Model saved!")

# ========== Step 4: Average Embedding ==========
log("\n" + "=" * 60)
log("Step 4: Average Embedding")
log("=" * 60)

X_train = get_avg_feature_vecs(train_clean, model, 300, desc="Train")
y_train = train_df['sentiment'].values
log(f"  X_train: {X_train.shape}, y: 0={sum(y_train==0)}, 1={sum(y_train==1)}")

X_test = get_avg_feature_vecs(test_clean, model, 300, desc="Test")
log(f"  X_test: {X_test.shape}")

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
log("  Features saved!")

# ========== Step 5: Logistic Regression ==========
log("\n" + "=" * 60)
log("Step 5: Logistic Regression")
log("=" * 60)

t0 = time.time()
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs')
lr.fit(X_train, y_train)
log(f"  LR training: {time.time()-t0:.1f}s")

train_pred = lr.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
log(f"  Train acc: {train_acc:.4f}")
log(f"  Train pred: 0={sum(train_pred==0)}, 1={sum(train_pred==1)}")

# ========== Step 6: Predict ==========
log("\n" + "=" * 60)
log("Step 6: Predict test set")
log("=" * 60)

predictions = lr.predict(X_test)
log(f"  Pred dist: 0={sum(predictions==0)}, 1={sum(predictions==1)}")

# ========== Step 7: Submit ==========
log("\n" + "=" * 60)
log("Step 7: Generate Word2Vec_Embedding_Logistic.csv")
log("=" * 60)

submission = pd.DataFrame({'id': test_df['id'], 'sentiment': predictions})
submission.to_csv('Word2Vec_Embedding_Logistic.csv', index=False)

# ========== Step 8: Verify ==========
log("\n" + "=" * 60)
log("Step 8: Verify")
log("=" * 60)

check = pd.read_csv('Word2Vec_Embedding_Logistic.csv')
log(f"  Rows: {len(check)}")
log(f"  Cols: {list(check.columns)}")
log(f"  Dist: 0={sum(check['sentiment']==0)}, 1={sum(check['sentiment']==1)}")
log(f"  Size: {os.path.getsize('Word2Vec_Embedding_Logistic.csv')} bytes")
log(f"  First 10:\n{check.head(10)}")
assert len(check) == 25000

log(f"\nALL DONE! Total: {time.time()-start_total:.1f}s")
log_file.close()
