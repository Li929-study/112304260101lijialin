import pandas as pd
import re
import time
import pickle
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

print("Step 1: Text cleaning - START", flush=True)
t0 = time.time()

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

print("Reading data...", flush=True)
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
print(f"  Train: {len(train_df)}", flush=True)

unlabeled_df = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', on_bad_lines='skip')
print(f"  Unlabeled: {len(unlabeled_df)}", flush=True)

test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
print(f"  Test: {len(test_df)}", flush=True)

print("Cleaning train...", flush=True)
t1 = time.time()
train_clean = [review_to_words(r) for r in train_df['review']]
print(f"  Done: {len(train_clean)} ({time.time()-t1:.1f}s)", flush=True)

print("Cleaning unlabeled...", flush=True)
t1 = time.time()
unlabeled_clean = [review_to_words(r) for r in unlabeled_df['review']]
print(f"  Done: {len(unlabeled_clean)} ({time.time()-t1:.1f}s)", flush=True)

print("Cleaning test...", flush=True)
t1 = time.time()
test_clean = [review_to_words(r) for r in test_df['review']]
print(f"  Done: {len(test_clean)} ({time.time()-t1:.1f}s)", flush=True)

print("Saving cleaned data...", flush=True)
with open('cleaned_data.pkl', 'wb') as f:
    pickle.dump({
        'train': train_clean,
        'unlabeled': unlabeled_clean,
        'test': test_clean,
        'train_labels': train_df['sentiment'].values,
        'test_ids': test_df['id'].values
    }, f)

print(f"Step 1 DONE! Total: {time.time()-t0:.1f}s", flush=True)
