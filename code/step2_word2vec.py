import pickle
import time
from gensim.models import Word2Vec

print("Step 2: Word2Vec training - START", flush=True)
t0 = time.time()

print("Loading cleaned data...", flush=True)
with open('cleaned_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_clean = data['train']
unlabeled_clean = data['unlabeled']
all_sentences = train_clean + unlabeled_clean
print(f"  Total sentences: {len(all_sentences)}", flush=True)

print("Training Word2Vec (300d, window=10, min_count=40, epochs=10, sg=1)...", flush=True)
t1 = time.time()
model = Word2Vec(
    sentences=all_sentences,
    vector_size=300,
    window=10,
    min_count=40,
    workers=4,
    epochs=10,
    sg=1
)
print(f"  Vocab: {len(model.wv)}, Dim: {model.wv.vector_size}", flush=True)
print(f"  Training time: {time.time()-t1:.1f}s", flush=True)

print("Saving model...", flush=True)
model.save('word2vec_model.bin')
print(f"Step 2 DONE! Total: {time.time()-t0:.1f}s", flush=True)
