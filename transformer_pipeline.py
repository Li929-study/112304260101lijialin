import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import rankdata
import traceback
import gc

log = open('transformer_log.txt', 'w', encoding='utf-8')
def p(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()

try:
    p("=" * 60)
    p("DistilBERT Fine-tuning + Rank Mean Blend")
    p("=" * 60)
    start_total = time.time()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p(f"Device: {DEVICE}")

    MODEL_NAME = './distilbert_model'
    MAX_LEN = 128
    BATCH_SIZE = 8
    EPOCHS = 2
    LR = 2e-5
    N_FOLDS = 2
    SEED = 42

    p(f"Model: {MODEL_NAME}")
    p(f"MaxLen: {MAX_LEN}, BatchSize: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}, Folds: {N_FOLDS}")

    p("\n[Step 1] Reading data")
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    p(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    train_texts = train_df['review'].tolist()
    train_labels = train_df['sentiment'].values
    test_texts = test_df['review'].tolist()
    test_ids = test_df['id'].values

    p("\n[Step 2] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    p(f"  Tokenizer loaded")

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    p("\n[Step 3] Quick model test")
    test_dataset = ReviewDataset(test_texts[:16], None, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    p(f"  Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        p(f"  Forward pass OK, logits shape: {outputs.logits.shape}")
    del model
    gc.collect()

    p("\n[Step 4] 2-Fold DistilBERT Training")

    def train_one_epoch(model, dataloader, optimizer, scheduler, device):
        model.train()
        total_loss = 0
        n_batches = len(dataloader)
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (i + 1) % 200 == 0:
                p(f"      Batch {i+1}/{n_batches}, avg_loss={total_loss/(i+1):.4f}")
        return total_loss / len(dataloader)

    def predict(model, dataloader, device):
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(len(train_labels))
    test_probs_all = np.zeros((N_FOLDS, len(test_texts)))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_texts, train_labels)):
        p(f"\n  === Fold {fold+1}/{N_FOLDS} ===")
        t0 = time.time()

        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_labels = train_labels[val_idx]

        train_dataset = ReviewDataset(fold_train_texts, fold_train_labels, tokenizer, MAX_LEN)
        val_dataset = ReviewDataset(fold_val_texts, fold_val_labels, tokenizer, MAX_LEN)
        test_dataset = ReviewDataset(test_texts, None, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        p(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        best_auc = 0
        best_val_probs = None
        best_test_probs = None

        for epoch in range(EPOCHS):
            p(f"    Epoch {epoch+1} training...")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            p(f"    Epoch {epoch+1} validating...")
            val_probs = predict(model, val_loader, DEVICE)
            val_auc = roc_auc_score(fold_val_labels, val_probs)
            val_acc = accuracy_score(fold_val_labels, (val_probs >= 0.5).astype(int))
            p(f"    Epoch {epoch+1}: loss={train_loss:.4f} val_auc={val_auc:.5f} val_acc={val_acc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_val_probs = val_probs.copy()
                p(f"    Epoch {epoch+1} predicting test...")
                best_test_probs = predict(model, test_loader, DEVICE)

        oof_probs[val_idx] = best_val_probs
        test_probs_all[fold] = best_test_probs
        fold_scores.append(best_auc)

        p(f"    Fold {fold+1} Best AUC: {best_auc:.5f} ({time.time()-t0:.1f}s)")

        del model
        gc.collect()

    oof_auc = roc_auc_score(train_labels, oof_probs)
    oof_acc = accuracy_score(train_labels, (oof_probs >= 0.5).astype(int))
    p(f"\n  OOF AUC: {oof_auc:.5f}, OOF Acc: {oof_acc:.4f}")
    p(f"  Mean Fold AUC: {np.mean(fold_scores):.5f}")

    test_probs_avg = np.mean(test_probs_all, axis=0)
    test_preds_bert = (test_probs_avg >= 0.5).astype(int)

    sub_bert = pd.DataFrame({'id': test_ids, 'sentiment': test_preds_bert})
    sub_bert.to_csv('submission_distilbert.csv', index=False)
    p(f"\n  DistilBERT submission saved: submission_distilbert.csv")
    p(f"  Test dist: 0={sum(test_preds_bert==0)}, 1={sum(test_preds_bert==1)}")

    p("\n[Step 5] Rank Mean Blend with NBSVM")

    if os.path.exists('submission_final_avg.csv'):
        nbsvm_sub = pd.read_csv('submission_final_avg.csv')
        nbsvm_preds = nbsvm_sub['sentiment'].values

        def rank_mean(probs_a, probs_b):
            rank_a = rankdata(probs_a) / len(probs_a)
            rank_b = rankdata(probs_b) / len(probs_b)
            return (rank_a + rank_b) / 2

        blended_ranks = rank_mean(test_probs_avg, nbsvm_preds.astype(float))
        blended_preds = (blended_ranks >= 0.5).astype(int)

        sub_blend = pd.DataFrame({'id': test_ids, 'sentiment': blended_preds})
        sub_blend.to_csv('submission_blend_bert_nbsvm.csv', index=False)
        p(f"  Blend saved: submission_blend_bert_nbsvm.csv")
        p(f"  Blend dist: 0={sum(blended_preds==0)}, 1={sum(blended_preds==1)}")

        sub_blend.to_csv('Word2Vec_Embedding_Logistic.csv', index=False)
        p(f"  Final output: Word2Vec_Embedding_Logistic.csv")
    else:
        p("  NBSVM submission not found, using DistilBERT only")
        sub_bert.to_csv('Word2Vec_Embedding_Logistic.csv', index=False)

    check = pd.read_csv('Word2Vec_Embedding_Logistic.csv')
    p(f"\n  Rows: {len(check)}, Dist: 0={sum(check['sentiment']==0)}, 1={sum(check['sentiment']==1)}")
    assert len(check) == 25000

    p(f"\nTotal time: {time.time()-start_total:.1f}s")

except Exception as e:
    p(f"\nFATAL ERROR: {type(e).__name__}: {e}")
    p(traceback.format_exc())

log.close()
