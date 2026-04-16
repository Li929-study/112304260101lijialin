import pandas as pd
import re
import numpy as np
import os
import time
import sys
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords


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
            print(f"  [{desc}] {counter}/{len(reviews)} reviews processed  ({time.time()-t0:.1f}s)")
    return review_feature_vecs


def main():
    start_total = time.time()
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

    # ========== Step 1: 读取全部数据 ==========
    print("=" * 60)
    print("Step 1: 读取全部数据文件")
    print("=" * 60)

    print("[1/3] 读取 labeledTrainData.tsv ...")
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    print(f"  -> labeledTrainData 行数: {len(train_df)}")
    assert len(train_df) == 25000, f"ERROR: 期望 25000, 实际 {len(train_df)}"

    print("[2/3] 读取 unlabeledTrainData.tsv ...")
    unlabeled_df = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', on_bad_lines='skip')
    print(f"  -> unlabeledTrainData 行数: {len(unlabeled_df)}")
    assert len(unlabeled_df) >= 49000, f"ERROR: 期望 ~50000, 实际 {len(unlabeled_df)}"

    print("[3/3] 读取 testData.tsv ...")
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    print(f"  -> testData 行数: {len(test_df)}")
    assert len(test_df) == 25000, f"ERROR: 期望 25000, 实际 {len(test_df)}"

    # ========== Step 2: 文本清洗 ==========
    print("\n" + "=" * 60)
    print("Step 2: 文本清洗（全部数据）")
    print("=" * 60)

    t0 = time.time()
    print("[1/3] 清洗 labeledTrainData (25000 条) ...")
    train_clean = []
    total = len(train_df)
    for i, review in enumerate(train_df['review']):
        train_clean.append(review_to_words(review))
        if (i + 1) % 5000 == 0 or i == total - 1:
            print(f"  已清洗 {i+1}/{total} 条训练评论  ({time.time()-t0:.1f}s)")
    print(f"  训练集清洗完成: {len(train_clean)} 条")

    t0 = time.time()
    print("[2/3] 清洗 unlabeledTrainData (50000 条) ...")
    unlabeled_clean = []
    total = len(unlabeled_df)
    for i, review in enumerate(unlabeled_df['review']):
        unlabeled_clean.append(review_to_words(review))
        if (i + 1) % 5000 == 0 or i == total - 1:
            print(f"  已清洗 {i+1}/{total} 条无标签评论  ({time.time()-t0:.1f}s)")
    print(f"  无标签集清洗完成: {len(unlabeled_clean)} 条")

    t0 = time.time()
    print("[3/3] 清洗 testData (25000 条) ...")
    test_clean = []
    total = len(test_df)
    for i, review in enumerate(test_df['review']):
        test_clean.append(review_to_words(review))
        if (i + 1) % 5000 == 0 or i == total - 1:
            print(f"  已清洗 {i+1}/{total} 条测试评论  ({time.time()-t0:.1f}s)")
    print(f"  测试集清洗完成: {len(test_clean)} 条")

    # ========== Step 3: Word2Vec 训练 ==========
    print("\n" + "=" * 60)
    print("Step 3: 训练 Word2Vec（使用 labeled + unlabeled 全部数据）")
    print("=" * 60)

    all_sentences = train_clean + unlabeled_clean
    print(f"  总训练句子数: {len(all_sentences)}")

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
    print(f"  Word2Vec 词表大小: {len(model.wv)}")
    print(f"  Word2Vec 向量维度: {model.wv.vector_size}")
    print(f"  Word2Vec 训练耗时: {time.time()-t0:.1f}s")

    # ========== Step 4: 评论均值 Embedding ==========
    print("\n" + "=" * 60)
    print("Step 4: 生成评论均值 Embedding 特征")
    print("=" * 60)

    num_features = 300

    print("[1/2] 生成训练集均值 Embedding (25000 条) ...")
    t0 = time.time()
    X_train = get_avg_feature_vecs(train_clean, model, num_features, desc="训练集")
    y_train = train_df['sentiment'].values
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train distribution: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print(f"  训练集 Embedding 耗时: {time.time()-t0:.1f}s")

    print("[2/2] 生成测试集均值 Embedding (25000 条) ...")
    t0 = time.time()
    X_test = get_avg_feature_vecs(test_clean, model, num_features, desc="测试集")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  测试集 Embedding 耗时: {time.time()-t0:.1f}s")

    # ========== Step 5: 逻辑回归分类 ==========
    print("\n" + "=" * 60)
    print("Step 5: 逻辑回归分类训练")
    print("=" * 60)

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs')
    lr.fit(X_train, y_train)
    print(f"  逻辑回归训练耗时: {time.time()-t0:.1f}s")

    train_pred = lr.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  训练集预测分布: 0={sum(train_pred==0)}, 1={sum(train_pred==1)}")

    # ========== Step 6: 全量预测 ==========
    print("\n" + "=" * 60)
    print("Step 6: 全量预测测试集 (25000 条)")
    print("=" * 60)

    t0 = time.time()
    predictions = lr.predict(X_test)
    print(f"  预测耗时: {time.time()-t0:.1f}s")
    print(f"  预测结果 shape: {predictions.shape}")
    print(f"  预测分布: 0={sum(predictions==0)}, 1={sum(predictions==1)}")
    assert len(predictions) == 25000, f"ERROR: 期望 25000 条预测, 实际 {len(predictions)}"

    # ========== Step 7: 生成提交文件 ==========
    print("\n" + "=" * 60)
    print("Step 7: 生成 Word2Vec_Embedding_Logistic.csv")
    print("=" * 60)

    submission = pd.DataFrame({'id': test_df['id'], 'sentiment': predictions})
    submission.to_csv('Word2Vec_Embedding_Logistic.csv', index=False)
    print(f"  CSV 已写入: Word2Vec_Embedding_Logistic.csv")

    # ========== Step 8: 验证提交文件 ==========
    print("\n" + "=" * 60)
    print("Step 8: 验证提交文件")
    print("=" * 60)

    assert os.path.exists('Word2Vec_Embedding_Logistic.csv'), "ERROR: CSV 文件不存在!"
    check = pd.read_csv('Word2Vec_Embedding_Logistic.csv')
    print(f"  CSV 行数: {len(check)}")
    print(f"  CSV 列: {list(check.columns)}")
    print(f"  情感分布: 0={sum(check['sentiment']==0)}, 1={sum(check['sentiment']==1)}")
    print(f"  文件大小: {os.path.getsize('Word2Vec_Embedding_Logistic.csv')} bytes")
    print(f"\n  前 10 行:")
    print(check.head(10))
    assert len(check) == 25000, f"ERROR: 期望 25000 行, 实际 {len(check)}"

    print("\n" + "=" * 60)
    print(f"全部完成! 总耗时: {time.time()-start_total:.1f}s")
    print(f"输出文件: Word2Vec_Embedding_Logistic.csv (25000 行)")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        with open('error_log.txt', 'w') as f:
            import traceback
            f.write(traceback.format_exc())
        print(f"ERROR: {e}")
        raise
