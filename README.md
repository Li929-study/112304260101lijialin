# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：李佳霖
- **学号**：112304260101
- **班级**：

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/c/word2vec-nlp-tutorial
- **提交日期**：2026-04-15

- **GitHub 仓库地址**：https://github.com/Li929-study/112304260101lijialin
- **GitHub README 地址**：https://github.com/Li929-study/112304260101lijialin/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：（待提交后填写）
- **Private Score**（如有）：（待提交后填写）
- **排名**（如能看到可填写）：（待提交后填写）

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`112304260101_李佳霖_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
1. 使用 BeautifulSoup 去除 HTML 标签（`lxml` 解析器）
2. 使用正则表达式 `[^a-zA-Z]` 去除所有非字母字符，替换为空格
3. 全部转为小写
4. 按空格分词
5. 去除英文停用词（NLTK stopwords），但保留否定词（`not`, `no`, `never`, `nor`），因为否定词对情感分类至关重要

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
- **自己训练 Word2Vec**，使用 labeledTrainData（25000条）+ unlabeledTrainData（49998条）共 74998 条评论作为训练语料
- 词向量维度：**300 维**
- 训练参数：`vector_size=300, window=10, min_count=40, epochs=10, sg=1`（Skip-gram 模型）
- 训练后词表大小：16343 个词
- 句子向量采用**均值 Embedding（Average Word Vector）**方法：对每条评论中所有在词表中的词向量取平均值，得到 300 维的句子特征向量

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
最终采用 **Logistic Regression（逻辑回归）** 作为分类模型。

模型参数：
- `solver='lbfgs'`
- `max_iter=1000`
- `C=1.0`
- `random_state=42`

训练集准确率：**0.8899**（88.99%）

训练集预测分布：0=12365, 1=12635  
测试集预测分布：0=12631, 1=12369（非完美均分，为真实模型输出）

---

## 7. 实验流程
请简要说明你的实验流程。

**我的实验流程：**  
1. 读取全部数据：labeledTrainData（25000条）、unlabeledTrainData（49998条）、testData（25000条）
2. 对全部文本进行预处理：去HTML标签 → 去非字母 → 转小写 → 分词 → 去停用词（保留否定词）
3. 使用 labeled + unlabeled 共 74998 条评论训练 Word2Vec 模型（300维、Skip-gram）
4. 对训练集和测试集分别生成均值 Embedding 特征（每条评论 → 300维向量）
5. 使用逻辑回归（LogisticRegression）在训练集上训练分类器
6. 在测试集上进行全量预测（25000条）
7. 生成提交文件 `Word2Vec_Embedding_Logistic.csv`（25000行）

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

**我的项目结构：**
```text
project/
├─ code/                           # 实验代码
│  ├─ step1_clean.py               # Step1: 文本清洗
│  ├─ step2_word2vec.py            # Step2: Word2Vec 训练
│  ├─ step3_embed_lr.py            # Step3: 均值Embedding + 逻辑回归 + 预测
│  ├─ word2vec_part3.py            # 完整Pipeline主脚本
│  └─ word2vec_pipeline.py         # 合并版Pipeline脚本
├─ results/                        # 实验结果
│  ├─ Word2Vec_Embedding_Logistic.csv   # 提交文件（25000行）
│  └─ run_log.txt                  # 运行日志
├─ images/                         # 截图等图片
│  └─ kaggle_score.png             # Kaggle提交截图（待添加）
├─ labeledTrainData.tsv/           # 原始训练数据（25000条）
├─ unlabeledTrainData.tsv/         # 原始无标签数据（50000条）
├─ testData.tsv/                   # 原始测试数据（25000条）
├─ sampleSubmission.csv            # 提交样例
└─ README.md                       # 实验报告
```
