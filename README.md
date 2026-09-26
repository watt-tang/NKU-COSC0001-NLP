# NKU-COSC0001-NLP

南开大学自然语言处理课程实验仓库，用于保存课程实验代码、实验报告、运行说明与相关结果。

## 个人信息

- 学校：南开大学
- 学院：密码与网络空间安全学院
- 专业：密码科学与技术
- 学号：2412103
- 姓名：唐健

## 实验列表

| Lab   | 实验标题     | 状态     | 目录    |
| ----- | ------------ | -------- | ------- |
| Lab 1 | 文本分类实现 | ✅ 已完成 | `lab1/` |

后续课程实验将在本仓库中继续添加。

## Lab 1：文本分类实现

本实验比较不同文本表示方法在 NYT 新闻分类任务中的效果。

主要内容包括：

- Task 1：Bag of Words
  - Binary Bag of Words
  - Word Frequency
- Task 2：Word Embedding / Word2Vec
  - Pre-trained GloVe 100d
  - Word2Vec trained on AG News
  - Word2Vec trained on NYT
- Task 3：Pre-trained Neural Model
  - `google-bert/bert-base-uncased`
  - `max_length = 64`
  - Fine-tuning 3 epochs

评价指标：

- Accuracy
- Macro-F1

## 目录说明

```text
NKU-COSC0001-NLP/
├── README.md
├── .gitignore
└── lab1/
```

> 预训练模型权重、GloVe 大文件、Embedding Cache 等可重新下载或生成的大文件不上传到 Git 仓库。

## 运行环境

主要依赖：

```text
Python 3.13
numpy
pandas
scikit-learn
gensim
matplotlib
tqdm
torch
transformers
```

安装依赖：

```bash
pip install numpy pandas scikit-learn gensim matplotlib tqdm torch transformers
```


## 说明

本仓库仅用于自然语言处理课程学习与实验记录。
