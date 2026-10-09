# -*- coding: utf-8 -*-

"""
NLP Lab 1 - Task 2
==================

实验目标：
1. 使用 NYT 训练集训练 Word2Vec 100维词向量
2. 使用 AG 外部语料训练 Word2Vec 100维词向量
3. 可选：使用 Stanford GloVe 100维预训练词向量
4. 使用 Mean Pooling（平均池化）将一篇新闻表示为 100维文档向量
5. 使用 Logistic Regression（逻辑回归）进行 NYT 新闻分类
6. 比较不同 Representation（表征）的：
   - Accuracy
   - Macro-F1
   - OOV Rate
   - Confusion Matrix
   - Bad Cases

注意：
- NYT 的 Train / Validation / Test 划分与 Task 1 保持一致：8 : 1 : 1
- random_state = 42
- NYT Word2Vec 只使用 NYT Train 训练，避免 Test 信息泄漏
- AG 没有标签也没关系，只使用其中的文本训练 Word2Vec
"""

from pathlib import Path
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. 全局配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NYT_PATH = BASE_DIR / "nyt.csv"
AG_PATH = BASE_DIR / "ag.csv"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "task2_results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# GloVe 文件
GLOVE_PATH = BASE_DIR / "glove" / "glove.6B.100d.txt"

RANDOM_STATE = 42

# Word2Vec 参数
VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_COUNT = 2
EPOCHS = 10

# CPU 核数
CPU_COUNT = os.cpu_count() or 4

# 留一个核心给系统
WORKERS = max(1, CPU_COUNT - 1)


# ============================================================
# 1. Tokenizer（分词器）
# ============================================================

def tokenize(text):
    """
    简单英文分词。

    例如：
    "The company earned $10 million."
    ->
    ["the", "company", "earned", "million"]

    这里统一转小写，并保留英文单词。
    """

    if pd.isna(text):
        return []

    text = str(text).lower()

    tokens = re.findall(
        r"\b[a-z]+\b",
        text
    )

    return tokens


# ============================================================
# 2. 读取 NYT
# ============================================================

def load_nyt():
    print("=" * 75)
    print("1. 读取 NYT 数据集")
    print("=" * 75)

    if not NYT_PATH.exists():
        raise FileNotFoundError(
            f"找不到：{NYT_PATH}"
        )

    df = pd.read_csv(NYT_PATH)

    print("\nNYT 列名：")
    print(list(df.columns))

    # 大小写不敏感寻找 text / label
    column_map = {
        str(col).lower(): col
        for col in df.columns
    }

    if "text" not in column_map:
        raise ValueError(
            "NYT 数据中找不到 text 列。"
        )

    if "label" not in column_map:
        raise ValueError(
            "NYT 数据中找不到 label 列。"
        )

    text_col = column_map["text"]
    label_col = column_map["label"]

    df = df[
        [text_col, label_col]
    ].copy()

    df.columns = [
        "text",
        "label",
    ]

    df = df.dropna(
        subset=["text", "label"]
    )

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    df = df[
        df["text"].str.strip() != ""
    ]

    print(f"\nNYT 样本数：{len(df)}")

    print("\n类别数量：")
    print(df["label"].value_counts())

    return df


# ============================================================
# 3. 自动读取 AG 文本
# ============================================================

def load_ag():
    """
    AG 数据可能有不同格式。

    程序会尝试自动找到：
    text / content / description / article / body 等字段。

    如果有 title + description，则自动合并。
    """

    print("\n")
    print("=" * 75)
    print("2. 读取 AG 数据集")
    print("=" * 75)

    if not AG_PATH.exists():
        raise FileNotFoundError(
            f"找不到：{AG_PATH}"
        )

    df = pd.read_csv(
        AG_PATH,
        low_memory=False
    )

    print("\nAG 原始列名：")
    print(list(df.columns))

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    # --------------------------------------------------------
    # 情况 1：存在 text
    # --------------------------------------------------------

    candidates = [
        "text",
        "content",
        "article",
        "body",
        "news",
        "description",
    ]

    for name in candidates:

        if name in lower_map:

            col = lower_map[name]

            print(
                f"\n自动使用 AG 文本列：{col}"
            )

            texts = df[col].fillna("").astype(str)

            texts = texts[
                texts.str.strip() != ""
            ]

            print(
                f"AG 有效文本数：{len(texts)}"
            )

            return texts.tolist()

    # --------------------------------------------------------
    # 情况 2：title + description
    # --------------------------------------------------------

    if (
        "title" in lower_map
        and
        "description" in lower_map
    ):
        title_col = lower_map["title"]
        desc_col = lower_map["description"]

        print(
            "\n检测到 title + description，自动合并。"
        )

        texts = (
            df[title_col].fillna("").astype(str)
            + " "
            + df[desc_col].fillna("").astype(str)
        )

        texts = texts[
            texts.str.strip() != ""
        ]

        print(
            f"AG 有效文本数：{len(texts)}"
        )

        return texts.tolist()

    # --------------------------------------------------------
    # 情况 3：
    # 自动合并所有字符串列
    # --------------------------------------------------------

    print(
        "\n没有检测到标准文本列，"
        "尝试自动合并字符串列。"
    )

    object_columns = []

    for col in df.columns:

        col_lower = str(col).lower()

        # 排除明显的标签/ID列
        if col_lower in {
            "label",
            "class",
            "class index",
            "target",
            "id",
            "index",
        }:
            continue

        if (
            df[col].dtype == "object"
            or
            pd.api.types.is_string_dtype(df[col])
        ):
            object_columns.append(col)

    if len(object_columns) == 0:
        raise ValueError(
            "无法自动找到 AG 的文本列。\n"
            f"AG 当前列：{list(df.columns)}"
        )

    print(
        f"自动合并这些文本列：{object_columns}"
    )

    texts = (
        df[object_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    texts = texts[
        texts.str.strip() != ""
    ]

    print(
        f"AG 有效文本数：{len(texts)}"
    )

    return texts.tolist()


# ============================================================
# 4. 8 : 1 : 1 划分 NYT
# ============================================================

def split_nyt(df):

    print("\n")
    print("=" * 75)
    print("3. NYT 划分 Train / Validation / Test = 8 : 1 : 1")
    print("=" * 75)

    X = df["text"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"\nTrain      : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Test       : {len(X_test)}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# 5. 批量分词
# ============================================================

def tokenize_corpus(texts, name):
    print(
        f"\n正在对 {name} 进行分词..."
    )

    start = time.time()

    corpus = [
        tokenize(text)
        for text in texts
    ]

    elapsed = time.time() - start

    total_tokens = sum(
        len(tokens)
        for tokens in corpus
    )

    print(
        f"{name} 分词完成。"
    )

    print(
        f"文档数：{len(corpus):,}"
    )

    print(
        f"Token 总数：{total_tokens:,}"
    )

    print(
        f"耗时：{elapsed:.2f} 秒"
    )

    return corpus


# ============================================================
# 6. 训练 Word2Vec
# ============================================================

def train_word2vec(
    corpus,
    model_name,
):
    print("\n")
    print("=" * 75)
    print(f"训练 Word2Vec：{model_name}")
    print("=" * 75)

    print(
        f"""
参数：
vector_size = {VECTOR_SIZE}
window      = {WINDOW_SIZE}
min_count   = {MIN_COUNT}
epochs      = {EPOCHS}
workers     = {WORKERS}
"""
    )

    start = time.time()

    model = Word2Vec(
        sentences=corpus,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=0,             # CBOW
        epochs=EPOCHS,
        seed=RANDOM_STATE,
    )

    elapsed = time.time() - start

    print(
        f"Word2Vec 训练完成，耗时："
        f"{elapsed:.2f} 秒"
    )

    print(
        f"Vocabulary Size："
        f"{len(model.wv):,}"
    )

    save_path = (
        MODEL_DIR
        / f"{model_name}.model"
    )

    model.save(
        str(save_path)
    )

    print(
        f"模型保存到：{save_path}"
    )

    return model


# ============================================================
# 7. 计算一个文档的平均词向量
# ============================================================

def mean_document_vector(
    tokens,
    embedding,
    vector_size=100,
):
    """
    embedding 可以是：
    - gensim Word2Vec.wv
    - Python dict（GloVe）
    """

    vectors = []

    for word in tokens:

        if word in embedding:
            vectors.append(
                embedding[word]
            )

    # 如果整篇文章的所有词都不在词表
    if len(vectors) == 0:
        return np.zeros(
            vector_size,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    ).astype(np.float32)


# ============================================================
# 8. 将全部文档转为向量
# ============================================================

def build_document_matrix(
    tokenized_docs,
    embedding,
    name,
    vector_size=100,
):
    print(
        f"\n正在生成 {name} 文档向量..."
    )

    start = time.time()

    matrix = np.vstack([
        mean_document_vector(
            tokens,
            embedding,
            vector_size,
        )
        for tokens in tokenized_docs
    ])

    elapsed = time.time() - start

    print(
        f"{name} Shape：{matrix.shape}"
    )

    print(
        f"耗时：{elapsed:.2f} 秒"
    )

    return matrix


# ============================================================
# 9. OOV 统计
# ============================================================

def calculate_oov(
    tokenized_docs,
    embedding,
):
    """
    OOV：
    Out-of-Vocabulary（未登录词）

    统计测试语料中有多少 token 不存在于词向量词表中。
    """

    total = 0
    oov = 0

    unique_total = set()
    unique_oov = set()

    for doc in tokenized_docs:

        for word in doc:

            total += 1
            unique_total.add(word)

            if word not in embedding:
                oov += 1
                unique_oov.add(word)

    token_oov_rate = (
        oov / total
        if total > 0
        else 0
    )

    unique_oov_rate = (
        len(unique_oov)
        / len(unique_total)
        if len(unique_total) > 0
        else 0
    )

    return {
        "total_tokens": total,
        "oov_tokens": oov,
        "token_oov_rate": token_oov_rate,

        "unique_words": len(unique_total),
        "unique_oov_words": len(unique_oov),
        "unique_oov_rate": unique_oov_rate,
    }


# ============================================================
# 10. 保存 Confusion Matrix
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    experiment_name,
):
    labels = sorted(
        pd.Series(y_true).unique()
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        f"{experiment_name} - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        RESULT_DIR
        / f"{experiment_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return cm


# ============================================================
# 11. Logistic Regression 分类实验
# ============================================================

def run_classifier(
    experiment_name,

    X_train_vec,
    X_val_vec,
    X_test_vec,

    y_train,
    y_val,
    y_test,

    X_test_text,

    oov_statistics,
):
    print("\n")
    print("=" * 75)
    print(
        f"分类实验：{experiment_name}"
    )
    print("=" * 75)

    model = LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    print(
        "\n训练 Logistic Regression..."
    )

    start = time.time()

    model.fit(
        X_train_vec,
        y_train,
    )

    elapsed = time.time() - start

    print(
        f"训练完成，耗时：{elapsed:.2f} 秒"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_pred = model.predict(
        X_val_vec
    )

    val_acc = accuracy_score(
        y_val,
        val_pred,
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
    )

    print("\nValidation：")

    print(
        f"Accuracy : {val_acc:.4f}"
    )

    print(
        f"Macro-F1 : {val_f1:.4f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_pred = model.predict(
        X_test_vec
    )

    test_acc = accuracy_score(
        y_test,
        test_pred,
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
    )

    print("\nTest：")

    print(
        f"Accuracy : {test_acc:.4f}"
    )

    print(
        f"Macro-F1 : {test_f1:.4f}"
    )

    print(
        "\nClassification Report："
    )

    report_text = classification_report(
        y_test,
        test_pred,
        digits=4,
    )

    print(report_text)

    # --------------------------------------------------------
    # 保存 Classification Report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        test_pred,
        output_dict=True,
    )

    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
    )

    report_df.to_csv(
        RESULT_DIR
        / f"{experiment_name}_classification_report.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 混淆矩阵
    # --------------------------------------------------------

    cm = save_confusion_matrix(
        y_test,
        test_pred,
        experiment_name,
    )

    print("\nConfusion Matrix：")
    print(cm)

    # --------------------------------------------------------
    # 错误案例
    # --------------------------------------------------------

    result_df = pd.DataFrame({
        "text": list(X_test_text),
        "true_label": list(y_test),
        "pred_label": test_pred,
    })

    wrong_df = result_df[
        result_df["true_label"]
        !=
        result_df["pred_label"]
    ].copy()

    wrong_path = (
        RESULT_DIR
        / f"{experiment_name}_wrong_cases.csv"
    )

    wrong_df.to_csv(
        wrong_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n错误案例数："
        f"{len(wrong_df)} / {len(result_df)}"
    )

    print(
        f"错误案例保存到：{wrong_path}"
    )

    # --------------------------------------------------------
    # OOV
    # --------------------------------------------------------

    print("\nOOV Statistics：")

    print(
        "Token OOV Rate："
        f"{oov_statistics['token_oov_rate']:.2%}"
    )

    print(
        "Unique Word OOV Rate："
        f"{oov_statistics['unique_oov_rate']:.2%}"
    )

    return {
        "name": experiment_name,

        "model": model,

        "val_pred": val_pred,
        "test_pred": test_pred,

        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,

        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,

        "oov": oov_statistics,
    }


# ============================================================
# 12. Word2Vec 相似词分析
# ============================================================

def save_similarity_examples(
    model,
    model_name,
):
    """
    用于实验报告分析。

    比较 NYT Word2Vec 和 AG Word2Vec
    在一些关键词附近学到了哪些词。
    """

    query_words = [
        "market",
        "company",
        "president",
        "government",
        "football",
        "team",
    ]

    rows = []

    print("\n")
    print(
        f"{model_name} Similarity Examples"
    )

    print("-" * 75)

    for word in query_words:

        if word not in model.wv:
            print(
                f"\n{word}: 不在词表中"
            )
            continue

        similar = model.wv.most_similar(
            word,
            topn=10,
        )

        print(f"\n[{word}]")

        for rank, (
            similar_word,
            score,
        ) in enumerate(
            similar,
            start=1,
        ):

            print(
                f"{rank:2d}. "
                f"{similar_word:<20}"
                f"{score:.4f}"
            )

            rows.append({
                "query": word,
                "rank": rank,
                "similar_word": similar_word,
                "similarity": score,
            })

    pd.DataFrame(rows).to_csv(
        RESULT_DIR
        / f"{model_name}_similar_words.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 13. 比较两种 Word2Vec 的预测差异
# ============================================================

def compare_word2vec_predictions(
    X_test,
    y_test,
    nyt_result,
    ag_result,
):
    print("\n")
    print("=" * 75)
    print(
        "NYT Word2Vec vs AG Word2Vec Bad Case 分析"
    )
    print("=" * 75)

    df = pd.DataFrame({
        "text": list(X_test),
        "true_label": list(y_test),

        "nyt_word2vec_pred":
            nyt_result["test_pred"],

        "ag_word2vec_pred":
            ag_result["test_pred"],
    })

    different = df[
        df["nyt_word2vec_pred"]
        !=
        df["ag_word2vec_pred"]
    ].copy()

    nyt_right = df[
        (
            df["nyt_word2vec_pred"]
            ==
            df["true_label"]
        )
        &
        (
            df["ag_word2vec_pred"]
            !=
            df["true_label"]
        )
    ].copy()

    ag_right = df[
        (
            df["ag_word2vec_pred"]
            ==
            df["true_label"]
        )
        &
        (
            df["nyt_word2vec_pred"]
            !=
            df["true_label"]
        )
    ].copy()

    print(
        f"\n预测不同：{len(different)}"
    )

    print(
        "NYT W2V 正确、AG W2V 错误："
        f"{len(nyt_right)}"
    )

    print(
        "AG W2V 正确、NYT W2V 错误："
        f"{len(ag_right)}"
    )

    different.to_csv(
        RESULT_DIR
        / "nyt_vs_ag_word2vec_disagreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nyt_right.to_csv(
        RESULT_DIR
        / "nyt_right_ag_wrong.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ag_right.to_csv(
        RESULT_DIR
        / "ag_right_nyt_wrong.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 14. 加载 GloVe
# ============================================================

def load_glove(
    glove_path,
    required_words,
):
    """
    为了节省内存：

    不把整个 GloVe 都加载进内存，
    只加载 NYT 中实际出现的单词。

    glove.6B.100d.txt 很大，
    这样做对普通 CPU / 内存电脑更友好。
    """

    print("\n")
    print("=" * 75)
    print("加载 Stanford GloVe 100d")
    print("=" * 75)

    if not glove_path.exists():

        print(
            f"\n没有找到 GloVe 文件：\n"
            f"{glove_path}"
        )

        print(
            "\n因此本次自动跳过 GloVe 实验。"
        )

        print(
            "下载 glove.6B.100d.txt 后重新运行即可。"
        )

        return None

    required_words = set(
        required_words
    )

    glove = {}

    start = time.time()

    with open(
        glove_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            parts = line.rstrip().split()

            if len(parts) != VECTOR_SIZE + 1:
                continue

            word = parts[0]

            # 只保留 NYT 用得到的词
            if word not in required_words:
                continue

            vector = np.asarray(
                parts[1:],
                dtype=np.float32,
            )

            glove[word] = vector

    elapsed = time.time() - start

    print(
        f"\nGloVe 有效词数："
        f"{len(glove):,}"
    )

    print(
        f"加载耗时：{elapsed:.2f} 秒"
    )

    return glove


# ============================================================
# 15. 保存最终总结
# ============================================================

def save_summary(results):

    rows = []

    for result in results:

        rows.append({
            "Representation":
                result["name"],

            "Dimension":
                VECTOR_SIZE,

            "Validation Accuracy":
                result["val_accuracy"],

            "Validation Macro-F1":
                result["val_macro_f1"],

            "Test Accuracy":
                result["test_accuracy"],

            "Test Macro-F1":
                result["test_macro_f1"],

            "Token OOV Rate":
                result["oov"][
                    "token_oov_rate"
                ],

            "Unique Word OOV Rate":
                result["oov"][
                    "unique_oov_rate"
                ],
        })

    df = pd.DataFrame(rows)

    output_path = (
        RESULT_DIR
        / "task2_summary.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print("=" * 75)
    print("Task 2 最终实验结果")
    print("=" * 75)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\n结果已保存："
        f"{output_path}"
    )


# ============================================================
# 16. 主程序
# ============================================================

def main():

    print()
    print("=" * 75)
    print("NLP Lab 1 - Task 2")
    print("Word2Vec / GloVe + Logistic Regression")
    print("=" * 75)

    print(
        f"\n检测到 CPU 核心数：{CPU_COUNT}"
    )

    print(
        f"Word2Vec 使用 workers：{WORKERS}"
    )

    # ========================================================
    # 读取数据
    # ========================================================

    nyt_df = load_nyt()

    ag_texts = load_ag()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_nyt(nyt_df)

    # ========================================================
    # NYT 分词
    # ========================================================

    train_tokens = tokenize_corpus(
        X_train.tolist(),
        "NYT Train",
    )

    val_tokens = tokenize_corpus(
        X_val.tolist(),
        "NYT Validation",
    )

    test_tokens = tokenize_corpus(
        X_test.tolist(),
        "NYT Test",
    )

    # ========================================================
    # 实验 A：
    # NYT Train Word2Vec
    # ========================================================

    nyt_w2v = train_word2vec(
        train_tokens,
        "nyt_word2vec",
    )

    save_similarity_examples(
        nyt_w2v,
        "nyt_word2vec",
    )

    # 文档向量
    nyt_train_vec = build_document_matrix(
        train_tokens,
        nyt_w2v.wv,
        "NYT W2V Train",
    )

    nyt_val_vec = build_document_matrix(
        val_tokens,
        nyt_w2v.wv,
        "NYT W2V Validation",
    )

    nyt_test_vec = build_document_matrix(
        test_tokens,
        nyt_w2v.wv,
        "NYT W2V Test",
    )

    # OOV
    nyt_oov = calculate_oov(
        test_tokens,
        nyt_w2v.wv,
    )

    # 分类
    nyt_result = run_classifier(
        experiment_name="nyt_word2vec",

        X_train_vec=nyt_train_vec,
        X_val_vec=nyt_val_vec,
        X_test_vec=nyt_test_vec,

        y_train=y_train,
        y_val=y_val,
        y_test=y_test,

        X_test_text=X_test,

        oov_statistics=nyt_oov,
    )

    # ========================================================
    # 实验 B：
    # AG Word2Vec
    # ========================================================

    ag_tokens = tokenize_corpus(
        ag_texts,
        "AG Corpus",
    )

    ag_w2v = train_word2vec(
        ag_tokens,
        "ag_word2vec",
    )

    save_similarity_examples(
        ag_w2v,
        "ag_word2vec",
    )

    # 用 AG Word2Vec 表示 NYT
    ag_train_vec = build_document_matrix(
        train_tokens,
        ag_w2v.wv,
        "AG W2V -> NYT Train",
    )

    ag_val_vec = build_document_matrix(
        val_tokens,
        ag_w2v.wv,
        "AG W2V -> NYT Validation",
    )

    ag_test_vec = build_document_matrix(
        test_tokens,
        ag_w2v.wv,
        "AG W2V -> NYT Test",
    )

    ag_oov = calculate_oov(
        test_tokens,
        ag_w2v.wv,
    )

    ag_result = run_classifier(
        experiment_name="ag_word2vec",

        X_train_vec=ag_train_vec,
        X_val_vec=ag_val_vec,
        X_test_vec=ag_test_vec,

        y_train=y_train,
        y_val=y_val,
        y_test=y_test,

        X_test_text=X_test,

        oov_statistics=ag_oov,
    )

    # ========================================================
    # NYT W2V vs AG W2V
    # ========================================================

    compare_word2vec_predictions(
        X_test,
        y_test,
        nyt_result,
        ag_result,
    )

    results = [
        nyt_result,
        ag_result,
    ]

    # ========================================================
    # 实验 C：
    # Stanford GloVe
    # ========================================================

    # 收集 NYT 中需要用到的词
    all_nyt_tokens = (
        train_tokens
        + val_tokens
        + test_tokens
    )

    required_words = set()

    for doc in all_nyt_tokens:
        required_words.update(doc)

    glove = load_glove(
        GLOVE_PATH,
        required_words,
    )

    if glove is not None:

        glove_train_vec = (
            build_document_matrix(
                train_tokens,
                glove,
                "GloVe -> NYT Train",
            )
        )

        glove_val_vec = (
            build_document_matrix(
                val_tokens,
                glove,
                "GloVe -> NYT Validation",
            )
        )

        glove_test_vec = (
            build_document_matrix(
                test_tokens,
                glove,
                "GloVe -> NYT Test",
            )
        )

        glove_oov = calculate_oov(
            test_tokens,
            glove,
        )

        glove_result = run_classifier(
            experiment_name="glove_100d",

            X_train_vec=
                glove_train_vec,

            X_val_vec=
                glove_val_vec,

            X_test_vec=
                glove_test_vec,

            y_train=y_train,
            y_val=y_val,
            y_test=y_test,

            X_test_text=X_test,

            oov_statistics=
                glove_oov,
        )

        results.append(
            glove_result
        )

    # ========================================================
    # 保存 OOV 统计
    # ========================================================

    oov_rows = []

    for result in results:

        stats = result["oov"]

        oov_rows.append({
            "Representation":
                result["name"],

            "Total Tokens":
                stats["total_tokens"],

            "OOV Tokens":
                stats["oov_tokens"],

            "Token OOV Rate":
                stats["token_oov_rate"],

            "Unique Words":
                stats["unique_words"],

            "Unique OOV Words":
                stats["unique_oov_words"],

            "Unique OOV Rate":
                stats["unique_oov_rate"],
        })

    pd.DataFrame(
        oov_rows
    ).to_csv(
        RESULT_DIR
        / "oov_statistics.csv",

        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 最终总结
    # ========================================================

    save_summary(
        results
    )

    print("\n")
    print("=" * 75)
    print("Task 2 全部完成！")
    print("=" * 75)

    print(
        f"\n结果目录：\n"
        f"{RESULT_DIR}"
    )

    print(
        "\n重点查看："
    )

    print(
        """
1. task2_summary.csv
2. oov_statistics.csv

3. nyt_word2vec_confusion_matrix.png
4. ag_word2vec_confusion_matrix.png

5. nyt_word2vec_wrong_cases.csv
6. ag_word2vec_wrong_cases.csv

7. nyt_vs_ag_word2vec_disagreement.csv
8. nyt_right_ag_wrong.csv
9. ag_right_nyt_wrong.csv

10. nyt_word2vec_similar_words.csv
11. ag_word2vec_similar_words.csv
"""
    )

    if glove is None:
        print(
            "注意：本次没有运行 GloVe。"
        )

        print(
            "将 glove.6B.100d.txt 放到："
        )

        print(
            BASE_DIR
            / "glove"
            / "glove.6B.100d.txt"
        )

        print(
            "\n然后重新运行本程序即可。"
        )


if __name__ == "__main__":
    main()