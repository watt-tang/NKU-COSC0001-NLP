# -*- coding: utf-8 -*-

"""
NLP Lab 1 - Task 1
==================

实验目标：
1. 使用 NYT 数据集进行新闻文本分类
2. 比较两种 Bag of Words（词袋）表示：
   - Binary Bag of Words：只记录单词是否出现（0/1）
   - Frequency Bag of Words：记录单词出现次数
3. 两种表示均使用 Logistic Regression（逻辑回归）分类器
4. 评价指标：
   - Accuracy（准确率）
   - Macro-F1（宏平均 F1）
   - 每个类别的 Precision / Recall / F1
5. 额外输出：
   - Confusion Matrix（混淆矩阵）
   - 错误分类样本
   - Binary 与 Frequency 预测不同的样本
   - 每个类别最重要的词

注意：
- 数据按照 Train / Validation / Test = 8 : 1 : 1 划分
- 使用 stratify 保证各个数据集中的类别比例基本一致
- Vectorizer 只能在训练集上 fit，避免 Data Leakage（数据泄漏）
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 0. 基本配置
# ============================================================

# task1_bow.py 所在目录
BASE_DIR = Path(__file__).resolve().parent

# NYT 数据集路径
DATA_PATH = BASE_DIR / "nyt.csv"

# 实验结果保存目录
RESULT_DIR = BASE_DIR / "task1_results"
RESULT_DIR.mkdir(exist_ok=True)

# 固定随机种子，保证每次运行划分结果一致
RANDOM_STATE = 42


# ============================================================
# 1. 读取并检查数据
# ============================================================

def load_data():
    print("=" * 70)
    print("1. 读取 NYT 数据集")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据集：{DATA_PATH}\n"
            f"请确认 nyt.csv 和 task1_bow.py 在同一个目录下。"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\n数据集路径：{DATA_PATH}")
    print(f"原始数据量：{len(df)}")
    print(f"列名：{list(df.columns)}")

    # 检查是否存在实验需要的两列
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "nyt.csv 必须至少包含 'text' 和 'label' 两列。\n"
            f"当前列名：{list(df.columns)}"
        )

    # 只保留 text 和 label
    df = df[["text", "label"]].copy()

    # 删除空值
    before = len(df)

    df = df.dropna(subset=["text", "label"])

    # 转成字符串
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    # 删除空文本
    df = df[df["text"].str.strip() != ""]

    after = len(df)

    print(f"清洗后数据量：{after}")
    print(f"删除无效数据：{before - after}")

    print("\n各类别样本数量：")
    print(df["label"].value_counts())

    print("\n各类别样本比例：")
    print(
        (df["label"].value_counts(normalize=True) * 100)
        .round(2)
        .astype(str)
        + "%"
    )

    print()

    return df


# ============================================================
# 2. 8 : 1 : 1 划分数据集
# ============================================================

def split_data(df):
    print("=" * 70)
    print("2. 划分 Train / Validation / Test = 8 : 1 : 1")
    print("=" * 70)

    X = df["text"]
    y = df["label"]

    # 第一次：
    # 80% Train
    # 20% 临时数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 第二次：
    # 将 20% 临时数据一半作为 Validation，一半作为 Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"\nTrain      : {len(X_train):5d} ({len(X_train) / len(df):.2%})")
    print(f"Validation : {len(X_val):5d} ({len(X_val) / len(df):.2%})")
    print(f"Test       : {len(X_test):5d} ({len(X_test) / len(df):.2%})")

    # 打印每个集合的类别分布
    print_split_distribution("Train", y_train)
    print_split_distribution("Validation", y_val)
    print_split_distribution("Test", y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_distribution(name, labels):
    print(f"\n{name} 类别分布：")

    counts = labels.value_counts()
    ratios = labels.value_counts(normalize=True)

    for label in sorted(counts.index):
        print(
            f"  {label:<12} "
            f"{counts[label]:5d} "
            f"({ratios[label] * 100:6.2f}%)"
        )


# ============================================================
# 3. 保存混淆矩阵
# ============================================================

def save_confusion_matrix(y_true, y_pred, experiment_name):
    labels = sorted(y_true.unique())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(f"{experiment_name} - Confusion Matrix")

    plt.tight_layout()

    output_path = RESULT_DIR / f"{experiment_name}_confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return cm


# ============================================================
# 4. 保存每个类别最重要的词
# ============================================================

def save_top_words(model, vectorizer, experiment_name, top_n=20):
    """
    Logistic Regression 每一个类别都有一组特征权重。

    权重越大的词，越容易把文本推向对应类别。

    例如：
        sports:
            game
            team
            player

        business:
            company
            market
            stock
    """

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    classes = model.classes_

    rows = []

    print(f"\n{experiment_name}：各类别 Top {top_n} 特征词")

    print("-" * 70)

    for class_index, class_name in enumerate(classes):

        coefficients = model.coef_[class_index]

        # 从大到小排序
        top_indices = np.argsort(coefficients)[-top_n:][::-1]

        top_features = feature_names[top_indices]
        top_scores = coefficients[top_indices]

        print(f"\n[{class_name}]")

        for rank, (word, score) in enumerate(
            zip(top_features, top_scores),
            start=1
        ):
            print(
                f"{rank:2d}. "
                f"{word:<20} "
                f"{score:.4f}"
            )

            rows.append({
                "class": class_name,
                "rank": rank,
                "word": word,
                "coefficient": score,
            })

    pd.DataFrame(rows).to_csv(
        RESULT_DIR / f"{experiment_name}_top_words.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 5. 单次 BoW 实验
# ============================================================

def run_bow_experiment(
    experiment_name,
    binary,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):
    print("\n")
    print("=" * 70)
    print(f"开始实验：{experiment_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # 5.1 建立 Bag of Words
    # --------------------------------------------------------

    vectorizer = CountVectorizer(
        binary=binary,

        # 英文统一转为小写
        lowercase=True,

        # 默认只保留长度 >= 2 的字母数字 token
        token_pattern=r"(?u)\b\w\w+\b",
    )

    # IMPORTANT：
    # 只能在 Train 上 fit。
    #
    # Validation/Test 只能 transform。
    #
    # 否则会造成 Data Leakage（数据泄漏）。
    X_train_vec = vectorizer.fit_transform(X_train)

    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print("\n文本向量化完成。")

    print(
        f"Vocabulary Size（词表大小）："
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"Train Matrix Shape："
        f"{X_train_vec.shape}"
    )

    # --------------------------------------------------------
    # 5.2 Logistic Regression
    # --------------------------------------------------------

    print("\n开始训练 Logistic Regression ...")

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_vec,
        y_train,
    )

    print("模型训练完成。")

    # --------------------------------------------------------
    # 5.3 Validation
    # --------------------------------------------------------

    val_pred = model.predict(X_val_vec)

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
    )

    print("\n" + "-" * 70)
    print("Validation Result")
    print("-" * 70)

    print(f"Accuracy : {val_accuracy:.4f}")
    print(f"Macro-F1 : {val_macro_f1:.4f}")

    print("\nClassification Report：")

    print(
        classification_report(
            y_val,
            val_pred,
            digits=4,
        )
    )

    # --------------------------------------------------------
    # 5.4 Test
    # --------------------------------------------------------

    test_pred = model.predict(X_test_vec)

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
    )

    print("\n" + "-" * 70)
    print("Test Result")
    print("-" * 70)

    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Macro-F1 : {test_macro_f1:.4f}")

    print("\nClassification Report：")

    test_report_text = classification_report(
        y_test,
        test_pred,
        digits=4,
    )

    print(test_report_text)

    # --------------------------------------------------------
    # 5.5 保存 Classification Report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        test_pred,
        output_dict=True,
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    report_df.to_csv(
        RESULT_DIR / f"{experiment_name}_classification_report.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 5.6 保存混淆矩阵
    # --------------------------------------------------------

    cm = save_confusion_matrix(
        y_test,
        test_pred,
        experiment_name,
    )

    print("\nConfusion Matrix：")
    print(cm)

    # --------------------------------------------------------
    # 5.7 保存错误样本
    # --------------------------------------------------------

    result_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "pred_label": test_pred,
    })

    wrong_df = result_df[
        result_df["true_label"]
        != result_df["pred_label"]
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
        f"\n错误分类数量："
        f"{len(wrong_df)} / {len(result_df)}"
    )

    print(
        f"错误案例已保存："
        f"{wrong_path}"
    )

    # --------------------------------------------------------
    # 5.8 打印几个错误案例
    # --------------------------------------------------------

    print("\n部分错误案例：")

    print("-" * 70)

    for i, row in wrong_df.head(5).iterrows():

        text = row["text"]

        # 防止输出整篇长新闻
        if len(text) > 300:
            text = text[:300] + "..."

        print(f"\nTrue : {row['true_label']}")
        print(f"Pred : {row['pred_label']}")
        print(f"Text : {text}")

    # --------------------------------------------------------
    # 5.9 保存重要特征词
    # --------------------------------------------------------

    save_top_words(
        model=model,
        vectorizer=vectorizer,
        experiment_name=experiment_name,
        top_n=20,
    )

    return {
        "name": experiment_name,

        "model": model,
        "vectorizer": vectorizer,

        "val_pred": val_pred,
        "test_pred": test_pred,

        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,

        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,

        "vocabulary_size": len(
            vectorizer.vocabulary_
        ),
    }


# ============================================================
# 6. 分析 Binary 和 Frequency 的预测差异
# ============================================================

def compare_predictions(
    X_test,
    y_test,
    binary_result,
    frequency_result,
):
    print("\n")
    print("=" * 70)
    print("Binary BoW vs Frequency BoW 错例对比")
    print("=" * 70)

    compare_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "binary_pred": binary_result["test_pred"],
        "frequency_pred": frequency_result["test_pred"],
    })

    # 两模型预测结果不同
    disagreement_df = compare_df[
        compare_df["binary_pred"]
        != compare_df["frequency_pred"]
    ].copy()

    # Binary 对，Frequency 错
    binary_right_df = compare_df[
        (compare_df["binary_pred"] == compare_df["true_label"])
        &
        (compare_df["frequency_pred"] != compare_df["true_label"])
    ].copy()

    # Frequency 对，Binary 错
    frequency_right_df = compare_df[
        (compare_df["frequency_pred"] == compare_df["true_label"])
        &
        (compare_df["binary_pred"] != compare_df["true_label"])
    ].copy()

    print(
        f"\n两种表示预测不同的样本数："
        f"{len(disagreement_df)}"
    )

    print(
        f"Binary 正确、Frequency 错误："
        f"{len(binary_right_df)}"
    )

    print(
        f"Frequency 正确、Binary 错误："
        f"{len(frequency_right_df)}"
    )

    # 保存文件
    disagreement_df.to_csv(
        RESULT_DIR / "binary_vs_frequency_disagreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    binary_right_df.to_csv(
        RESULT_DIR / "binary_right_frequency_wrong.csv",
        index=False,
        encoding="utf-8-sig",
    )

    frequency_right_df.to_csv(
        RESULT_DIR / "frequency_right_binary_wrong.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 打印一些典型样本
    print("\n")
    print("-" * 70)
    print("案例 A：Binary 正确，Frequency 错误")
    print("-" * 70)

    show_cases(binary_right_df)

    print("\n")
    print("-" * 70)
    print("案例 B：Frequency 正确，Binary 错误")
    print("-" * 70)

    show_cases(frequency_right_df)


def show_cases(df, n=3):
    if len(df) == 0:
        print("没有找到对应案例。")
        return

    for _, row in df.head(n).iterrows():

        text = row["text"]

        if len(text) > 400:
            text = text[:400] + "..."

        print()
        print(f"True      : {row['true_label']}")
        print(f"Binary    : {row['binary_pred']}")
        print(f"Frequency : {row['frequency_pred']}")
        print(f"Text      : {text}")


# ============================================================
# 7. 保存最终实验总结
# ============================================================

def save_summary(binary_result, frequency_result):
    summary = pd.DataFrame([
        {
            "Representation": "Binary Bag of Words",
            "Vocabulary Size": binary_result["vocabulary_size"],
            "Validation Accuracy": binary_result["val_accuracy"],
            "Validation Macro-F1": binary_result["val_macro_f1"],
            "Test Accuracy": binary_result["test_accuracy"],
            "Test Macro-F1": binary_result["test_macro_f1"],
        },
        {
            "Representation": "Frequency Bag of Words",
            "Vocabulary Size": frequency_result["vocabulary_size"],
            "Validation Accuracy": frequency_result["val_accuracy"],
            "Validation Macro-F1": frequency_result["val_macro_f1"],
            "Test Accuracy": frequency_result["test_accuracy"],
            "Test Macro-F1": frequency_result["test_macro_f1"],
        },
    ])

    summary_path = RESULT_DIR / "task1_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print("=" * 70)
    print("最终实验结果")
    print("=" * 70)

    # 不显示科学计数法
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\n实验总结已保存到："
        f"{summary_path}"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print("NLP Lab 1 - Task 1")
    print("Bag of Words + Logistic Regression")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 数据划分
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    # --------------------------------------------------------
    # Experiment 1:
    # Binary Bag of Words
    # --------------------------------------------------------

    binary_result = run_bow_experiment(
        experiment_name="binary_bow",
        binary=True,

        X_train=X_train,
        X_val=X_val,
        X_test=X_test,

        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # Experiment 2:
    # Frequency Bag of Words
    # --------------------------------------------------------

    frequency_result = run_bow_experiment(
        experiment_name="frequency_bow",
        binary=False,

        X_train=X_train,
        X_val=X_val,
        X_test=X_test,

        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # 比较两个模型
    # --------------------------------------------------------

    compare_predictions(
        X_test,
        y_test,
        binary_result,
        frequency_result,
    )

    # --------------------------------------------------------
    # 保存实验总结
    # --------------------------------------------------------

    save_summary(
        binary_result,
        frequency_result,
    )

    print("\n")
    print("=" * 70)
    print("Task 1 全部完成！")
    print("=" * 70)

    print(
        f"\n所有实验结果保存在：\n"
        f"{RESULT_DIR}"
    )

    print(
        "\n接下来重点查看：\n"
        "1. task1_summary.csv\n"
        "2. binary_bow_confusion_matrix.png\n"
        "3. frequency_bow_confusion_matrix.png\n"
        "4. binary_bow_wrong_cases.csv\n"
        "5. frequency_bow_wrong_cases.csv\n"
        "6. binary_right_frequency_wrong.csv\n"
        "7. frequency_right_binary_wrong.csv\n"
        "8. binary_bow_top_words.csv\n"
        "9. frequency_bow_top_words.csv"
    )


if __name__ == "__main__":
    main()