# -*- coding: utf-8 -*-

"""
NLP Lab 1 - Task 3
BERT-base-uncased Fine-tuning

作业要求：
- Pre-trained model: google-bert/bert-base-uncased
- Maximum sequence length: 64
- Number of epochs: 3
- Fine-tuning
- 在 NYT Test Set 上报告 Accuracy 和 Macro-F1

本代码还会保存：
- 3 个 epoch 的训练历史
- 最佳 Validation Macro-F1 checkpoint
- Test classification report
- confusion matrix
- wrong cases
- Task 1 / Task 2 / Task 3 总对比
"""

from pathlib import Path
import json
import random
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
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
DATA_PATH = BASE_DIR / "nyt.csv"

# 与旧的 Frozen Tiny BERT 结果分开，避免覆盖
RESULT_DIR = BASE_DIR / "task3_results"
BEST_MODEL_DIR = RESULT_DIR / "best_model"

RESULT_DIR.mkdir(exist_ok=True)
BEST_MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. 作业规定参数
# ============================================================

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
EPOCHS = 3

# GPU 通常可以用 16；CPU 为稳妥起见使用 8
if torch.cuda.is_available():
    BATCH_SIZE = 16
else:
    BATCH_SIZE = 8

EVAL_BATCH_SIZE = BATCH_SIZE

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

RANDOM_STATE = 42


# ============================================================
# 2. 自动检测设备
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# 3. 固定随机种子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. 读取 NYT 数据
# ============================================================

def load_data():

    print("=" * 78)
    print("1. 读取 NYT 数据集")
    print("=" * 78)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\n数据路径：{DATA_PATH}")
    print(f"原始样本数：{len(df)}")
    print(f"列名：{list(df.columns)}")

    if (
        "text" not in df.columns
        or "label" not in df.columns
    ):
        raise ValueError(
            "nyt.csv 必须包含 text 和 label 两列。"
        )

    df = df[["text", "label"]].copy()
    df = df.dropna(subset=["text", "label"])

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    df = df[
        df["text"].str.strip() != ""
    ].copy()

    df = df.reset_index(drop=True)

    print(f"清洗后样本数：{len(df)}")

    print("\n类别分布：")
    print(df["label"].value_counts())

    return df


# ============================================================
# 5. 与 Task 1 / Task 2 一致的数据划分
# ============================================================

def split_data(df):

    print("\n")
    print("=" * 78)
    print("2. Train / Validation / Test = 8 : 1 : 1")
    print("=" * 78)

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
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ============================================================
# 6. 标签编码
# ============================================================

def build_label_mapping(y_train):

    labels = sorted(
        y_train.unique().tolist()
    )

    label2id = {
        label: i
        for i, label in enumerate(labels)
    }

    id2label = {
        i: label
        for label, i in label2id.items()
    }

    print("\n标签映射：")

    for label, idx in label2id.items():
        print(f"{label:10s} -> {idx}")

    mapping_path = (
        RESULT_DIR
        /
        "label_mapping.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {
                    str(k): v
                    for k, v in id2label.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label2id, id2label


# ============================================================
# 7. Dataset
# ============================================================

class BertTextDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        label2id,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids":
                encoded["input_ids"].squeeze(0),

            "attention_mask":
                encoded["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.label2id[label],
                    dtype=torch.long,
                ),
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = (
                encoded["token_type_ids"]
                .squeeze(0)
            )

        return item


# ============================================================
# 8. DataLoader
# ============================================================

def create_dataloaders(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    tokenizer,
    label2id,
):

    train_dataset = BertTextDataset(
        X_train,
        y_train,
        tokenizer,
        label2id,
    )

    val_dataset = BertTextDataset(
        X_val,
        y_val,
        tokenizer,
        label2id,
    )

    test_dataset = BertTextDataset(
        X_test,
        y_test,
        tokenizer,
        label2id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# 9. 单轮训练
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    epoch,
):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{EPOCHS} Train",
    )

    for batch in progress_bar:

        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss
        /
        max(len(dataloader), 1)
    )

    return average_loss


# ============================================================
# 10. Validation / Test 评估
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    id2label,
    desc,
):

    model.eval()

    total_loss = 0.0

    true_ids = []
    pred_ids = []

    progress_bar = tqdm(
        dataloader,
        desc=desc,
    )

    for batch in progress_bar:

        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        if outputs.loss is not None:
            total_loss += outputs.loss.item()

        logits = outputs.logits

        preds = torch.argmax(
            logits,
            dim=1,
        )

        pred_ids.extend(
            preds.detach().cpu().tolist()
        )

        true_ids.extend(
            batch["labels"]
            .detach()
            .cpu()
            .tolist()
        )

    y_true = [
        id2label[i]
        for i in true_ids
    ]

    y_pred = [
        id2label[i]
        for i in pred_ids
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    average_loss = (
        total_loss
        /
        max(len(dataloader), 1)
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================
# 11. 混淆矩阵
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
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
        "BERT-base Fine-tuning - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        RESULT_DIR
        /
        "bert_finetune_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return cm, output_path


# ============================================================
# 12. 保存最佳模型
# ============================================================

def save_best_model(
    model,
    tokenizer,
):

    model.save_pretrained(
        BEST_MODEL_DIR
    )

    tokenizer.save_pretrained(
        BEST_MODEL_DIR
    )


# ============================================================
# 13. Fine-tuning 主过程
# ============================================================

def fine_tune(
    model,
    tokenizer,
    train_loader,
    val_loader,
    id2label,
):

    print("\n")
    print("=" * 78)
    print("4. BERT-base Fine-tuning")
    print("=" * 78)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_training_steps = (
        len(train_loader)
        *
        EPOCHS
    )

    warmup_steps = int(
        total_training_steps
        *
        WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print(
        f"\nTraining Steps : "
        f"{total_training_steps}"
    )

    print(
        f"Warmup Steps   : "
        f"{warmup_steps}"
    )

    print(
        f"Learning Rate  : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Epochs         : "
        f"{EPOCHS}"
    )

    history = []

    best_val_macro_f1 = -1.0
    best_epoch = -1

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        print("\n")
        print("-" * 78)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 78)

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch,
        )

        val_result = evaluate(
            model,
            val_loader,
            id2label,
            desc=f"Epoch {epoch}/{EPOCHS} Validation",
        )

        epoch_elapsed = (
            time.time()
            -
            epoch_start
        )

        print(
            f"\nTrain Loss          : "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss     : "
            f"{val_result['loss']:.4f}"
        )

        print(
            f"Validation Accuracy : "
            f"{val_result['accuracy']:.4f}"
        )

        print(
            f"Validation Macro-F1 : "
            f"{val_result['macro_f1']:.4f}"
        )

        print(
            f"Epoch Time          : "
            f"{epoch_elapsed:.2f} 秒"
        )

        history.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Validation Loss":
                val_result["loss"],
            "Validation Accuracy":
                val_result["accuracy"],
            "Validation Macro-F1":
                val_result["macro_f1"],
            "Epoch Time Seconds":
                epoch_elapsed,
        })

        # 不 Early Stop，必须训练满 3 epochs。
        # 这里只根据 Validation Macro-F1 保存最佳 checkpoint。
        if (
            val_result["macro_f1"]
            >
            best_val_macro_f1
        ):

            best_val_macro_f1 = (
                val_result["macro_f1"]
            )

            best_epoch = epoch

            save_best_model(
                model,
                tokenizer,
            )

            print(
                "\n当前为最佳 Validation Macro-F1，"
                "已保存 checkpoint。"
            )

    total_elapsed = (
        time.time()
        -
        total_start
    )

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        RESULT_DIR
        /
        "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print("=" * 78)
    print("3 epochs Fine-tuning 完成")
    print("=" * 78)

    print(
        f"\n最佳 Epoch：{best_epoch}"
    )

    print(
        f"最佳 Validation Macro-F1："
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"总训练耗时："
        f"{total_elapsed:.2f} 秒"
    )

    print(
        f"训练历史保存到："
        f"{history_path}"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1":
            best_val_macro_f1,
        "total_training_seconds":
            total_elapsed,
    }


# ============================================================
# 14. Test Set 最终评估
# ============================================================

def final_test(
    test_loader,
    id2label,
    X_test,
    training_info,
):

    print("\n")
    print("=" * 78)
    print("5. 在 NYT Test Set 上进行最终评估")
    print("=" * 78)

    print(
        f"\n载入最佳模型："
        f"{BEST_MODEL_DIR}"
    )

    best_model = (
        BertForSequenceClassification
        .from_pretrained(
            BEST_MODEL_DIR
        )
    )

    best_model = best_model.to(
        DEVICE
    )

    test_result = evaluate(
        best_model,
        test_loader,
        id2label,
        desc="Test",
    )

    y_true = test_result["y_true"]
    y_pred = test_result["y_pred"]

    print("\n")
    print("-" * 78)
    print("Test Result")
    print("-" * 78)

    print(
        f"Accuracy : "
        f"{test_result['accuracy']:.4f}"
    )

    print(
        f"Macro-F1 : "
        f"{test_result['macro_f1']:.4f}"
    )

    print(
        "\nClassification Report："
    )

    report_text = classification_report(
        y_true,
        y_pred,
        digits=4,
    )

    print(report_text)

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )

    report_df = (
        pd.DataFrame(
            report_dict
        )
        .transpose()
    )

    report_path = (
        RESULT_DIR
        /
        "bert_finetune_classification_report.csv"
    )

    report_df.to_csv(
        report_path,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm, cm_path = save_confusion_matrix(
        y_true,
        y_pred,
    )

    print(
        "\nConfusion Matrix："
    )

    print(cm)

    # --------------------------------------------------------
    # Wrong Cases
    # --------------------------------------------------------

    result_df = pd.DataFrame({
        "text": list(X_test),
        "true_label": y_true,
        "bert_pred": y_pred,
    })

    wrong_df = result_df[
        result_df["true_label"]
        !=
        result_df["bert_pred"]
    ].copy()

    wrong_path = (
        RESULT_DIR
        /
        "bert_finetune_wrong_cases.csv"
    )

    wrong_df.to_csv(
        wrong_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n错误案例数："
        f"{len(wrong_df)} / "
        f"{len(result_df)}"
    )

    print(
        f"错误案例保存到："
        f"{wrong_path}"
    )

    print("\n")
    print("-" * 78)
    print("部分 BERT Fine-tuning 错误案例")
    print("-" * 78)

    for _, row in (
        wrong_df.head(5)
        .iterrows()
    ):

        text = row["text"]

        if len(text) > 400:
            text = text[:400] + "..."

        print()

        print(
            f"True : "
            f"{row['true_label']}"
        )

        print(
            f"Pred : "
            f"{row['bert_pred']}"
        )

        print(
            f"Text : {text}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "Representation":
                "BERT-base-uncased Fine-tuning",

            "Model":
                MODEL_NAME,

            "Max Length":
                MAX_LENGTH,

            "Epochs":
                EPOCHS,

            "Batch Size":
                BATCH_SIZE,

            "Learning Rate":
                LEARNING_RATE,

            "Best Epoch":
                training_info["best_epoch"],

            "Best Validation Macro-F1":
                training_info[
                    "best_val_macro_f1"
                ],

            "Test Accuracy":
                test_result["accuracy"],

            "Test Macro-F1":
                test_result["macro_f1"],
        }
    ])

    summary_path = (
        RESULT_DIR
        /
        "task3_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print("=" * 78)
    print("Task 3 最终实验结果")
    print("=" * 78)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\nSummary 保存到："
        f"{summary_path}"
    )

    return {
        "test_accuracy":
            test_result["accuracy"],

        "test_macro_f1":
            test_result["macro_f1"],
    }


# ============================================================
# 15. Task 1 / Task 2 / Task 3 总对比
# ============================================================

def build_full_comparison(
    bert_result,
):

    print("\n")
    print("=" * 78)
    print("6. 与 Task 1 / Task 2 结果进行比较")
    print("=" * 78)

    rows = []

    # Task 1
    task1_path = (
        BASE_DIR
        /
        "task1_results"
        /
        "task1_summary.csv"
    )

    if task1_path.exists():

        try:

            task1 = pd.read_csv(
                task1_path
            )

            for _, row in task1.iterrows():

                rows.append({
                    "Task":
                        "Task 1",

                    "Representation":
                        row["Representation"],

                    "Test Accuracy":
                        row["Test Accuracy"],

                    "Test Macro-F1":
                        row["Test Macro-F1"],
                })

        except Exception as e:

            print(
                f"读取 Task 1 结果失败："
                f"{e}"
            )

    else:

        print(
            f"\n未找到 Task 1 Summary："
            f"{task1_path}"
        )

    # Task 2
    task2_path = (
        BASE_DIR
        /
        "task2_results"
        /
        "task2_summary.csv"
    )

    if task2_path.exists():

        try:

            task2 = pd.read_csv(
                task2_path
            )

            for _, row in task2.iterrows():

                rows.append({
                    "Task":
                        "Task 2",

                    "Representation":
                        row["Representation"],

                    "Test Accuracy":
                        row["Test Accuracy"],

                    "Test Macro-F1":
                        row["Test Macro-F1"],
                })

        except Exception as e:

            print(
                f"读取 Task 2 结果失败："
                f"{e}"
            )

    else:

        print(
            f"\n未找到 Task 2 Summary："
            f"{task2_path}"
        )

    # Task 3
    rows.append({
        "Task":
            "Task 3",

        "Representation":
            "BERT-base-uncased Fine-tuning",

        "Test Accuracy":
            bert_result["test_accuracy"],

        "Test Macro-F1":
            bert_result["test_macro_f1"],
    })

    comparison = pd.DataFrame(
        rows
    )

    output_path = (
        RESULT_DIR
        /
        "all_tasks_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\n总对比表保存到："
        f"{output_path}"
    )


# ============================================================
# 16. 主程序
# ============================================================

def main():

    set_seed(
        RANDOM_STATE
    )

    print()
    print("=" * 78)
    print("NLP Lab 1 - Task 3")
    print("BERT-base-uncased Fine-tuning")
    print("=" * 78)

    print(
        f"\nDevice       : {DEVICE}"
    )

    print(
        f"Model        : {MODEL_NAME}"
    )

    print(
        f"Max Length   : {MAX_LENGTH}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Batch Size   : {BATCH_SIZE}"
    )

    print(
        f"Learning Rate: {LEARNING_RATE}"
    )

    if DEVICE.type == "cpu":

        print(
            "\n注意：当前使用 CPU。"
            "BERT-base 进行 3 epochs Fine-tuning "
            "可能需要较长时间。"
        )

        print(
            "如运行时间过长，建议将该脚本上传到 "
            "Google Colab / Kaggle 并使用 GPU。"
        )

    # 数据
    df = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(
        df
    )

    label2id, id2label = (
        build_label_mapping(
            y_train
        )
    )

    # --------------------------------------------------------
    # 加载 Tokenizer
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("3. 加载 BERT-base-uncased")
    print("=" * 78)

    print(
        f"\n正在加载："
        f"{MODEL_NAME}"
    )

    load_start = time.time()

    # 使用 Slow Tokenizer，避开部分新版 transformers
    # 在 Python 3.13 环境中的 Fast Tokenizer 兼容问题。
    tokenizer = (
        BertTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    # DataLoader
    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        tokenizer,
        label2id,
    )

    # --------------------------------------------------------
    # 加载 BERT 分类模型
    # --------------------------------------------------------

    model = (
        BertForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
        )
    )

    model = model.to(
        DEVICE
    )

    load_elapsed = (
        time.time()
        -
        load_start
    )

    print(
        "\n模型加载完成。"
    )

    print(
        f"类别数："
        f"{len(label2id)}"
    )

    print(
        f"加载耗时："
        f"{load_elapsed:.2f} 秒"
    )

    print(
        "\n本实验不会冻结 BERT 参数。"
    )

    print(
        "BERT Encoder 与 Classification Head "
        "都会参与反向传播和参数更新。"
    )

    # Fine-tuning
    training_info = fine_tune(
        model,
        tokenizer,
        train_loader,
        val_loader,
        id2label,
    )

    # 释放训练模型后再加载最佳 checkpoint
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test
    bert_result = final_test(
        test_loader,
        id2label,
        X_test,
        training_info,
    )

    # 总对比
    build_full_comparison(
        bert_result
    )

    print("\n")
    print("=" * 78)
    print("Task 3 全部完成！")
    print("=" * 78)

    print(
        f"\n结果目录：\n"
        f"{RESULT_DIR}"
    )

    print(
        """
重点查看：

1. task3_summary.csv
   最终 Accuracy / Macro-F1

2. training_history.csv
   3 个 epoch 的 Train Loss / Validation 指标

3. bert_finetune_classification_report.csv
   每个类别的 Precision / Recall / F1

4. bert_finetune_confusion_matrix.png
   混淆矩阵

5. bert_finetune_wrong_cases.csv
   BERT Fine-tuning 错例

6. all_tasks_comparison.csv
   Task 1 / Task 2 / Task 3 最终对比

7. best_model/
   Validation Macro-F1 最好的 BERT checkpoint

作业规定参数：
- google-bert/bert-base-uncased
- max_length = 64
- epochs = 3
- Fine-tuning
"""
    )


if __name__ == "__main__":
    main()
