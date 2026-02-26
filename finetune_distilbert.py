import numpy as np
import time
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "macro_f1": f1}

def main():
    model_name = "distilbert-base-uncased"
    ds = load_dataset("ag_news")

    # ---- key: fixed split for a fast, reproducible experiment ----
    train_ds = ds["train"].shuffle(seed=42).select(range(20000))  # 2万条训练，够展示
    eval_ds  = ds["test"].select(range(5000))                     # 5k评估，快

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_tok  = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    train_tok = train_tok.rename_column("label", "labels")
    eval_tok  = eval_tok.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    args = TrainingArguments(
        output_dir="./distilbert_agnews_ckpt",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    metrics = trainer.evaluate()
    elapsed = time.time() - start

    print("\n===== DistilBERT Results (subset) =====")
    print(metrics)
    print(f"Training + eval time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()