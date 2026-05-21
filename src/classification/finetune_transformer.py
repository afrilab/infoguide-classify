# src/classification/finetune_transformer.py
"""Fine-tune a transformer classifier with stratified 5-fold CV.

Default model: distilbert-base-uncased — chosen for speed on a tiny corpus
(N=94). With this little data, larger models overfit before they generalize;
the goal is a fair comparison point, not state-of-the-art.

Training regime:
- 5 stratified folds (same seed as supervised_baselines.py)
- 4 epochs, batch size 8, LR 2e-5, weight decay 0.01, linear warmup 10%
- max_length 256 tokens (head_only=True, max_chars=6000 already truncates input)
- Class weighting via WeightedRandomSampler is omitted; class_weight in the
  loss would be the more principled fix but adds complexity for marginal gain
  on this small set. The supervised lexical models do use balanced weighting.

Outputs:
- outputs/classification_outputs/finetune_distilbert_cv.json
- outputs/classification_outputs/finetune_distilbert_cv.md
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from eval_harness import (
    REPO_ROOT,
    aggregate_cv_metrics,
    class_distribution,
    load_dataset,
    score_predictions,
    stratified_kfold_indices,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "classification_outputs"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 4
WARMUP_RATIO = 0.1
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_one_fold(
    train_texts, train_labels, test_texts, test_labels, num_labels, device
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(device)

    train_enc = tokenizer(
        list(train_texts), truncation=True, padding=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        list(test_texts), truncation=True, padding=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_ds = TextDataset(train_enc, train_labels)
    test_ds = TextDataset(test_enc, test_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    num_train_steps = EPOCHS * len(train_loader)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(WARMUP_RATIO * num_train_steps),
        num_training_steps=num_train_steps,
    )

    model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()

    model.eval()
    all_preds: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())

    # Free model memory before next fold (helps on MPS especially)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return all_preds


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    examples = load_dataset()
    print(f"Loaded {len(examples)} examples — {class_distribution(examples)}")

    class_labels = sorted(set(ex.label for ex in examples))
    label_to_id = {lbl: i for i, lbl in enumerate(class_labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    texts = [ex.text for ex in examples]
    labels = [ex.label for ex in examples]
    label_ids = [label_to_id[l] for l in labels]

    folds = stratified_kfold_indices(labels, k=5, seed=SEED)
    device = get_device()
    print(f"Device: {device}, model: {MODEL_NAME}")

    fold_metrics: list[dict] = []
    aggregated_predictions: list[dict] = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [label_ids[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [label_ids[i] for i in test_idx]

        print(f"\n=== Fold {fold_i + 1}/5 (train={len(train_idx)}, test={len(test_idx)}) ===")
        preds = train_one_fold(
            train_texts, train_labels, test_texts, test_labels,
            num_labels=len(class_labels), device=device,
        )

        y_pred = [id_to_label[i] for i in preds]
        y_true = [id_to_label[i] for i in test_labels]
        m = score_predictions(y_true, y_pred, class_labels)
        m["fold"] = fold_i
        fold_metrics.append(m)
        for j, doc_idx in enumerate(test_idx):
            aggregated_predictions.append({
                "doc_id": examples[doc_idx].doc_id,
                "fold": fold_i,
                "y_true": y_true[j],
                "y_pred": y_pred[j],
            })
        print(f"  fold {fold_i}: acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")

    agg = aggregate_cv_metrics(fold_metrics)
    results = {
        "model": MODEL_NAME,
        "n_examples": len(examples),
        "k_folds": 5,
        "class_distribution": class_distribution(examples),
        "class_labels": class_labels,
        "hyperparameters": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "seed": SEED,
        },
        "fold_metrics": fold_metrics,
        "aggregate": agg,
        "predictions": aggregated_predictions,
    }

    json_path = OUTPUT_DIR / "finetune_distilbert_cv.json"
    md_path = OUTPUT_DIR / "finetune_distilbert_cv.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    write_markdown(results, md_path)
    print(
        f"\nFinal: acc={agg['accuracy_mean']:.3f}±{agg['accuracy_std']:.3f}  "
        f"macroF1={agg['macro_f1_mean']:.3f}±{agg['macro_f1_std']:.3f}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def write_markdown(results: dict, path: Path) -> None:
    a = results["aggregate"]
    lines = []
    lines.append(f"# Fine-Tuned Transformer — {results['model']}\n\n")
    lines.append(f"- N = {results['n_examples']} documents, k = {results['k_folds']} folds\n")
    lines.append("- Class distribution: " + ", ".join(
        f"{k}={v}" for k, v in sorted(results['class_distribution'].items())
    ) + "\n")
    hp = results["hyperparameters"]
    lines.append(
        f"- Hyperparameters: epochs={hp['epochs']}, batch={hp['batch_size']}, "
        f"lr={hp['lr']}, weight_decay={hp['weight_decay']}, "
        f"max_len={hp['max_length']}, warmup={hp['warmup_ratio']}, seed={hp['seed']}\n\n"
    )

    lines.append("## Aggregate metrics (mean ± std across folds)\n\n")
    lines.append("| Metric | Value |\n| --- | --- |\n")
    lines.append(f"| Accuracy | {a['accuracy_mean']:.3f} ± {a['accuracy_std']:.3f} |\n")
    lines.append(f"| Macro F1 | {a['macro_f1_mean']:.3f} ± {a['macro_f1_std']:.3f} |\n")
    lines.append(f"| Weighted F1 | {a['weighted_f1_mean']:.3f} ± {a['weighted_f1_std']:.3f} |\n")
    for lbl in results["class_labels"]:
        key = f"f1_{lbl}_mean"
        lines.append(f"| F1 ({lbl}) | {a.get(key, 0.0):.3f} ± {a.get(f'f1_{lbl}_std', 0.0):.3f} |\n")

    lines.append("\n## Per-fold metrics\n\n")
    lines.append("| Fold | Accuracy | Macro F1 | Weighted F1 |\n| --- | --- | --- | --- |\n")
    for m in results["fold_metrics"]:
        lines.append(
            f"| {m['fold']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
