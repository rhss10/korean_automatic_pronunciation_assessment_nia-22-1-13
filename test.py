import evaluate
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, load_from_disk
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# collator
def collate_fn(batch):
    return {
        "input_values": feature_extractor(
            [x["audio"]["array"] for x in batch],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values,
        "labels": [x[SCORE] for x in batch],
    }


def create_cm(preds, labels, name_labels, pcc, split):
    cm = confusion_matrix(preds, labels, labels=name_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        linewidth=0.2,
        cbar=False,
        xticklabels=name_labels,
        yticklabels=name_labels,
        annot_kws={"size": 5},
        ax=ax[0, 0],
    )
    ax.set(
        xlabel=f"{split} Pred Labels (PCC:{pcc:.2f})",
        ylabel=f"{split} Real Labels",
    )
    fig.savefig(f"{MODEL}/{SCORE}_{split}.pdf", bbox_inches="tight")
    fig.savefig(f"{MODEL}/{SCORE}_{split}.png", bbox_inches="tight")


# evaluation
def evaluate_metrics(dataloader, split):
    pcc_metric = evaluate.load("pearsonr")
    mse_metric = evaluate.load("mse")
    total_preds = []
    total_labels = []

    for x in tqdm(dataloader):
        with torch.no_grad():
            logits = model(
                x["input_values"].to(device),
            ).logits
        preds = torch.argmax(logits, dim=-1).item()
        pcc_metric.add(prediction=preds, reference=dataloader["labels"])
        mse_metric.add(prediction=preds, reference=dataloader["labels"])
        total_preds.extend(preds)
        total_labels.extend(dataloader["labels"])

    print("PCC:", pcc_metric.compute())
    print("MSE:", mse_metric.compute())
    create_cm(total_preds, total_labels, list(range(6)))


# load data, processor and model
VALID = "./data/nia-10_train/"
TEST = "./data/nia-10_train/"
BATCH_SIZE = 64
SCORE = "compreh"
MODEL = "./models/NIA_bat=8_lr=0.0001_warm=0.1_type=linear/"
valid_ds = load_from_disk(VALID)
test_ds = load_from_disk(TEST)
valid_dataloader = torch.utils.data.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, pin_memory=True
)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL)
model = AutoModelForAudioClassification.from_pretrained(MODEL, num_labels=6)
device = "cuda"
model.to(device)


print("BATCHSIZE:", BATCH_SIZE)
print("SCORE:", SCORE)
print("MODEL:", MODEL)

print("VALIDSET:", VALID)
evaluate_metrics(valid_dataloader, split="valid")
print("TESTSET:", TEST)
evaluate_metrics(test_dataloader, split="test")


print("- Test finished.")
