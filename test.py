import evaluate
import torch
from datasets import load_dataset, load_from_disk
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


# evaluation
def test(dataloader):
    pcc_metric = evaluate.load("pearsonr")
    mse_metric = evaluate.load("mse")
    for x in tqdm(dataloader):
        with torch.no_grad():
            logits = model(
                x["input_values"].to(device),
            ).logits
        preds = torch.argmax(logits, dim=-1).item()
        pcc_metric.add(prediction=preds, reference=dataloader["labels"])
        mse_metric.add(prediction=preds, reference=dataloader["labels"])
    print("PCC:", pcc_metric.compute())
    print("MSE:", mse_metric.compute())


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
test(valid_dataloader)
print("TESTSET:", TEST)
test(test_dataloader)


print("- Test finished.")
