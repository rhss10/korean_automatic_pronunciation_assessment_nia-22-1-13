from datasets import load_dataset, load_from_disk

SCORE = "compreh"
SPLIT = "test"
eu_ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"/data1/nia13/eu/{SCORE}/{SPLIT}.csv",
    split="train",
)
eng_ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"/data1/nia13/eng/{SCORE}/{SPLIT}.csv",
    split="train",
)
asia_ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"/data1/nia13/asia/{SCORE}/{SPLIT}.csv",
    split="train",
)
cj_ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"/data1/nia13/cj/{SCORE}/{SPLIT}.csv",
    split="train",
)

eu_labels = {}
eng_labels = {}
asia_labels = {}
cj_labels = {}

for i in range(len(eu_ds)):
    eu_labels[eu_ds[i][SCORE]] = eu_labels.get(eu_ds[i][SCORE], 0) + 1
for i in range(len(eng_ds)):
    eng_labels[eng_ds[i][SCORE]] = eng_labels.get(eng_ds[i][SCORE], 0) + 1
for i in range(len(asia_ds)):
    asia_labels[asia_ds[i][SCORE]] = asia_labels.get(asia_ds[i][SCORE], 0) + 1
for i in range(len(cj_ds)):
    cj_labels[cj_ds[i][SCORE]] = cj_labels.get(cj_ds[i][SCORE], 0) + 1


print(f"==={SCORE} {SPLIT}===")
print("EU:", eu_labels)
print("ENG:", eng_labels)
print("ASIA:", asia_labels)
print("CJ:", cj_labels)
