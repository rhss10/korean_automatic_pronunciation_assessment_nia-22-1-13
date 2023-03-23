from datasets import load_from_disk, load_dataset

SCORE = 'pronun'
eu_ds = load_dataset("csv", delimiter=",", data_files=f"/data1/nia13/eu/{SCORE}/test.csv", split="train")
eng_ds = load_dataset("csv", delimiter=",", data_files=f"/data1/nia13/eng/{SCORE}/test.csv", split="train")
asia_ds = load_dataset("csv", delimiter=",", data_files=f"/data1/nia13/asia/{SCORE}/test.csv", split="train")
cj_ds = load_dataset("csv", delimiter=",", data_files=f"/data1/nia13/cj/{SCORE}/test.csv", split="train")

eu_labels = {}
eng_labels = {}
asia_labels = {}
cj_labels = {}

for i in range(len(eu_ds)):
    eu_labels['{}'.format(eu_ds[i][SCORE])] = eu_labels.get('{}'.format(eu_ds[i][SCORE]), 0) + 1
for i in range(len(eng_ds)):
    eng_labels['{}'.format(eng_ds[i][SCORE])] = eng_labels.get('{}'.format(eng_ds[i][SCORE]), 0) + 1
for i in range(len(asia_ds)):
    asia_labels['{}'.format(asia_ds[i][SCORE])] = asia_labels.get('{}'.format(asia_ds[i][SCORE]), 0) + 1
for i in range(len(cj_ds)):
    cj_labels['{}'.format(cj_ds[i][SCORE])] = cj_labels.get('{}'.format(cj_ds[i][SCORE]), 0) + 1


print(f"==={SCORE}===")
print("EU:", eu_labels)
print("ENG:", eng_labels)
print("ASIA:", asia_labels)
print("CJ:", cj_labels)
