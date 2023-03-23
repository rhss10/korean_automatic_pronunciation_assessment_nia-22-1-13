import re

from datasets import Audio, concatenate_datasets, load_dataset, load_from_disk

COUNTRY = "cj"
print(COUNTRY)

# load csv files
train_ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"/data1/nia13/{COUNTRY}/compreh/train.csv",
    # column_names=["audio", "text", "comprehensibility_score"],
    split="train",
)
valid_ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"/data1/nia13/{COUNTRY}/compreh/val.csv",
    # column_names=["audio", "text", "comprehensibility_score"],
    split="train",
)
test_ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"/data1/nia13/{COUNTRY}/compreh/test.csv",
    # column_names=["audio", "text", "comprehensibility_score"],
    split="train",
)


# make audio sampling rate into 16KHz and filter out audio frames bigger than 17 sec
train_ds = train_ds.map(
    lambda x: {
        "path": "/data2/rhss10/speech_corpus/NIA-13/nia13"
        + x["path"].split("nia13")[-1]
    }
)
valid_ds = valid_ds.map(
    lambda x: {
        "path": "/data2/rhss10/speech_corpus/NIA-13/nia13"
        + x["path"].split("nia13")[-1]
    }
)
test_ds = test_ds.map(
    lambda x: {
        "path": "/data2/rhss10/speech_corpus/NIA-13/nia13"
        + x["path"].split("nia13")[-1]
    }
)

train_ds = train_ds.map(lambda x: {"audio": x["path"]})
valid_ds = valid_ds.map(lambda x: {"audio": x["path"]})
test_ds = test_ds.map(lambda x: {"audio": x["path"]})
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio", Audio(sampling_rate=16000))
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
train_ds = train_ds.filter(lambda x: len(x["audio"]["array"]) <= 272000, num_proc=15)
valid_ds = valid_ds.filter(lambda x: len(x["audio"]["array"]) <= 272000, num_proc=15)
test_ds = test_ds.filter(lambda x: len(x["audio"]["array"]) <= 272000, num_proc=15)
valid_ds_small = valid_ds.train_test_split(0.1)
valid_ds_small = valid_ds_small["test"]

# save the final datasets
train_ds.save_to_disk(f"./train_{COUNTRY}_ds", num_proc=15, max_shard_size="5GB")
valid_ds.save_to_disk(f"./valid_{COUNTRY}_ds", num_proc=15, max_shard_size="5GB")
valid_ds_small.save_to_disk(f"./valid_{COUNTRY}_ds_small")
test_ds.save_to_disk(f"./test_{COUNTRY}_ds", num_proc=15, max_shard_size="5GB")

print("- Finished creating datasets")
