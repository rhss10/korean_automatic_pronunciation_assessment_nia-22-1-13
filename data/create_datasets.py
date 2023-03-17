import re

from datasets import Audio, concatenate_datasets, load_dataset, load_from_disk

# load csv files
train_ds = load_dataset(
    "csv",
    delimiter="\t",
    data_files="./train.txt",
    # column_names=["audio", "text", "comprehensibility_score"],
    split="train",
)
valid_ds = load_dataset(
    "csv",
    delimiter="\t",
    data_files="./valid.txt",
    # column_names=["audio", "text", "comprehensibility_score"],
    split="train",
)
test_ds = load_dataset(
    "csv",
    delimiter="\t",
    data_files="./test.txt",
    # column_names=["audio", "text", "comprehensibility_score"],
    split="train",
)


# make audio sampling rate into 16KHz and filter out audio frames bigger than 17 sec
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio", Audio(sampling_rate=16000))
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
train_ds = train_ds.filter(lambda x: len(x["audio"]["array"]) <= 272000)
valid_ds = valid_ds.filter(lambda x: len(x["audio"]["array"]) <= 272000)
test_ds = test_ds.filter(lambda x: len(x["audio"]["array"]) <= 272000)
valid_ds_small = valid_ds.train_test_split(0.1)
valid_ds_small = valid_ds_small["test"]

# save the final datasets
train_ds.save_to_disk("./train_ds")
valid_ds.save_to_disk("./valid_ds")
valid_ds_small.save_to_disk("./valid_ds_small")
test_ds.save_to_disk("./test_ds")

print("- Finished creating datasets")
