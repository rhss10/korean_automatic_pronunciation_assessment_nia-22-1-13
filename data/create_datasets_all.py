import re

from datasets import Audio, concatenate_datasets, load_dataset, load_from_disk

COUNTRY = "eng"
SPLIT = "val"
CUSTOM = "noRW_BAL"
# load csv files
ds = load_dataset(
    "csv",
    delimiter=",",
    data_files=f"{SPLIT}_compreh_{COUNTRY}_{CUSTOM}.csv",
    # column_names=["audio", "text", "comprehensibility_score"],
    split="train",
)


# make audio sampling rate into 16KHz and filter out audio frames bigger than 17 sec
ds = ds.map(
    lambda x: {
        "path": "/data2/rhss10/speech_corpus/NIA-13/nia13"
        + x["path"].split("nia13")[-1]
    }
)
ds = ds.map(lambda x: {"audio": x["path"]})
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
ds = ds.filter(lambda x: len(x["audio"]["array"]) <= 272000, num_proc=15)
ds.save_to_disk(f"./{SPLIT}_{COUNTRY}_{CUSTOM}_ds", num_proc=15, max_shard_size="5GB")

print("- Finished creating datasets")
