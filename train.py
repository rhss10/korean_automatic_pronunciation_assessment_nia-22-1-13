import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import torch
import torchaudio
from addict import Dict as addict
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    load_metric,
)
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)


def prepare_arguments():
    parser = argparse.ArgumentParser(
        description=("Train and evaluate the model."),
    )
    parser.add_argument("--num_labels", type=int, default=6)
    parser.add_argument("--train", type=str, default="./data/train_cj_ds/")
    parser.add_argument("--test", type=str, default="./data/valid_cj_ds_small/")
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="kresnik/wav2vec2-large-xlsr-korean",
        help="N/A",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_pearsonr")
    parser.add_argument(
        "--greater_is_better",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--exp_prefix",
        type=str,
        default="",
        help="Custom string to add to the experiment name.",
    )

    args = parser.parse_args()
    args.exp_name = f"{args.exp_prefix}_bat{args.per_device_batch_size}_lr{args.learning_rate}_warm{args.warmup_ratio}"
    args.save_dir_path = "./models/" + args.exp_name
    args.save_log_path = "./logs/" + args.exp_name
    os.makedirs(args.save_dir_path, exist_ok=True)
    os.makedirs(args.save_log_path, exist_ok=True)

    return args


def prepare_dataset(batch):
    # batched output is "un-batched"
    array = batch["audio"]["array"]
    batch["input_values"] = feature_extractor(array, sampling_rate=16000).input_values[
        0
    ]

    return batch


def evaluate_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    res = mse_metric.compute(predictions=pred_ids, references=pred.label_ids)
    res.update(**pcc_metric.compute(predictions=pred_ids, references=pred.label_ids))

    return res


def prepare_trainer(args, feature_extractor, train_ds, test_ds):
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name_or_path, num_labels=args.num_labels
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=args.save_dir_path,
        logging_dir=args.save_log_path,
        group_by_length=False,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=3,
        push_to_hub=False,
        load_best_model_at_end=True,
        greater_is_better=args.greater_is_better,
        metric_for_best_model=args.metric_for_best_model,
        dataloader_num_workers=15,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=evaluate_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
    )

    return trainer


if __name__ == "__main__":
    args = prepare_arguments()
    print(args)

    TRAIN_DS = args.train
    TEST_DS = args.test
    print(TRAIN_DS, TEST_DS, sep="\n")
    train_ds = load_from_disk(TRAIN_DS)
    valid_ds = load_from_disk(TEST_DS)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)

    train_ds = train_ds.map(prepare_dataset)
    valid_ds = valid_ds.map(prepare_dataset)
    train_ds = train_ds.rename_column("compreh", "labels")
    valid_ds = valid_ds.rename_column("compreh", "labels")

    mse_metric = load_metric("mse")
    pcc_metric = load_metric("pearsonr")

    trainer = prepare_trainer(
        args, feature_extractor, train_ds=train_ds, test_ds=valid_ds
    )
    train_res = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_res.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(valid_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print("- Training complete.")
