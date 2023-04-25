import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from addict import Dict as addict
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    load_metric,
)
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# "/data1/rhss10/speech_corpus/huggingface_dataset/NIA-13/train_ds/"
# "/data1/rhss10/speech_corpus/huggingface_dataset/NIA-13/valid_ds_small/"
# "./checkpoints/eu-checkpoints/"
# "kresnik/wav2vec2-large-xlsr-korean"

_HIDDEN_STATES_START_POSITION = 2


def prepare_arguments():
    parser = argparse.ArgumentParser(description=("Train and evaluate the model."))
    parser.add_argument("--num_labels", type=int, default=5)
    parser.add_argument("--train", type=str, default="./data/train_asia_all_ds")
    parser.add_argument("--test", type=str, default="./data/val_asia_all_ds")
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument(
        "--model_name_or_path", type=str, default="kresnik/wav2vec2-large-xlsr-korean"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_pearsonr")
    parser.add_argument("--greater_is_better", type=bool, default=True)
    parser.add_argument(
        "--exp_prefix",
        type=str,
        default="",
        help="Custom string to add to the experiment name.",
    )
    parser.add_argument("--loss_fcn", type=str, default="CE")
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    args.total_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    args.exp_name = f"{args.exp_prefix}_bat{args.total_batch}_lr{args.learning_rate}_warm{args.warmup_ratio}_loss{args.loss_fcn}"
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


class Wav2Vec2ForMSE(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = (
            config.num_hidden_layers + 1
        )  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(
                -1, 1
            )

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            onehot_labels = F.one_hot(labels, num_classes=self.config.num_labels)
            loss = loss_fct(logits, onehot_labels.to(torch.float))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class WeightedSamplingTrainer(Trainer):
    """
    Custom Trainer class

    The model should return tuples or subclasses of ModelOutput.
    (1) your model can compute the loss if a labels argument is provided and that loss is returned as the first element of the tuple (if your model returns tuples)
    (2) your model can accept multiple label arguments (use the label_names in your TrainingArguments to indicate their name to the Trainer) but none of them should be named "label".
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        # print(inputs)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (NIA-1-13 APA datasets have 5 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([25.0, 5.0, 1.5, 1.0, 3.0]).cuda()
        )
        # print(logits)
        # print(labels)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def prepare_trainer(args, feature_extractor, train_ds, test_ds, label2id, id2label):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        # label2id=label2id,
        # id2label=id2label,
        # finetuning_task="audio-classification",
    )
    if args.loss_fcn == "CE":
        model = AutoModelForAudioClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    elif args.loss_fcn == "MSE":
        print("- Using MSE Model")
        model = Wav2Vec2ForMSE.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    model.freeze_feature_extractor()
    print(model.config)

    training_args = TrainingArguments(
        output_dir=args.save_dir_path,
        logging_dir=args.save_log_path,
        group_by_length=args.group_by_length,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=3,
        push_to_hub=False,
        load_best_model_at_end=True,
        greater_is_better=args.greater_is_better,
        metric_for_best_model=args.metric_for_best_model,
        dataloader_num_workers=15,
    )

    print(training_args.to_dict())

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=evaluate_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
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

    train_ds = train_ds.map(prepare_dataset, num_proc=16)
    valid_ds = valid_ds.map(prepare_dataset, num_proc=16)
    train_ds = train_ds.rename_column("compreh", "label")
    valid_ds = valid_ds.rename_column("compreh", "label")

    mse_metric = load_metric("mse")
    pcc_metric = load_metric("pearsonr")
    label2id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    id2label = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    train_ds = train_ds.map(lambda x: {"label": label2id[x["label"]]}, num_proc=16)
    valid_ds = valid_ds.map(lambda x: {"label": label2id[x["label"]]}, num_proc=16)

    trainer = prepare_trainer(
        args,
        feature_extractor,
        train_ds=train_ds,
        test_ds=valid_ds,
        label2id=label2id,
        id2label=id2label,
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
