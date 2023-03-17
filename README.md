## General
- A source code to fine-tune self-supervised learning model (SSL) on NIA-2022-1-13 Non-native L2 Korean Dataset for Automatic Pronunciation Assessment (APA).
- NIA-2022-1-13 Non-native L2 Korean Dataset for Automatic Pronunciation Assessment (APA) will soon be released within 2023.
- More information regarding the **usage of the dataset** and **docker support** will be updated with the relase of dataset.

## License
- SPDX-FileCopyrightText: © 2023 Hyungshin Ryu \<rhss10@snu.ac.kr\>
- SPDX-License-Identifier: Apache-2.0

## Notes
- NIA-2022-1-13 Non-native L2 Korean Dataset supports proficiency scores of 3 aspects, **'comprehensibility'**, **'fluency'**, **'accentedness'**.
- The example code is aimed at scoring **'comprehensibility'**.
- By changing the data/preprocess_data.py code, you may asess **'fluency'** or **'accentedness'** scores.

## Commands
### Prepare Data
```python
# Data processing should be done with the ACTUAL data path
python preprocess_data.py
# create Huggingface-based datasets arrows.
python create_datasets.py
```
### Train
```python
# Example command for training. For more supported arguments, please refer to train.py
python train.py --exp_prefix NIA
```
### Test
```python
# Example
python test.py
```
