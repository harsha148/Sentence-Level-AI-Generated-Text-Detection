import os
import json
import random
import re
import numpy as np
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_en_sentence(sentence, use_sp=False):
    pattern = re.compile(r'\S+|\s')
    words = pattern.findall(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words


def split_cn_sentence(sentence, use_sp=False):
    words = list(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words


def split_sentence(sentence, use_sp=False, cn_percent=0.2):
    total_char_count = len(sentence)
    total_char_count += 1 if total_char_count == 0 else 0
    chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
    if chinese_char_count / total_char_count > cn_percent:
        return split_cn_sentence(sentence, use_sp)
    else:
        return split_en_sentence(sentence, use_sp)


def save_dataset(fpath, data_samples):
    """
    Saves data samples to a file in JSON format.

    Args:
    fpath (str): The file path where the data should be saved.
    data_samples (list): A list of data samples to save.
    """
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            for sample in tqdm(data_samples, desc=f"Writing to {fpath}"):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logging.info("Data successfully saved to %s", fpath)
    except IOError as e:
        logging.error("Failed to save data to %s: %s", fpath, e)


def dataset_split_helper(data_path, train_path, test_path, train_ratio=0.9):
    """
    Splits data files into training and testing datasets.

    Args:
    data_path (str): Directory containing data files.
    train_path (str): Path to save the training dataset.
    test_path (str): Path to save the testing dataset.
    train_ratio (float): Proportion of data to include in the training set.
    """
    file_names = [file for file in os.listdir(data_path)]
    logging.info("The overall data sources: %s", file_names)

    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    data_samples = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                samples = [json.loads(line) for line in file]
                data_samples.extend(samples)
        except IOError as e:
            logging.error("Failed to process file %s: %s", file_path, e)
            continue

    splitting_index = int(len(data_samples) * train_ratio)
    train_data = data_samples[:splitting_index]
    test_data = data_samples[splitting_index:]

    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)

    logging.info("The number of train dataset samples: %d", len(train_data))
    logging.info("The number of test dataset samples: %d", len(test_data))


def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1

    return id2label