import os
import torch
import json
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, DatasetDict
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from src.utilities.feature_extractor_util import split_sentence
from src.utilities.utils import set_seed


class DataHandler:

    def __init__(self, train_path, test_path, batch_size, max_len, human_label, id2label, word_pad_idx=0,
                 label_pad_idx=-1):
        set_seed(0)
        self.batch_size = batch_size
        self.max_len = max_len
        self.human_label = human_label
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

        dataset = {}

        if train_path:
            train_dict = self.prepare_dataset(train_path)
            dataset["train"] = Dataset.from_dict(train_dict)
        if test_path:
            test_dict = self.prepare_dataset(test_path)
            dataset["test"] = Dataset.from_dict(test_dict)

        self.datasets_dict = DatasetDict(dataset)
        if train_path:
            self.train_dataloader = self.create_dataloader(self.datasets_dict["train"], is_train=True)
        if test_path:
            self.test_dataloader = self.create_dataloader(self.datasets_dict["test"])

    def prepare_dataset(self, data_path, save_dir=''):
        processed_data_filename = Path(data_path).stem + "_processed.pkl"
        processed_data_path = os.path.join(save_dir, processed_data_filename)

        with open(data_path, 'r') as f:
            if data_path.endswith('json'):
                samples = json.load(f)
            else:
                samples = [json.loads(line) for line in f]

        samples_dict = {'features': [], 'prompt_len': [], 'label': [], 'text': []}
        for sample in tqdm(samples, desc="Processing Data"):
            if len(sample["wordwise_loss_list"]) == 0 or len(sample["wordwise_loss_list"][0]) == 0:
                continue
            self.process_sample(sample, samples_dict)

        return samples_dict

    def process_sample(self, sample, samples_dict):
        text, label = sample['text'], sample['label']
        prompt_len, label_int = sample['prompt_len'], sample['label_int']
        wordwise_loss_list = sample['wordwise_loss_list']

        features_per_word = list(zip(*wordwise_loss_list))
        processed_features = [list(word_features) for word_features in features_per_word]

        samples_dict['features'].append(processed_features)
        samples_dict['prompt_len'].append(prompt_len)
        samples_dict['label'].append(label)
        samples_dict['text'].append(text)

        return samples_dict

    def create_dataloader(self, dataset, is_train=False):
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=self.collate_data)

    def collate_data(self, samples):
        batch = {}

        features = [sample['features'] for sample in samples]
        prompt_len = [sample['prompt_len'] for sample in samples]
        text = [sample['text'] for sample in samples]
        label = [sample['label'] for sample in samples]

        features, masks = self.tensorize_and_pad(features)
        pad_masks = (1 - masks) * self.label_pad_idx

        for idx, p_len in enumerate(prompt_len):
            self.process_text_segment(idx, p_len, text, label, masks, pad_masks)

        batch['features'] = features
        batch['labels'] = masks
        batch['text'] = text

        return batch

    def tensorize_and_pad(self, data):
        # Determine feature dimension and prepare padding
        max_len = self.max_len
        feat_dim = len(data[0][0])
        padded_data = [seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data]
        padded_data = [seq[:max_len] for seq in padded_data]

        # Create masks for sequence lengths
        masks = [[1] * min(len(seq), max_len) + [0] * (max_len - min(len(seq), max_len)) for seq in data]

        tensor_data, tensor_mask = torch.tensor(padded_data, dtype=torch.float), torch.tensor(masks, dtype=torch.long)

        return tensor_data, tensor_mask

    def process_text_segment(self, idx, p_len, text, label, masks, pad_masks):
        prefix_len = len(split_sentence(text[idx][:p_len]))
        if prefix_len > self.max_len:
            prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
            masks[idx][:] = prefix_ids[:]
            return
        total_len = len(split_sentence(text[idx]))

        if prefix_len > 0:
            prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
            masks[idx][:prefix_len] = prefix_ids[:]
        if total_len - prefix_len > 0:
            if total_len > self.max_len:
                human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, label[idx])
            else:
                human_ids = self.sequence_labels_to_ids(total_len - prefix_len, label[idx])
            masks[idx][prefix_len:total_len] = human_ids[:]
        masks[idx] += pad_masks[idx]

    def sequence_labels_to_ids(self, seq_len, label):
        # Convert sequence lengths to label IDs
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return None
        elif seq_len == 1:
            label = 'S-' + label
            return torch.tensor([self.label2id[label]], dtype=torch.long)
        else:
            ids = [self.label2id['B-' + label]]
            ids.extend([self.label2id['M-' + label]] * (seq_len - 2))
            ids.append(self.label2id['E-' + label])
            return torch.tensor(ids, dtype=torch.long)
