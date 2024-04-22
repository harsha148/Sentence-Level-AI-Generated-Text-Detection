import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from collections import Counter
import nltk
from src.utilities.utils import split_sentence


class Evaluator:
    def __init__(self, data, model, en_labels, id2label, device):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label = id2label
        self.device = device

    def test(self, content_level_eval=False):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []
        for step, inputs in enumerate(tqdm(self.data.test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'])
                logits = output['logits']
                preds = output['preds']

                texts.extend(inputs['text'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                total_logits.extend(logits.cpu().tolist())

        if content_level_eval:
            print("*" * 8, "Content Level Evaluation", "*" * 8)
            return self.content_level_eval(texts, true_labels, pred_labels)
        else:
            print("*" * 8, "Sentence Level Evaluation", "*" * 8)
            return self.sent_level_eval(texts, true_labels, pred_labels)

    def content_level_eval(self, texts, true_labels, pred_labels):
        true_content_labels = []
        pred_content_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            true_common_tag = self._get_most_common_tag(true_label)
            pred_common_tag = self._get_most_common_tag(pred_label)
            true_content_labels.append(self.en_labels[true_common_tag[0]])
            pred_content_labels.append(self.en_labels[pred_common_tag[0]])
        return self._get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels)

    def sent_level_eval(self, texts, true_labels, pred_labels):
        true_sent_labels = []
        pred_sent_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_sent_label = self.get_sent_label(text, true_label)
            pred_sent_label = self.get_sent_label(text, pred_label)
            true_sent_labels.extend(true_sent_label)
            pred_sent_labels.extend(pred_sent_label)
        return self._get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels)

    def get_sent_label(self, text, label):
        sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_separator.tokenize(text)
        offset = 0
        sent_label = []
        for sent in sents:
            start = text[offset:].find(sent) + offset
            end = start + len(sent)
            offset = end
            split_sent = split_sentence(self.data)
            end_word_idx = len(split_sent(text[:end]))
            if end_word_idx > self.data.seq_len:
                break
            word_num = len(split_sent(text[start:end]))
            start_word_idx = end_word_idx - word_num
            tags = label[start_word_idx:end_word_idx]
            most_common_tag = self._get_most_common_tag(tags)
            sent_label.append(most_common_tag[0])
        return sent_label

    def _get_most_common_tag(self, tags):
        tags = [self.id2label[tag] for tag in tags if tag != -1]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]
        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')
        print(f"Accuracy: {accuracy * 100:.1f}%")
        print(f"Macro F1 Score: {macro_f1 * 100:.1f}%")
        print("Precision/Recall per class: ")
        precision_recall = ' '.join([f"{p * 100:.1f}/{r * 100:.1f}" for p, r in zip(precision, recall)])
        print(precision_recall)
        return {"precision": precision, "recall": recall, "accuracy": accuracy, "macro_f1": macro_f1}
