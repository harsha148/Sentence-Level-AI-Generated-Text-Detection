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

    def evaluate_model(self, content_level_eval=False):
        self.model.eval()
        document_texts = []
        actual_labels = []
        predicted_labels = []
        all_logits = []
        for step, batch in enumerate(tqdm(self.dataset.test_dataloader, desc="Evaluating")):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            with torch.no_grad():
                labels = batch['labels']
                output = self.model(batch['features'], batch['labels'])
                logits = output['logits']
                predictions = output['preds']

                document_texts.extend(batch['text'])
                predicted_labels.extend(predictions.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
                all_logits.extend(logits.cpu().tolist())

        if content_level_eval:
            print("**** Content Level Evaluation ****")
            return self.evaluate_content_level(document_texts, actual_labels, predicted_labels)
        else:
            print("**** Sentence Level Evaluation ****")
            return self.evaluate_sentence_level(document_texts, actual_labels, predicted_labels)

    def evaluate_content_level(self, texts, actual_labels, predicted_labels):
        content_actual_labels = []
        content_predicted_labels = []
        for text, actual, predicted in zip(texts, actual_labels, predicted_labels):
            actual = np.array(actual)
            predicted = np.array(predicted)
            valid_mask = actual != -1
            actual = actual[valid_mask].tolist()
            predicted = predicted[valid_mask].tolist()
            actual_major_tag = self.get_most_common_tag(actual)
            predicted_major_tag = self.get_most_common_tag(predicted)
            content_actual_labels.append(self.en_labels[actual_major_tag[0]])
            content_predicted_labels.append(self.en_labels[predicted_major_tag[0]])
        return self.calculate_metrics(content_actual_labels, content_predicted_labels)

    def evaluate_sentence_level(self, texts, actual_labels, predicted_labels):
        sentence_actual_labels = []
        sentence_predicted_labels = []
        for text, actual, predicted in zip(texts, actual_labels, predicted_labels):
            sentence_actual_labels.extend(self.extract_sentence_labels(text, actual))
            sentence_predicted_labels.extend(self.extract_sentence_labels(text, predicted))
        return self.calculate_metrics(sentence_actual_labels, sentence_predicted_labels)

    def extract_sentence_labels(self, text, labels):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        offset = 0
        sentence_labels = []
        for sentence in sentences:
            start = text[offset:].find(sentence) + offset
            end = start + len(sentence)
            offset = end
            split_function = split_sentence(self.data)
            last_word_index = len(split_function(text[:end]))
            if last_word_index > self.data.seq_len:
                break
            num_words = len(split_function(text[start:end]))
            first_word_index = last_word_index - num_words
            relevant_tags = labels[first_word_index:last_word_index]
            most_common_tag = self.get_most_common_tag(relevant_tags)
            sentence_labels.append(most_common_tag[0])
        return sentence_labels

    def get_most_common_tag(self, tags):
        tags = [self.id2label[tag] for tag in tags if tag != -1]
        tag_count = Counter(tags)
        most_common_tag = tag_count.most_common(1)[0]
        return most_common_tag

    def calculate_metrics(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        print(f"Accuracy: {accuracy * 100:.1f}%")
        print(f"Macro F1 Score: {macro_f1 * 100:.1f}%")
        print("Precision/Recall per class:")
        precision_recall = ' '.join([f"{p * 100:.1f}/{r * 100:.1f}" for p, r in zip(precision, recall)])
        print(precision_recall)
        return {"precision": precision, "recall": recall, "accuracy": accuracy, "macro_f1": macro_f1}
