import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from collections import Counter
import nltk
from src.utilities.feature_extractor_util import split_sentence


class Evaluator:
    def __init__(self, data, model, en_labels, id2label, seq_len, device):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label = id2label
        self.seq_len = seq_len
        self.device = device

    def evaluate_model(self, document_level_eval=False):
        self.model.eval()
        document_texts = []
        actual_labels = []
        predicted_labels = []
        all_logits = []
        for step, batch in enumerate(tqdm(self.data.test_dataloader, desc="Evaluating")):
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

        if document_level_eval:
            print("**** Document Level Evaluation ****")
            return self.evaluate_document_level(document_texts, actual_labels, predicted_labels)
        else:
            print("**** Sentence Level Evaluation ****")
            return self.evaluate_sentence_level(document_texts, actual_labels, predicted_labels)

    def evaluate_document_level(self, texts, actual_labels, predicted_labels):
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
        return self.calculate_metrics(content_actual_labels, content_predicted_labels, self.en_labels)

    def evaluate_sentence_level(self, texts, actual_labels, predicted_labels):
        sentence_actual_labels = []
        sentence_predicted_labels = []
        for text, actual, predicted in zip(texts, actual_labels, predicted_labels):
            sentence_actual_labels.extend(self.extract_sentence_labels(text, actual))
            sentence_predicted_labels.extend(self.extract_sentence_labels(text, predicted))
        true_sent_labels = [self.en_labels[label] for label in sentence_actual_labels]
        pred_sent_labels = [self.en_labels[label] for label in sentence_predicted_labels]
        return self.calculate_metrics(true_sent_labels, pred_sent_labels, self.en_labels)

    def extract_sentence_labels(self, text, labels):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        offset = 0
        sentence_labels = []
        for sentence in sentences:
            start = text[offset:].find(sentence) + offset
            end = start + len(sentence)
            offset = end
            # Use split_sentence directly on the text segment
            sentence_tokens = split_sentence(text[start:end])  # This should return a list of words/tokens
            last_word_index = len(split_sentence(text[:end]))  # This should also return a list, then take its length
            if last_word_index > self.seq_len:
                break
            num_words = len(sentence_tokens)
            first_word_index = last_word_index - num_words
            relevant_tags = labels[first_word_index:last_word_index]
            most_common_tag = self.get_most_common_tag(relevant_tags)
            sentence_labels.append(most_common_tag[0])
        return sentence_labels

    def get_most_common_tag(self, tags):
        tags = [self.id2label[tag] for tag in tags]
        tags = [tag.split('-')[-1] for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]
        return most_common_tag

    """
        Uncomment the below and use as the calculate_metrics for binary classification
        Comment it for mixed model multi-class AIGT detection and for particular model binary AIGT detection
    """

    # def calculate_metrics(self, true_labels, predicted_labels, labels_dict):
    #     # Reverse the labels_dict to get a dictionary from numbers to labels
    #     number_to_label = {v: k for k, v in labels_dict.items()}
    #
    #     # Convert numeric labels to 'AI' or 'Human'
    #     true_labels_converted = ['Human' if number_to_label[label] == 'human' else 'AI' for label in true_labels]
    #     predicted_labels_converted = ['Human' if number_to_label[label] == 'human' else 'AI' for label in
    #                                   predicted_labels]
    #
    #     # Calculate the metrics
    #     accuracy = accuracy_score(true_labels_converted, predicted_labels_converted)
    #     precision = precision_score(true_labels_converted, predicted_labels_converted, average=None,
    #                                 labels=['AI', 'Human'])
    #     recall = recall_score(true_labels_converted, predicted_labels_converted, average=None, labels=['AI', 'Human'])
    #     f1 = f1_score(true_labels_converted, predicted_labels_converted, average=None, labels=['AI', 'Human'])
    #     # Prepare the data for display
    #     metrics_df = pd.DataFrame({
    #         'Class': ['AI', 'Human'],
    #         'Precision': precision,
    #         'Recall': recall,
    #         'F1-score': f1
    #     })
    #
    #     # Print formatted table
    #     print(metrics_df.to_string(index=False))
    #
    #     return {"precision": precision, "recall": recall, "accuracy": accuracy,
    #             "macro_f1": f1_score(true_labels_converted, predicted_labels_converted, average='macro')}
    """
        Use the below calculate_metrics method for mixed model multi-class AIGT detection 
        Also for particular model binary AIGT detection
        Comment it out for mixed model binary AIGT detection     
    """
    def calculate_metrics(self, true_labels, predicted_labels, labels_dict):
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average=None)
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
        precision = precision_score(true_labels, predicted_labels, average=None)
        recall = recall_score(true_labels, predicted_labels, average=None)

        metrics_df = pd.DataFrame({
            'Class': [key for key in labels_dict.keys()],
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        })

        print("Evaluation Metrics:")
        print(metrics_df.to_string(index=False))

        return {"precision": precision, "recall": recall, "accuracy": accuracy, "macro_f1": macro_f1}

