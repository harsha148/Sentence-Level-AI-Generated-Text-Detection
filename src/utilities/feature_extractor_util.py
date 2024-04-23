import re

import torch
from torch.nn import CrossEntropyLoss


def get_words(text: str, is_llama: bool = False) -> list[str]:
    pattern = re.compile(r'\S+|\s')
    words = pattern.findall(text)
    if is_llama:
        words = ["▁" if item == " " else item for item in words]
    return words


def split_sentence(sentence, use_sp=False):
    total_char_count = len(sentence)
    total_char_count += 1 if total_char_count == 0 else 0
    return get_words(sentence, use_sp)


def tokenwise_loss(outputs, labels):
    logits = outputs.logits.squeeze()
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels.squeeze()[1:].contiguous()
    cross_entropy_loss = CrossEntropyLoss(reduction='none')
    loss_list = cross_entropy_loss(shifted_logits, shifted_labels)
    return loss_list.mean().item(), loss_list


def get_bytewise_loss(token_ids, loss_list, tokenizer, unicode_to_byte):
    # fetching tokens based on the token id
    token_ids = token_ids.squeeze()
    tokens = [
        tokenizer._convert_id_to_token(input_id)
        for input_id in token_ids
    ]
    print(f'Tokens for the input text')
    print(tokens)
    bytewise_loss = []
    # setting the loss for the first subword that is tokenized as 0 (as we calculate the probabilities based as a causal
    # LM, and we won't be predicting the first word.
    # calculating the number of bytes in the first subword
    try:
        byte_list_for_token = [unicode_to_byte[i] for i in tokens[0]]
        # mapping the loss for the first n bytes as 0, where n is the number of bytes for the first subword
        bytewise_loss.extend([0 for i in range(len(byte_list_for_token))])
        for i in range(len(tokens) - 1):
            byte_list_for_ith_word = [unicode_to_byte[c] for c in tokens[i + 1]]
            bytewise_loss.extend([loss_list[i] for j in range(len(byte_list_for_ith_word))])
    except:
        print('Exception while trying to create a list of the bytewise loss')
    return bytewise_loss


def get_bytes_to_words_mapping(words, byte_encoder):
    bytes_to_words = []
    bytes_list = []
    for i in range(len(words)):
        bytes_for_word = [byte_encoder[b] for b in words[i].encode('utf-8')]
        bytes_list.extend(bytes_for_word)
        bytes_to_words.extend([i for j in range(len(bytes_for_word))])
    return bytes_list, bytes_to_words


def get_wordwise_loss_list(bytes_to_words, bytewise_loss):
    wordwise_loss = []
    token_start_index = 0
    while token_start_index < len(bytes_to_words):
        token_end_index = token_start_index + 1
        while (token_end_index < len(bytes_to_words) and bytes_to_words[token_end_index] ==
               bytes_to_words[token_start_index]):
            token_end_index += 1
        if token_end_index > len(bytewise_loss):
            break
        token_loss = bytewise_loss[token_start_index:token_end_index]
        wordwise_loss.append(torch.Tensor(token_loss).mean().item())
        token_start_index = token_end_index
    return wordwise_loss


def get_begin_word_idx(self, input_ids, bbs_to_words):
    input_ids = input_ids.squeeze()
    begin_token = self.base_tokenizer._convert_id_to_token(input_ids[0])
    byte_list = [self.byte_decoder[c] for c in begin_token]
    begin_word_idx = bbs_to_words[len(byte_list) - 1] + 1
    return begin_word_idx
