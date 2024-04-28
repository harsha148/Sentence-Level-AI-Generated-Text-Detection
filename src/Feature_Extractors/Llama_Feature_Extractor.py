import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

from src.Feature_Extractors.Base_Feature_Extractor import Base_Feature_Extractor
from src.utilities.feature_extractor_util import get_words, tokenwise_loss, get_bytes_to_words_mapping, \
    get_bytewise_loss, get_wordwise_loss_list


class Llama_Feature_Extractor(Base_Feature_Extractor):
    def __init__(self):
        super().__init__()
        model_path = ''
        print('Initializing the tokenizer')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path).to(self.device)
        print('Initialized tokenizer')
        print('Initializing the model for Llama')
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        print('Initialized the Llama model to memory')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.unk_token_id = self.tokenizer.unk_token_id
        self.byte_encoder = {i: f'<0x{i:02X}>' for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def extract_features(self, txt):
        """
        Extracts the features for the given text sequence based on the perplexities of the model Llama for the given sequence
        """
        tokens = self.tokenizer(txt, return_tensors='pt', truncation=True).to(self.device)
        input_token_ids = labels = tokens.input_ids[:, :1024, ].to(self.device)
        words = get_words(txt, True)
        outputs = self.model(input_token_ids)
        mean_loss, token_wise_loss_list = tokenwise_loss(outputs, labels)
        byte_wise_loss = get_bytewise_loss(input_token_ids, token_wise_loss_list, self.tokenizer,
                                           self.byte_decoder)
        bytes_list, bytes_to_words = get_bytes_to_words_mapping(words, self.byte_encoder)
        word_wise_loss = get_wordwise_loss_list(bytes_to_words, byte_wise_loss)
        return word_wise_loss
