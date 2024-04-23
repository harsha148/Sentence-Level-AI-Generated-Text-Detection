import torch
import transformers
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from src.Feature_Extractors.Base_Feature_Extractor import Base_Feature_Extractor
from src.utilities.feature_extractor_util import tokenwise_loss, get_bytewise_loss, get_words, \
    get_bytes_to_words_mapping, get_wordwise_loss_list


class GPTNeo_Feature_Extractor(Base_Feature_Extractor):
    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {unicode_val: byte_key for byte_key, unicode_val in self.byte_encoder.items()}

    def extract_features(self, txt):
        """
        Extracts the features for the given text sequence based on the perplexities of the model GPTNeo for the given sequence
        """
        tokens = self.tokenizer(txt, return_tensors='pt').to(self.device)
        input_token_ids = labels = tokens.input_ids[:, :1024, ]
        outputs = self.model(input_token_ids)
        mean_loss, token_wise_loss_list = tokenwise_loss(outputs, labels)
        bytewise_loss_list = get_bytewise_loss(input_token_ids, token_wise_loss_list, self.tokenizer,
                                               self.byte_decoder)
        words = get_words(txt)
        bytes_list, bytes_to_words = get_bytes_to_words_mapping(words, self.byte_encoder)
        wordwise_loss_list = get_wordwise_loss_list(bytes_to_words, bytewise_loss_list)
        print('Extracted word wise loss list for the sentence given')
        print(wordwise_loss_list)
        return wordwise_loss_list