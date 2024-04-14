import torch
import transformers

from src.Feature_Extractors.Base_Feature_Extractor import Base_Feature_Extractor


class GPT2_Feature_Extractor(Base_Feature_Extractor):
    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-xl')
        self.model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl')

    def extract_features(self, txt):
        # extracts the features for the given text sequence based on the perplexities of the model GPT-2 for the given
        # sequence
        tokens = self.tokenizer(txt, return_tensors='pt').to(self.device)
        print('Generated tokens for the given text:')
        print(tokens)
