import torch
import transformers

from src.Feature_Extractors.Base_Feature_Extractor import Base_Feature_Extractor


class GPTNeo_Feature_Extractor(Base_Feature_Extractor):
    def __init__(self):
        super().__init__()
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')

    def extract_features(self, txt):
        # extracts the features for the given text sequence based on the perplexities of the model GPTNeo for the given
        # sequence
        tokens = self.base_tokenizer(txt, return_tensors='pt').to(self.device)
        print('Generated tokens for the given text:')
        print(tokens)
