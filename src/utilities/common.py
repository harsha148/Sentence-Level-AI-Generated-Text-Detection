import logging
import re

from src.Feature_Extractors.GPT2_Feature_Extractor import GPT2_Feature_Extractor
from src.Feature_Extractors.GPTJ_Feature_Extractor import GPTJ_Feature_Extractor
from src.Feature_Extractors.GPTNeo_Feature_Extractor import GPTNeo_Feature_Extractor
from src.Feature_Extractors.Llama_Feature_Extractor import Llama_Feature_Extractor
import sys


def get_model_by_enum(model: str) -> object:
    if model == 'GPT2':
        return GPT2_Feature_Extractor()
    elif model == 'GPTJ':
        return GPTJ_Feature_Extractor()
    elif model == 'GPTNEO':
        return GPTNeo_Feature_Extractor()
    elif model == 'LLAMA':
        return Llama_Feature_Extractor()
    logging.error('Invalid Model type')
    sys.exit(0)






