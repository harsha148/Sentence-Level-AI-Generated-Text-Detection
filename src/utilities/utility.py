from src.Feature_Extractors.GPT2_Feature_Extractor import GPT2_Feature_Extractor
from src.Feature_Extractors.GPTJ_Feature_Extractor import GPTJ_Feature_Extractor
from src.Feature_Extractors.GPTNeo_Feature_Extractor import GPTNeo_Feature_Extractor
from src.Feature_Extractors.Llama_Feature_Extractor import Llama_Feature_Extractor
from src.utilities.Model_Enum import Model_Enum


def get_model_by_enum(model: Model_Enum) -> object:
    models_by_enum = {
        Model_Enum.GPT2: GPT2_Feature_Extractor(),
        Model_Enum.GPTJ: GPTJ_Feature_Extractor(),
        Model_Enum.GPTNEO: GPTNeo_Feature_Extractor(),
        Model_Enum.LLAMA: Llama_Feature_Extractor()
    }
    return models_by_enum[model]
