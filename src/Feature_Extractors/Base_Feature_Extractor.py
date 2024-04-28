import logging

import torch.cuda


class Base_Feature_Extractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

