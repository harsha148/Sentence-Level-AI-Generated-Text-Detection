import logging

import torch.cuda


class Base_Feature_Extractor:
    def __init__(self):
        if torch.cuda.is_available():
            logging.error('GPU available so setting device to cuda')
        else:
            logging.error('GPU not available, so setting device to CPU')
        self.device = 'cpu'
