import argparse
import logging

from src.utilities.Model_Enum import Model_Enum
from src.utilities.utility import get_model_by_enum


def configure_cmds(subparsers):
    extract_features_parser = subparsers.add_parser('extract_features',
                                                    help='Extract perplexity features from a model and a text sequence')
    add_args_extract_features(extract_features_parser)


def add_args_extract_features(parser):
    parser.add_argument('-m', '--model', required=True, type=str, help='Type of the model, you want to extract '
                                                                       'perplexities from. Please choose from GPT2, '
                                                                       'GPTJ, GPTNeo, Llama')
    parser.add_argument('-t', '--text', required=True, type=str, help='The text sequence you want to extract features '
                                                                      'for.')


def extract_features(args):
    model = None
    try:
        logging.info(f'Initializing model object by the model type given:{args.model}')
        model = get_model_by_enum(Model_Enum[args.model.upper()])
        logging.info(f'Initialized model: {model}')
    except Exception:
        logging.error(f'Invalid model name entered: {args.model}. Please pick a model from (GPT2, GPTJ, GPTNEO,'
                      f' Llama)')
    if model:
        logging.info('Extracting features from model')
        features = model.extract_features(args.text)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='1.0')
    subparsers = parser.add_subparsers(dest='command', required=True)
    configure_cmds(subparsers)
    args = parser.parse_args()
    if args.command == 'extract_features':
        extract_features(args)


if __name__ == '__main__':
    main()
