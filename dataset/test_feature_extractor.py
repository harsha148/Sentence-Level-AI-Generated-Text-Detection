import json
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import the feature extractor classes
from src.Feature_Extractors.GPT2_Feature_Extractor import GPT2_Feature_Extractor
from src.Feature_Extractors.GPTJ_Feature_Extractor import GPTJ_Feature_Extractor
from src.Feature_Extractors.GPTNeo_Feature_Extractor import GPTNeo_Feature_Extractor
from src.Feature_Extractors.Llama_Feature_Extractor import Llama_Feature_Extractor
from src.utilities.common import get_model_by_enum

# Label mapping dictionary
en_labels = {
    'gpt2': 0,
    'gptneo': 1,
    'gptj': 1,
    'llama': 2,
    'gpt3re': 3,
    'gpt3sum': 3,
    'human': 4
}


def extract_features(model, text):
    return model.extract_features(text)


def process_text_line(data, model):
    text = data['text']
    label = data['label']
    label_int = en_labels.get(label, None)

    aggregated_features = {
        'wordwise_loss_list': [],
        'label_int': label_int,
        'label': label,
        'text': text
    }

    try:
        wordwise_loss_list = extract_features(model, text)
        aggregated_features['wordwise_loss_list'].append(wordwise_loss_list)
    except Exception as e:
        print(f"Error processing {text} with {model}: {str(e)}")

    return aggregated_features


def process_file(input_file, output_file, model, max_workers=3):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_text_line, json.loads(line), model) for line in lines]
        for future in tqdm(futures, desc=f"Processing {input_file}"):
            results.append(future.result())

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction using the LLMs with multithreading:")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--input_dir", type=str, help="Directory with input JSONL files.")
    parser.add_argument("--output_dir", type=str, help="Directory for output JSONL files.")
    parser.add_argument("--get_features", action="store_true", help="generate features for a single file")
    parser.add_argument("--get_features_multithreading", action="store_true",
                        help="Generate features for multiple files parallely")
    parser.add_argument('-m', '--model', required=True, type=str, help='Model type for feature extraction')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.get_features:
        model = get_model_by_enum(args.model.upper())
        process_file(args.input_file, args.output_file, model)

    elif args.get_features_multithreading:
        input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        for input_file in input_files:
            output_file = os.path.join(args.output_dir,
                                       os.path.basename(input_file).replace('.jsonl', '_features.jsonl'))
            model = get_model_by_enum(args.model.upper())
            process_file(input_file, output_file, model)


if __name__ == "__main__":
    main()
