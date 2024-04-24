import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from src.Feature_Extractors.GPT2_Feature_Extractor import GPT2_Feature_Extractor
from src.Feature_Extractors.GPTJ_Feature_Extractor import GPTJ_Feature_Extractor
from src.Feature_Extractors.GPTNeo_Feature_Extractor import GPTNeo_Feature_Extractor
from src.Feature_Extractors.Llama_Feature_Extractor import Llama_Feature_Extractor

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


def process_text_line(data, models):
    text = data['text']
    label = data['label']
    label_int = en_labels.get(label, None)

    aggregated_features = {
        'wordwise_loss_list': [],
        'label_int': label_int,
        'label': label,
        'text': text
    }

    for model_name, model in models.items():
        try:
            wordwise_loss_list = extract_features(model, text)
            aggregated_features['wordwise_loss_list'].append(wordwise_loss_list)
        except Exception as e:
            print(f"Error processing {text} with {model_name}: {str(e)}")

    return aggregated_features


def process_file(input_file, output_file, models, max_workers=10):

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []
    # Process lines in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_text_line, json.loads(line), models) for line in lines]
        for future in tqdm(futures, desc=f"Processing {input_file}"):
            results.append(future.result())

    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction using LLMs with multithreading:")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--get_features", action="store_true",
                        help="Generate features for a single file with multithreading")
    return parser.parse_args()


def main():
    args = parse_args()

    model_extractors = {
        'gpt2': GPT2_Feature_Extractor(),
        # 'gptj': GPTJ_Feature_Extractor(),
        # 'gptneo': GPTNeo_Feature_Extractor(),
        # 'llama': Llama_Feature_Extractor()
    }

    if args.get_features:
        process_file(args.input_file, args.output_file, model_extractors)


if __name__ == "__main__":
    main()
ᐧ