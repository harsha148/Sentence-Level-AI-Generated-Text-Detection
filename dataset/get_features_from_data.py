import json
import argparse
import os
import threading
from tqdm import tqdm

# Import the feature extractor classes
from src.Feature_Extractors.GPT2_Feature_Extractor import GPT2_Feature_Extractor
from src.Feature_Extractors.GPTJ_Feature_Extractor import GPTJ_Feature_Extractor
from src.Feature_Extractors.GPTNeo_Feature_Extractor import GPTNeo_Feature_Extractor
from src.Feature_Extractors.Llama_Feature_Extractor import Llama_Feature_Extractor

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
    """
    Extract features using the specified model.
    """
    return model.extract_features(text)


def process_file(input_file, output_file, models):
    """
    Process a single file using all LLMs.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []
    for line in tqdm(lines, desc=f"Processing {input_file}"):
        data = json.loads(line)
        text = data['text']
        label = data['label']
        label_int = en_labels.get(label, None)  # Get label integer from dictionary

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
                continue

        results.append(aggregated_features)

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction using the LLMs:")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--get_features", action="store_true", help="generate features for a single file")
    parser.add_argument("--get_features_multithreading", action="store_true",
                        help="Generate features for multiple files parallely")
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Map of models
    model_extractors = {
        'gpt2': GPT2_Feature_Extractor(),
        'gptj': GPTJ_Feature_Extractor(),
        'gptneo': GPTNeo_Feature_Extractor(),
        'llama': Llama_Feature_Extractor()
    }

    if args.get_features:
        process_file(args.input_file, args.output_file, model_extractors)

    elif args.get_features_multithreading:
        input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
        threads = []

        for file_name in input_files:
            input_file = os.path.join(args.input_dir, file_name)
            output_file = os.path.join(args.output_dir, f"{file_name}_features")
            t = threading.Thread(target=process_file, args=(input_file, output_file, model_extractors))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()


if __name__ == "__main__":
    main()

