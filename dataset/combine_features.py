import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import os


def read_json_objects(file_path):
    """ Reads a JSONL file and returns a list of dictionaries. """
    with open(file_path, 'r', encoding='utf-8') as file:
        json_objects = [json.loads(line.strip()) for line in file if line.strip()]
    return json_objects


def merge_features(input_dir, output_file):
    """ Merges features from feature files of different models in the specified directory for a particular data file"""
    file_names = sorted(os.listdir(input_dir))
    file_paths = [os.path.join(input_dir, file_name) for file_name in file_names if file_name.endswith('.jsonl')]

    if len(file_paths) < 3:
        print("Error: Not enough JSONL files in the directory.")
        return

    # Reading JSONL files in parallel
    with ThreadPoolExecutor() as executor:
        datasets = list(executor.map(read_json_objects, file_paths))

    combined_data = []
    for entry_group in zip(*datasets):
        base_entry = entry_group[0]
        all_wordwise_loss_lists = base_entry['wordwise_loss_list']
        for other_entry in entry_group[1:]:
            all_wordwise_loss_lists += other_entry['wordwise_loss_list']

        base_entry['wordwise_loss_list'] = all_wordwise_loss_lists
        combined_data.append(base_entry)

    # Writing the combined output to a new JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in combined_data:
            json.dump(entry, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine model output files into a single file.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing JSONL files to combine.")
    parser.add_argument("--output_file", type=str, required=True, help="File path to save the combined output.")
    args = parser.parse_args()
    merge_features(args.directory, args.output_file)
