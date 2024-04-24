import json
import os
from glob import glob


def combine_model_outputs(directory, output_file):
    # Build a list of all JSONL files in the directory
    files = glob(os.path.join(directory, '*.jsonl'))
    combined_results = []

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
            if not combined_results:
                # Initialize combined_results with the first file's data
                combined_results = data
            else:
                # Append additional wordwise_loss_lists to existing entries
                for i, entry in enumerate(data):
                    combined_results[i]['wordwise_loss_list'].extend(entry['wordwise_loss_list'])

    # Write the combined results to the specified output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in combined_results:
            f.write(json.dumps(result) + '\n')


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine model output files into a single file.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing JSONL files to combine.")
    parser.add_argument("--output_file", type=str, required=True, help="File path to save the combined output.")
    args = parser.parse_args()

    combine_model_outputs(args.directory, args.output_file)
