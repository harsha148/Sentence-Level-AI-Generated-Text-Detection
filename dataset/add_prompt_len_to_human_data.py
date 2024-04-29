import argparse
import json


def add_prompt_len_to_human_data(input_file, output_file):
    with open(input_file, 'r') as input_file, open(output_file, 'w') as out_file:
        for input in input_file:
            input_data = json.loads(input)

            # Compute the prompt_len as the length of the text in human_lines.jsonl
            prompt_len = len(input_data['text'])

            # Add prompt_len to the input data
            input_data['prompt_len'] = prompt_len

            # Write the updated data to output file
            json.dump(input_data, out_file)
            out_file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add prompt length field to human lines JSONL file")
    parser.add_argument("--input_file", type=str, required=True, help="File path for human lines JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="File path for the processed JSONL data file")
    args = parser.parse_args()

    add_prompt_len_to_human_data(args.features_file, args.output_file)
