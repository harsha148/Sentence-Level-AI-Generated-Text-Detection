import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Filter JSONL files to include only specified labels.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument('--labels', nargs='+', required=True, help="Labels to include in the output.")
    return parser.parse_args()


def filter_jsonl(input_file, output_file, labels):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            if data['label'] in labels:
                json.dump(data, outfile)
                outfile.write('\n')


def main():
    args = parse_args()
    filter_jsonl(args.input_file, args.output_file, args.labels)


if __name__ == "__main__":
    main()
