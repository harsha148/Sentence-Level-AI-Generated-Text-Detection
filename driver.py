import argparse
import os
import torch
from src.training.trainer import Trainer
from src.model import SeqXGPTModel
from src.utilities.datahandler import DataHandler
from src.utilities.utils import dataset_split_helper, create_tag_mapping
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--train_mode', type=str, default='classify')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=1024)

    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_dataset', action='store_true')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')

    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    en_labels = {'gpt2': 0, 'gptneo': 0, 'gptj': 0, 'llama': 0, 'gpt3re': 0, 'human': 1}
    id2label = create_tag_mapping(en_labels)

    if args.inference:
        print("Log INFO: Performing test...")
        model_path = 'saved_model_100_epochs.pt'  # Ensure this path is correct
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)  # Load the entire model
        data = DataHandler(args.train_path, args.test_path, args.batch_size, args.seq_len, 'human', id2label)
        evaluator = Evaluator(data, model, en_labels, id2label, args.seq_len, device)
        evaluation_results = evaluator.evaluate_model(content_level_eval=args.test_content)
        print("Evaluation Results:", evaluation_results)
    else:
        data = DataHandler(args.train_path, args.test_path, args.batch_size, args.seq_len, 'human', id2label)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SeqXGPTModel(id2labels=id2label, seq_len=args.seq_len).to(device)
        trainer = Trainer(data, model, en_labels, id2label, args)
        print("Log INFO: Starting training...")
        trainer.train()

if __name__ == "__main__":
    main()

