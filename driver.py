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
    parser.add_argument('--gpu', type=str, default='0')
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
    parser.add_argument('--document_level_eval', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Use the below labels for mixed model multi-class classification
    labels = {'gpt2': 0, 'gptneo': 1, 'gptj': 2, 'llama': 3, 'gpt3re': 4, 'human': 5}

    # Use the below labels instead for mixed model binary classification
    # labels = {'gpt2': 0, 'gptneo': 0, 'gptj': 0, 'llama': 0, 'gpt3re': 0, 'human': 1}

    id2label = create_tag_mapping(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.inference:
        print("Log INFO: Performing test...")
        model_path = 'saved_model.pt'  # Provide the path of the saved checkpoint file
        data = DataHandler(args.train_path, args.test_path, args.batch_size, args.seq_len, 'human', id2label)
        model = torch.load(model_path, map_location=device)
        evaluator = Evaluator(data, model, labels, id2label, args.seq_len, device)
        evaluation_results = evaluator.evaluate_model(document_level_eval=args.document_level_eval)
        print("Evaluation Results:", evaluation_results)
    else:
        if args.split_dataset:
            dataset_split_helper(args.data_path, args.train_path, args.test_path, args.train_ratio)
        data = DataHandler(args.train_path, args.test_path, args.batch_size, args.seq_len, 'human', id2label)
        model = SeqXGPTModel(id2labels=id2label, seq_len=args.seq_len).to(device)
        trainer = Trainer(data, model, args)
        print("Log INFO: Starting training...")
        trainer.train()  # Can pass checkpoint file name as argument, default is saved_model.pt


if __name__ == "__main__":
    main()
