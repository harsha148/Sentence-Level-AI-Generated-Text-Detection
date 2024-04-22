import argparse
import os
import torch
from training.trainer import Trainer
from model import SeqXGPTModel
from src.utilities.utils import dataset_split_helper, construct_bmes_labels
from evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_mode', type=str, default='classify')
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

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Splitting the data
    if args.split_dataset:
        dataset_split_helper(args.data_path, args.train_path, args.test_path, args.train_ratio)

    # Labels and model initialization
    en_labels = {'gpt2': 0, 'gptneo': 1, 'gptj': 2, 'llama': 3, 'gpt3re': 4, 'human': 5}
    id2label = construct_bmes_labels(en_labels)
    label2id = {v: k for k, v in id2label.items()}

    # change this as per the dataloader
    data = DataManager(args.train_path, args.test_path, args.batch_size, args.seq_len, 'human', id2label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqXGPTModel(id2labels=id2label, seq_len=args.seq_len).to(device)

    # Training and evaluation setup
    trainer = Trainer(data, model, en_labels, id2label, args)
    evaluator = Evaluator(data.test_dataloader, model, en_labels, id2label, device)

    if args.do_test:
        print("Log INFO: Performing test...")
        model_path = 'path_to_saved_model.pth'  # Update with actual path if needed
        model.load_state_dict(torch.load(model_path))
        evaluation_results = evaluator.test(content_level_eval=args.test_content)
        print("Evaluation Results:", evaluation_results)
    else:
        print("Log INFO: Starting training...")
        trainer.train()


if __name__ == "__main__":
    main()
