import torch
from tqdm import tqdm, trange
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label = id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.warm_up_ratio = args.warm_up_ratio

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)

    def train(self, ckpt_name='linear_en.pt'):
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['features'], inputs['labels'])
                    logits = output['logits']
                    loss = output['loss']
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    tr_loss += loss.item()
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch + 1}: training loss {loss}')
            torch.save(self.model.cpu(), ckpt_name)
            self.model.to(self.device)

        torch.save(self.model.cpu(), ckpt_name)
        saved_model = torch.load(ckpt_name)
        self.model.load_state_dict(saved_model.state_dict())
        return
