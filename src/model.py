from typing import List, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, TransformerEncoder, TransformerEncoderLayer
from fastNLP.modules.torch import MLP, ConditionalRandomField, allowed_transitions


class SeqXGPTModel(nn.Module):

    def __init__(self, id2labels, seq_len, intermediate_size=512, num_layers=2, dropout_rate=0.1):
        super(SeqXGPTModel, self).__init__()
        conv_feature_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        conv_layers = nn.Sequential()
        input_dim = 1
        for dim, kernel_size, stride in conv_feature_layers:
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=dim, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2, bias=True),
                nn.Dropout(0.0),
                nn.ReLU()
            ))
            input_dim = dim
        self.conv = conv_layers
        self.seq_len = seq_len  # MAX Seq_len
        embedding_size = 4 * 64
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=16,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                          num_layers=num_layers)

        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000 ** ((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000 ** ((2 *
                                                   (i + 1)) / embedding_size))))

        self.norm = nn.LayerNorm(embedding_size)

        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0

    def extract_convolution_features(self, x):
        out = self.conv(x).transpose(1,2)
        return out

    def forward(self, x, labels):
        mask = labels.gt(-1)
        padding_mask = ~mask

        x = x.transpose(1, 2)
        out1 = self.extract_convolution_features(x[:, 0:1, :])
        out2 = self.extract_convolution_features(x[:, 1:2, :])
        out3 = self.extract_convolution_features(x[:, 2:3, :])
        out4 = self.extract_convolution_features(x[:, 3:4, :])
        out = torch.cat((out1, out2, out3, out4), dim=2)

        outputs = out + self.position_encoding.to(out.device)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)

        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask == 0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output
