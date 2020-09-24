from transformers import BertForSequenceClassification
import torch.nn as nn


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, instruction, label):
        loss, text_fea = self.encoder(instruction, labels=label)[:2]

        return loss, text_fea