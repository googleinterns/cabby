from transformers import BertForSequenceClassification
import torch.nn as nn


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        self.encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=True, num_labels = 2)

    def forward(self, instruction, label):
        outputs = self.encoder(instruction, labels=label)

        return outputs.loss, outputs.logits