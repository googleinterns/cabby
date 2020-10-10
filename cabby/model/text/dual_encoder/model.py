import torch
import torch.nn as nn
from transformers import BertModel


class DualEncoder(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=200,s2cell_dim=64, output_dim= 100):
        super(DualEncoder, self).__init__()

        self.hidden_layer = nn.Linear(text_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bert = BertModel.from_pretrained(
          "bert-base-uncased", return_dict=True)
        self.text_main = torch.nn.Sequential(
                torch.nn.Linear(text_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim),
                )
        self.cellid_main = torch.nn.Sequential(
                torch.nn.Linear(s2cell_dim, output_dim),
                )

    def forward(self, text_feat, cellid):
        
        outputs = self.bert(**text_feat)
        cls_token = outputs.last_hidden_state[:,-1,:]
        text_embedding = self.text_main(cls_token)
        cellid_embedding = self.cellid_main(cellid)
        
        return text_embedding, cellid_embedding