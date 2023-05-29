# coding=utf-8
# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import logging

import numpy as np
import sys
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from vector_quantize_pytorch import VectorQuantize
from transformers import ViTForMaskedImageModeling

from typing import Dict, Sequence
import re


from cabby.model import util as mutil
from cabby.geo import util as gutil

T5_TYPE = "t5-small"
T5_DIM = 512 if T5_TYPE == "t5-small" else 768
BERT_TYPE = "distilbert-base-uncased"

criterion = nn.CosineEmbeddingLoss()


class GeneralModel(nn.Module):
  def __init__(self, device):
    super(GeneralModel, self).__init__()
    self.is_generation = False
    self.device = device

  def get_embed(self, text_feat, cellid):
    return text_feat, cellid

  def forward(self, text: Dict, is_print, *args
              ):
    sys.exit("Implement compute_loss function in model")

  def predict(self, text, is_print, *args):
    sys.exit("Implement prediction function in model")


class DualEncoder(GeneralModel):
  def __init__(
    self,
    device,
    text_dim=768,
    hidden_dim=200,
    s2cell_dim=64,
    output_dim=100,
    is_distance_distribution=False
  ):
    GeneralModel.__init__(self, device)

    self.hidden_layer = nn.Linear(text_dim, hidden_dim)
    self.softmax = nn.Softmax(dim=-1)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.is_distance_distribution = is_distance_distribution

    self.model = DistilBertModel.from_pretrained(
      BERT_TYPE, return_dict=True)

    self.text_main = nn.Sequential(
      nn.Linear(text_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, output_dim),
    )
    self.cellid_main = nn.Sequential(
      nn.Linear(s2cell_dim, output_dim),
    )
    self.cos = nn.CosineSimilarity(dim=2)

  def get_embed(self, text_feat, cellid):
    text_embedding = self.text_embed(text_feat)
    cellid_embedding = self.cellid_main(cellid.float())

    return text_embedding.shape[0], text_embedding, cellid_embedding

  def predict(self, text, is_print, all_cells, *args):
    batch = args[1]
    batch_dim, text_embedding_exp, cellid_embedding = self.get_embed(text, all_cells)
    cell_dim = cellid_embedding.shape[0]
    output_dim = cellid_embedding.shape[1]

    text_embedding_exp = text_embedding_exp.unsqueeze(1).expand(batch_dim, cell_dim, output_dim)
    cellid_embedding_exp = cellid_embedding.expand(batch_dim, cell_dim, output_dim)

    label_to_cellid = args[0]
    assert cellid_embedding_exp.shape == text_embedding_exp.shape
    output = self.cos(cellid_embedding_exp, text_embedding_exp)

    if self.is_distance_distribution:
      prob = batch['prob'].float().to(self.device)
      output = output * prob

    output = output.detach().cpu().numpy()

    predictions = np.argmax(output, axis=1)

    points = mutil.predictions_to_points(predictions, label_to_cellid)
    return points

  def forward(self, text, cellid, is_print, *args
              ):
    batch = args[0]
    neighbor_cells = batch['neighbor_cells']
    far_cells = batch['far_cells']

    # Correct cellid.
    target = torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding, cellid_embedding = self.get_embed(text, cellid)
    loss_cellid = criterion(text_embedding, cellid_embedding, target)

    # Neighbor cellid.
    target_neighbor = -1 * torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding_neighbor, cellid_embedding = self.get_embed(text, neighbor_cells)
    loss_neighbor = criterion(text_embedding_neighbor,
                              cellid_embedding, target_neighbor)

    # Far cellid.
    target_far = -1 * torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding_far, cellid_embedding = self.get_embed(text, far_cells)
    loss_far = criterion(text_embedding_far, cellid_embedding, target_far)

    loss = loss_cellid + loss_neighbor + loss_far

    return loss.mean()

  def text_embed(self, text):
    outputs = self.model(**text)
    cls_token = outputs.last_hidden_state[:, -1, :]
    return self.text_main(cls_token)

class S2GenerationModel(GeneralModel):
  def __init__(
    self,
    label_to_cellid,
    device,
    graph_codebook,
    model_type='S2-Generation-T5',
    vq_dim=224,
  ):
    GeneralModel.__init__(self, device)
    self.model = T5ForConditionalGeneration.from_pretrained(T5_TYPE)
    self.tokenizer = T5Tokenizer.from_pretrained(T5_TYPE)
    self.is_generation = True
    self.label_to_cellid = label_to_cellid
    self.model_type = model_type
    self.max_size = len(str(len(label_to_cellid)))
    self.graph_codebook = graph_codebook

    self.decoder = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")
    self.num_patches = (self.decoder.config.image_size // self.decoder.config.patch_size) ** 2
    self.discriminator = Discriminator()
    self.vq_dim = vq_dim
    logging.info(f"Vector quantization size {vq_dim}")

    self.vq = VectorQuantize(
      dim=vq_dim,
      codebook_size=graph_codebook,
      codebook_dim=16,
      decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
      commitment_weight = 1.   # the weight on the commitment loss
    )

    if model_type not in ['S2-Generation-T5']:
      self.max_size = self.max_size * 100

    self.quant = torch.quantization.QuantStub()

    if self.vq_dim:
      self.original_size_tokenizer = len(self.tokenizer)
      logging.info(f"Size of tokenizer before resized: {self.original_size_tokenizer}")

      add_tokens = [f"GRAPH_{t}" for t in range(self.graph_codebook)]
      self.tokenizer.add_tokens(add_tokens)
      self.model.resize_token_embeddings(len(self.tokenizer))
      logging.info(f"Resized tokenizer to: {len(self.tokenizer)}")

  def forward(self, text, cellid, is_print, *args):

    batch = args[0]

    input_ids, attention_mask, labels = self.get_input_output(batch, text)

    if self.vq_dim:
      loss, input_ids = self.get_loss_for_graph_embed(batch, input_ids, labels)

    else:
      loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True).loss

    if is_print:
      input_ids_decoded = self.tokenizer.batch_decode(
        input_ids, skip_special_tokens=True)
      logging.info(f"Actual input decoded: {input_ids_decoded[0]}")

      output_ids_decoded = self.tokenizer.batch_decode(
        labels, skip_special_tokens=True)
      logging.info(f"Actual output decoded: {output_ids_decoded[0]}")

    return loss

  def get_embed(self, text, cellid):
    text_dim = text['input_ids'].shape[0]
    return text_dim, text, cellid

  def predict(self, text, is_print, cells_tensor, label_to_cellid, batch):

    # batch = args[-1]

    input_ids, attention_mask, _ = self.get_input_output(batch, text)

    output_sequences = self.model.generate(
      input_ids=input_ids,
      attention_mask=attention_mask,
      num_beams=2,
      max_length=self.max_size + 2,
      min_length=1,
    )

    if self.vq_dim:
      graph_embed_start = batch['graph_embed_start']
      input_ids, _, _ = self.get_input_indices_for_embedding(graph_embed_start, input_ids)
      output_sequences = self.model.generate(
        input_ids=input_ids,
        num_beams=2,
        max_length=self.max_size + 2,
        min_length=1,
      )

      add_tokens = [f"GRAPH_{t}" for t in range(self.graph_codebook)]
      self.tokenizer.add_tokens(add_tokens)
      self.model.resize_token_embeddings(len(self.tokenizer))

    prediction = self.tokenizer.batch_decode(
      output_sequences, skip_special_tokens=True)

    if is_print:
      logging.info(f"Actual prediction decoded: {prediction[0]}")

    prediction_coords = []

    pattern = r'loc_\d+ loc_\d+'

    start_list = batch['start_point'].squeeze().detach().cpu().tolist()

    for pred_raw, start_point in zip(prediction, start_list):

      coord = pred_raw.split(";")[0].strip()
      
      try:
        coord_regex = re.findall(pattern, coord)[0]
      except:
        coord_regex = coord

      if coord_regex in label_to_cellid:
        cell_id = label_to_cellid[coord_regex]
        prediction_coords.append(gutil.get_center_from_s2cellids([cell_id])[0])
      else:
        # logging.info(pred_raw)
        # logging.info(f"!!!! coord not in dict: {coord_regex}")
        # logging.info(list(label_to_cellid.keys())[-1])
        # logging.info(f"start_point: {start_point} {type(start_point)}")
        # logging.info(f"apposed to: {prediction_coords[0]} {type(prediction_coords[0])}")
        prediction_coords.append(start_point)

    return np.array(prediction_coords)

  def get_vg(self, graph_embed):
    quantized, indices, vq_loss = self.vq(graph_embed)  # (1, 1024, 256), (1, 1024), (1)

    assert torch.max(indices)<self.graph_codebook, indices
    indices += self.original_size_tokenizer
    assert torch.max(indices)<self.graph_codebook+self.original_size_tokenizer , indices

    return quantized, indices, vq_loss

  def get_input_indices_for_embedding(self, graph_embed_start, text_input):

    batch_size = graph_embed_start.shape[0]

    graph_embed = graph_embed_start.unsqueeze(1).expand(batch_size, 1, -1)

    quantized, indices, vq_loss = self.get_vg(graph_embed)

    final_input = torch.cat((text_input, indices), axis=-1)

    return final_input, quantized, vq_loss

  def get_loss_for_graph_embed(self, batch, text_input, labels):

    graph_embed_start = batch['graph_embed_start']

    graph_size = graph_embed_start.shape[-1]
    batch_size = graph_embed_start.shape[0]

    final_input, quantized, vq_loss = self.get_input_indices_for_embedding(
      graph_embed_start, text_input)

    graph_embed_fake = torch.randn(batch_size, graph_size).to(self.device)

    _, quantized_fake, vq_loss_fake = self.get_input_indices_for_embedding(
      graph_embed_fake, text_input)

    bool_masked_pos = torch.randint(
      low=0, high=1, size=(1, self.num_patches)).bool().to(self.device)
    decoded_output = self.decoder(
      quantized.unsqueeze(1).expand(-1, 3, quantized.shape[-1], -1), bool_masked_pos=bool_masked_pos)

    decoded_output_fake = self.decoder(quantized_fake.unsqueeze(1).expand(-1, 3, quantized_fake.shape[-1], -1),
                                       bool_masked_pos=bool_masked_pos)

    discriminator_true = self.discriminator(decoded_output.logits).squeeze(0)
    discriminator_fake = self.discriminator(decoded_output_fake.logits).squeeze(0)

    loss_D = hinge_loss_dis(discriminator_fake, discriminator_true)

    loss_t5 = self.model(
      input_ids=final_input, labels=labels, return_dict=True).loss

    all_loss = vq_loss + decoded_output.loss + loss_t5 + loss_D

    return all_loss, final_input

  def get_input_output(self, batch, text_input):

    text_output = batch['text_output']
    input_ids = text_input['input_ids']
    attention_mask = text_input['attention_mask']


    return input_ids, attention_mask, text_output


class ClassificationModel(GeneralModel):
  def __init__(self, n_lables, device, hidden_dim=200):
    GeneralModel.__init__(self, device)
    self.is_generation = True

    self.model = DistilBertForSequenceClassification.from_pretrained(
      'distilbert-base-uncased', num_labels=n_lables, return_dict=True)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, text, cellid, is_print, *args):
    labels = args[0]['label']

    outputs = self.model(
      input_ids=text['input_ids'],
      attention_mask=text['attention_mask'],
      labels=labels)

    return outputs.loss

  def predict(self, text, is_print, all_cells, *args):
    label_to_cellid = args[0]

    outputs = self.model(**text)
    logits = outputs.logits.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)

    points = mutil.predictions_to_points(predictions, label_to_cellid)
    return points



def hinge_loss_dis(fake, real):
  # assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape, f'{fake.shape} {real.shape}'
  loss = torch.nn.functional.relu(1.0 - real).mean() + \
         torch.nn.functional.relu(1.0 + fake).mean()
  return loss.mean()

class Discriminator(torch.nn.Module):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn=True):
        super(Discriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        self.conv1 = sn_fn(torch.nn.Conv2d(3, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))
        self.fc = sn_fn(torch.nn.Linear(4 * 4 * 512, 1))
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        m = self.act(self.conv1(x))
        m = self.act(self.conv2(m))
        m = self.act(self.conv3(m))
        m = self.act(self.conv4(m))
        m = self.act(self.conv5(m))
        m = self.act(self.conv6(m))
        m = self.act(self.conv7(m))

        return self.fc(m.reshape(m.shape[0],-1, 4 * 4 * 512))
