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

import copy
import numpy as np
import nltk
from tqdm import trange
import torch
from transformers import AdamW, pipeline, get_linear_schedule_with_warmup
from termcolor import colored
from transformers import AutoTokenizer
from statistics import mean
import re

tokenizer = AutoTokenizer.from_pretrained("t5-small", padding=True, truncation=True)


device = torch.device(
  'cuda') if torch.cuda.is_available() else torch.device('cpu')

print (f"Device used: {device}.")

def train(model, train_dataloader, val_dataloader, args):

  model = model.to(device)

  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'gamma', 'beta']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
  ]

  optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
  )

  total_steps = len(train_dataloader) * args.epochs

  ## Store the average loss after each epoch so we can plot them.
  loss_values, validation_loss_values = [], []

  best_score = {'model': None, 'f1_score': 0, 'epoch': -1}


  for epoch in trange(args.epochs, desc="Epoch"):
      # Perform one full pass over the training set.

      # Put the model into training mode.
      model.train()
      # Reset the total loss for this epoch.
      total_loss = 0

      # Training loop.
      for step, batch in enumerate(train_dataloader):
          # Set device (GPU or CPU) to each element in batch.
          input_ids = batch['source_ids'].squeeze(0)
          labels = batch['target_ids'].squeeze(0)
          src_mask = batch['src_mask'].squeeze(0)
          target_mask = batch['target_mask'].squeeze(0)

          outputs = model(input_ids=input_ids,
                          labels=labels,
                          attention_mask=src_mask,
                          decoder_attention_mask=target_mask,
                          return_dict=True)

          loss = outputs.loss
          loss.backward()
          total_loss += loss
          optimizer.step()

          total_loss += loss.item()


      # Calculate the average loss over the training data.
      avg_train_loss = total_loss / len(train_dataloader)
      print(f"Average train loss: {avg_train_loss}")

      loss_values.append(avg_train_loss)

      # After the completion of each training epoch,
      # measure the performance on the validation set.

      model.eval() # Put the model into evaluation mode.

      eval_loss = 0
      predictions , true_labels = [], []
      for batch in val_dataloader:
          input_ids = batch['source_ids'].squeeze(0)
          src_mask = batch['src_mask'].squeeze(0)
          target_text = batch['target_text']

          with torch.no_grad():
              outputs = model.generate(input_ids, attention_mask=src_mask)

          predictions.extend(outputs)
          true_labels.extend(target_text)

      eval_loss = eval_loss / len(val_dataloader)
      validation_loss_values.append(eval_loss)
      print(f"Validation loss: {eval_loss}")

      predictions = [list(filter(lambda t: t != '<pad>', l)) for l in predictions]
      predictions = [tokenizer.decode(p) for p in predictions]
      predictions = [p.replace("<pad>", "").replace("</s>", "").strip() for p in predictions]


      print (predictions[0])
      print(true_labels[0])

      f1, accuracy = metrics_score(predictions, true_labels)

      if best_score['f1_score'] < f1:
        best_score = {'model': copy.deepcopy(model),
                      'f1_score': f1,
                      'epoch': epoch,
                      }

  torch.save(best_score, args.model_path)
  return best_score['model']


def test(model, test_dataloader):

  model = model.to(device)

  model.eval()

  eval_loss = 0
  predictions , true_labels = [], []


  for batch in test_dataloader:
    input_ids = batch['source_ids'].squeeze(0)
    src_mask = batch['src_mask'].squeeze(0)
    target_text = batch['target_text']

    with torch.no_grad():
      outputs = model.generate(input_ids, attention_mask=src_mask)

    predictions.extend(outputs)
    true_labels.extend(target_text)

  eval_loss = eval_loss / len(test_dataloader)

  predictions = [list(filter(lambda t: t != '<pad>', l)) for l in predictions]
  predictions = [tokenizer.decode(p) for p in predictions]
  predictions = [p.replace("<pad>", "").replace("</s>", "").strip() for p in predictions]

  print(predictions[0])
  print(true_labels[0])

  f1, accuracy = metrics_score(predictions, true_labels)

def metrics_score(predictions, true_labels, split="Validation"):
  accuracy = sum([p==t for p,t in zip(predictions, true_labels)])/len(predictions)

  predictions = [p.split(',') for p in predictions]
  true_labels = [t.split(',') for t in true_labels]

  BLEUscore = mean([nltk.translate.bleu_score.sentence_bleu(t, p) for p,t in zip(predictions, true_labels)])


  print(f"{split} Accuracy: {accuracy}")
  print(f"{split} BLEUscore: {BLEUscore}")

  return BLEUscore, accuracy


def test_samples(instructions, tokenizer, model):

  model = model.to(device)
  model.eval()

  token = tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
  predictions = model.generate(token['input_ids'].to(device),
                           attention_mask=token['attention_mask'].to(device))

  predictions = [list(filter(lambda t: t != '<pad>', l)) for l in predictions]
  predictions = [tokenizer.decode(p) for p in predictions]
  predictions = [p.replace("<pad>", "").replace("</s>", "") for p in predictions]

  for idx, (p, i) in enumerate(zip(predictions, instructions)):
    print(idx)
    landmarks = p.split(",")
    if len(landmarks)==0:
      print (i)
      continue
    for landmark in landmarks:
      pattern = re.compile(landmark, re.IGNORECASE)
      # word = colored(' <'+landmark+'> ', 'green')

      instruction_landmark = pattern.sub(' <'+landmark+'> ', i)

    print(instruction_landmark)
    # print (p)
    # print (i)