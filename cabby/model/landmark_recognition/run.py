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
from seqeval.metrics import f1_score, accuracy_score
from tqdm import trange
import torch
from transformers import AdamW, pipeline, get_linear_schedule_with_warmup
from termcolor import colored

tag_values_idx = {0: 'O', 1: 'I', -100: 'PAD'}

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

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )

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
          for param_k, param_v in batch.items():
            if torch.is_tensor(param_v):
              batch[param_k] = param_v.to(device)

          model.zero_grad()
          # Forward pass for calculate loss.

          outputs = model(**batch)

          loss = outputs[0]

          # Perform a backward pass to calculate the gradients.
          loss.backward()

          total_loss += loss.item()

          torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)

          # Update parameters.
          optimizer.step()

          # Update the learning rate.
          scheduler.step()

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
          for param_k, param_v in batch.items():
            if torch.is_tensor(param_v):
              batch[param_k] = param_v.to(device)

          # Telling the model not to compute or store gradients,
          # saving memory and speeding up validation
          with torch.no_grad():
              # Forward pass, calculate logit predictions.

              outputs = model(**batch)

          # Move logits and labels to CPU.
          logits = outputs[1].detach().cpu().numpy()
          label_ids = batch['labels'].to('cpu').numpy()

          # Calculate the accuracy for this batch of test sentences.
          eval_loss += outputs[0].mean().item()
          predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
          true_labels.extend(label_ids)

      eval_loss = eval_loss / len(val_dataloader)
      validation_loss_values.append(eval_loss)
      print(f"Validation loss: {eval_loss}")

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
    for param_k, param_v in batch.items():
      if torch.is_tensor(param_v):
        batch[param_k] = param_v.to(device)

    # Telling the model not to compute or store gradients,
    # saving memory and speeding up validation.
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        outputs = model(**batch)
    # Move logits and labels to CPU.
    logits = outputs[1].detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()

    eval_loss += outputs[0].mean().item()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.extend(label_ids)

  eval_loss = eval_loss / len(test_dataloader)
  print(f"Test loss: {eval_loss}")

  _, _ = metrics_score(predictions, true_labels, "Test")


def metrics_score(predictions, true_labels, split="Validation"):
  predictions = [tag_values_idx[p_i] for p, l in zip(predictions, true_labels)
               for p_i, l_i in zip(p, l) if tag_values_idx[l_i] != "PAD"]

  true_labels = [tag_values_idx[l_i] for l in true_labels
                for l_i in l if tag_values_idx[l_i] != "PAD"]

  f1 = f1_score([predictions], [true_labels])

  accuracy = accuracy_score(predictions, true_labels)
  print(f"{split} Accuracy: {accuracy}")
  print(f"{split} F1-Score: {f1}")

  return f1, accuracy


def test_samples(instructions, tokenizer, model):

  model = model.to(device)
  nlp = pipeline("ner",
                 model=model,
                 tokenizer=tokenizer,
                 device=torch.cuda.current_device())

  ner_results = nlp(instructions)
  for result in ner_results:
    list_words = []
    for word_dic in result:
      assert word_dic['entity'] in ['LABEL_0', 'LABEL_1']
      if word_dic['entity'] == "LABEL_1":
        word = colored(word_dic['word'], 'green')
      else:
        word = colored(word_dic['word'], 'white')
      list_words.append(word)

    print(' '.join(list_words))

