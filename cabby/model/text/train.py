
# Copyright 2020 The Flax Authors.
#
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

import torch 
import torch.optim as optim
import torch.nn as nn
from transformers import AdamW


import dataset
import model 


def train(model,
          optimizer,
          criterion,
          train_loader,
          valid_loader,
          num_epochs = 100,
          eval_every = 10,
          ):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (instruction, labels) in train_loader:
            labels = labels.type(torch.LongTensor)           
            instruction = instruction.type(torch.LongTensor)  
            output = model(instruction, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for (instruction, labels) in valid_loader:
                        labels = labels.type(torch.LongTensor)           
                        instruction = instruction.type(torch.LongTensor)  
                        output = model(instruction, labels)
                        loss, _ = output
                        
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                

    print('Finished Training!')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter, test_iter =  dataset.create_dataset("~/data/morp/morp-balanced", (16, 256, 256), device)

bert_model = model.BERT().to(device)
optimizer = AdamW(bert_model.parameters(), lr=2e-5)
criterion = nn.BCELoss()

train(model=bert_model, optimizer=optimizer, criterion=criterion, train_loader=train_iter,valid_loader = val_iter)
