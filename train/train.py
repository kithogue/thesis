import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# todo: refactor for .py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

PRETRAINED_MODEL = 't5-base'
DIR = "question_generator/"
BATCH_SIZE = 4
SEQ_LENGTH = 512

tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL)
tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<answer>', '<context>']}
)


class QGDataset(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv, engine='python')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx, 1:]

        encoded_text = tokenizer(
            row['text'],
            pad_to_max_length=True,
            max_length=SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        encoded_text['input_ids'] = torch.squeeze(encoded_text['input_ids'])
        encoded_text['attention_mask'] = torch.squeeze(encoded_text['attention_mask'])

        encoded_question = tokenizer(
            row['question'],
            pad_to_max_length=True,
            max_length=SEQ_LENGTH,
            truncation=True,
            return_tensors='pt'
        )
        encoded_question['input_ids'] = torch.squeeze(encoded_question['input_ids'])

        return encoded_text.to(device), encoded_question.to(device)


train_set = QGDataset(os.path.join(DIR, 'question_generator/datasets/qg_train.csv'))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_set = QGDataset(os.path.join(DIR, 'question_generator/datasets/qg_valid.csv'))
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

LR = 0.001
EPOCHS = 20
LOG_INTERVAL = 5000

config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
model = T5ForConditionalGeneration(config).from_pretrained(PRETRAINED_MODEL)
model.resize_token_embeddings(len(tokenizer))  # to account for new special tokens
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

SAVED_MODEL_PATH = "question_generator/qg_pretrained_t5_model_trained.pth"


def train(epoch, best_val_loss):
    model.train()
    total_loss = 0.
    for batch_index, batch in enumerate(train_loader):
        data, target = batch
        optimizer.zero_grad()
        masked_labels = mask_label_padding(target['input_ids'])
        output = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            lm_labels=masked_labels
        )
        loss = output[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch_index % LOG_INTERVAL == 0 and batch_index > 0:
            cur_loss = total_loss / LOG_INTERVAL
            print('| epoch {:3d} | '
                  '{:5d}/{:5d} batches | '
                  'loss {:5.2f}'.format(
                epoch,
                batch_index, len(train_loader),
                cur_loss))
            save(
                TEMP_SAVE_PATH,
                epoch,
                model.state_dict(),
                optimizer.state_dict(),
                best_val_loss
            )
            total_loss = 0


def evaluate(eval_model, data_loader):
    eval_model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            data, target = batch
            masked_labels = mask_label_padding(target['input_ids'])
            output = eval_model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                lm_labels=masked_labels
            )
            total_loss += output[0].item()
    return total_loss / len(data_loader)


def mask_label_padding(labels):
    MASK_ID = -100
    labels[labels == tokenizer.pad_token_id] = MASK_ID
    return labels


def save(path, epoch, model_state_dict, optimizer_state_dict, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'best_loss': loss,
    }, path)


def load(path):
    return torch.load(path)


def print_line():
    LINE_WIDTH = 60
    print('-' * LINE_WIDTH)


best_val_loss = float("inf")
best_model = None

val_loss = evaluate(model, valid_loader)
print_line()
print('| Before training | valid loss {:5.2f}'.format(
    val_loss)
)
print_line()

for epoch in range(1, EPOCHS + 1):

    train()
    val_loss = evaluate(model, valid_loader)
    print_line()
    print('| end of epoch {:3d} | valid loss {:5.2f}'.format(
        epoch,
        val_loss)
    )
    print_line()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        save(
             SAVED_MODEL_PATH,
             epoch,
             model.state_dict(),
             optimizer.state_dict(),
             best_val_loss
        )
        print("| Model saved.")
        print_line()