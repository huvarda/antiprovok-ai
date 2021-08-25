import transformers
from transformers import BertModel, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import emoji

import re
torch.cuda.empty_cache()


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0")


max_len=128
batch_size=6
EPOCHS=40

def preprocessTwitterData(dataset):
    #dataset = dataset[dataset['label'] != "unusable"] #remove unusables

    for i in range(len(dataset.example)):
        removedHashtags = re.sub(r'(\s)#\w+', '', dataset.example[i])
        removeAt = " ".join([word for word in removedHashtags.split(" ") if "@" not in word])
        removeEmojis = emoji.replace_emoji(removeAt, "")
        final = re.sub(r'http\S+', '', removeEmojis)
        dataset.at[i, "example"] = final

    dataset = dataset[dataset['label'] != "unusable"] #remove unusables

    return dataset

def outputNumbers(sentiment):
    if sentiment == "neither":
        return 0
    elif sentiment == "political":
        return 0
    elif sentiment == "provocative":
        return 1
        
ds = pd.read_csv("datasetFinal.csv")
ds = preprocessTwitterData(ds)

ds["sentiment"] = ds.label.apply(outputNumbers)

class_names = ["neither", "provocative"] #"political", 


class TweetSentimentDataset(Dataset):
    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets=tweets
        self.targets=targets
        self.tokenizer=tokenizer
        self.max_len=max_len
        
    def __len__(self):
        return len(self.tweets)
        
    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        target=self.targets[item]
        encoding = tokenizer.encode_plus(tweet, max_length=self.max_len, add_special_tokens=True, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, return_tensors='pt')
        return {"tweets":tweet, "input_ids":encoding["input_ids"].flatten(), "attention_mask": encoding["attention_mask"].flatten(), "targets":torch.tensor(target, dtype=torch.long)}



tokenizer = transformers.BertTokenizerFast.from_pretrained('dbmdz/bert-base-turkish-uncased')

train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=RANDOM_SEED)
val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=RANDOM_SEED)

def make_data_loader(ds, tokenizer, max_len, batch_size):
    dataset = TweetSentimentDataset(tweets=ds.example.to_numpy(), targets=ds.sentiment.to_numpy(), tokenizer=tokenizer, max_len=max_len)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size)#, num_workers=2)
    return dataloader
    
train_data_loader = make_data_loader(train_ds, tokenizer, max_len, batch_size)
test_data_loader = make_data_loader(test_ds, tokenizer, max_len, batch_size)
val_data_loader = make_data_loader(val_ds, tokenizer, max_len, batch_size)



class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('dbmdz/bert-base-turkish-uncased', return_dict=False)
        self.drop = nn.Dropout(p=0.35)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)



model = SentimentClassifier(len(class_names))
model = model.to(device)



optimizer=AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss().to(device)




def loop_data(model, data_loader, loss_fn, losses, correct_predictions, optimization=True):
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
                
        if optimization == True:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return losses, correct_predictions




def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions=0
    
    losses, correct_predictions = loop_data(model, data_loader, loss_fn, losses, correct_predictions)
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions=0
    
    losses, correct_predictions = loop_data(model, data_loader, loss_fn, losses, correct_predictions, optimization=False)
            
    return correct_predictions.double() / n_examples, np.mean(losses)




if __name__ == "__main__":
    history = defaultdict(list)
    best_accuracy=0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print("-----------")

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_ds))
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = train_epoch(model, val_data_loader, loss_fn, optimizer, device, scheduler, len(val_ds))
        print(f'Validation loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'output/modelWithoutPolSplitClassCountRight.bin')
            best_accuracy=val_acc



#       TURKISH BERT MODEL CITATION 
#@software{stefan_schweter_2020_3770924,
# author       = {Stefan Schweter},
# title        = {BERTurk - BERT models for Turkish},
# month        = apr,
# year         = 2020,
# publisher    = {Zenodo},
# version      = {1.0.0},
# doi          = {10.5281/zenodo.3770924},
# url          = {https://doi.org/10.5281/zenodo.3770924}
# }











































