#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
train_path = 'data/train_data_public.csv'
test_path = 'data/test_public.csv'
model_name = 'bert-base-chinese'

max_len = 100
batch_size = 16
tokenizer = BertTokenizer.from_pretrained(
    model_name, 
    do_lower_case=True) 

class MyDataSet(Dataset):
    def __init__(self, tokenizer, file_path, max_len, mode):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.label_dict = self.get_labels()
        self.label_number = len(self.label_dict)
        self.data_set = self.convert_data_to_ids(file_path)
    
    def _read(self, filename):
        df = pd.read_csv(filename)
        samples = [] #dictionary-like list for recording original data sample
        for idx,row in df.iterrows():
            text = row['text']
            if(type(text)==float):
                print(text)
                continue
            tokens = list(row['text'])
            if(self.mode == 'test'):
                tags = []
                class_ = None
            else:
                tags = row['BIO_anno'].split()
                class_ = row['class']
            samples.append({"tokens": tokens, "labels":tags, "class":class_})
        return samples

    # make a label dictionary: key=string, value=index(id)
    def get_labels(self):
        label_dic={}
        label_list=["B-BANK","I-BANK","B-PRODUCT","O","I-PRODUCT","B-COMMENTS_N","I-COMMENTS_N","B-COMMENTS_ADJ","I-COMMENTS_ADJ"]
        for idx,label in enumerate(label_list):
            label_dic[label]=idx

        return label_dic

    def convert_data_to_ids(self, file_path):
        self.data_set=[]
        samples=self._read(file_path)
        for sample in tqdm(samples, desc="Convert data to ids", disable=False):
            if self.mode == 'train':
                sample = self.convert_sample_to_id_train(sample) 
            else :
                sample = self.convert_sample_to_id_test(sample)
                
            self.data_set.append(sample)
        return self.data_set

    def convert_sample_to_id_train(self, sample):
        # adding more details to a single sample
        # 1. tokens -> input_ids && token_type_ids
        # 2. labels -> labels_ids
        # 3. class is useless
        # AuxInfo: attention_mask/position_ids/len
        
        tokens = sample["tokens"]
        labels = sample["labels"]
        class_ = sample["class"]
        assert len(tokens) == len(labels), 'unmatched things happen'
        new_tokens = []
        for token in tokens:
            if not len(self.tokenizer.tokenize(token)):
                new_tokens.append('[UNK]')
            else:
                new_tokens.append(token)
        if len(new_tokens) > self.max_len - 2:
            new_tokens = new_tokens[:self.max_len - 2]
            labels = labels[:self.max_len - 2]

        new_tokens = ["[CLS]"] + new_tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
        attention_mask = [1] * len(input_ids)
        # the additional two 'O' correspond to '[CLS]' and '[SEP]'
        labels_ids = [self.label_dict["O"]] + [self.label_dict[l] for l in labels] + [self.label_dict["O"]]
        # fill some shorter sample to the max_len with nonsense [PAD]
        padding_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])
        len_ = len(input_ids)

        input_ids = input_ids + padding_id * (self.max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))
        labels_ids = labels_ids + [self.label_dict["O"]] * (self.max_len -len(labels_ids))
        token_type_ids = [0] * len(input_ids) #?? for what?
        position_ids = list(np.arange(len(input_ids)))
        sample["input_ids"] = input_ids
        sample["labels_ids"] = labels_ids
        sample["attention_mask"] = attention_mask
        sample["token_type_ids"] = token_type_ids
        sample["position_ids"] = position_ids
        sample["class"] = class_
        sample["len"] = len_
        assert len(input_ids) == len(labels_ids), "input unmatch with label-length"        
        assert len(input_ids) == self.max_len
        return sample

    def convert_sample_to_id_test(self, sample):
        tokens = sample["tokens"]
        
        new_tokens = []
        for token in tokens:
            if not len(self.tokenizer.tokenize(token)):
                new_tokens.append('[UNK]')
            else:
                new_tokens.append(token)
        if len(new_tokens) > self.max_len - 2:
            new_tokens = new_tokens[:self.max_len - 2]

        new_tokens = ["[CLS]"] + new_tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
        attention_mask = [1] * len(input_ids)
        padding_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])
        len_ = len(input_ids)

        input_ids = input_ids + padding_id * (self.max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))
        token_type_ids = [0] * len(input_ids)
        position_ids = list(np.arange(len(input_ids)))
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["token_type_ids"] = token_type_ids
        sample["position_ids"] = position_ids
        sample["len"] = len_
        assert len(input_ids) == self.max_len
        return sample

    
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance

# collate_func_x: organize the Dataset into a dictionary combination
def collate_func_train(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return{}
    input_ids_list, attention_mask_list, token_type_ids_list, labels_ids_list = [], [], [], []
    position_ids_list, tokens_list = [], []
    len_list = []
    class_list = []
    for instance in batch_data:
        input_ids_list.append(instance["input_ids"])
        attention_mask_list.append(instance["attention_mask"])
        token_type_ids_list.append(instance["token_type_ids"])
        labels_ids_list.append(instance["labels_ids"])
        position_ids_list.append(instance["position_ids"])
        tokens_list.append(instance["tokens"])
        len_list.append(instance["len"])
        class_list.append(instance["class"])
    
    return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
            "position_ids": torch.tensor(position_ids_list, dtype=torch.long),
            "labels_ids": torch.tensor(labels_ids_list, dtype=torch.long),
            "classes": torch.tensor(class_list, dtype=torch.long),
            "tokens": tokens_list,
            "lens": len_list}

def collate_func_test(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return{}
    input_ids_list, attention_mask_list, token_type_ids_list = [], [], []
    position_ids_list, tokens_list = [], []
    len_list = []
    for instance in batch_data:
        input_ids_list.append(instance["input_ids"])
        attention_mask_list.append(instance["attention_mask"])
        token_type_ids_list.append(instance["token_type_ids"])
        position_ids_list.append(instance["position_ids"])
        tokens_list.append(instance["tokens"])
        len_list.append(instance["len"])
    
    return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
            "position_ids": torch.tensor(position_ids_list, dtype=torch.long),
            "tokens": tokens_list,
            "len": torch.tensor(len_list, dtype=torch.long)}

train_data_original = MyDataSet(tokenizer, train_path, max_len, mode='train')
test_data_original = MyDataSet(tokenizer, test_path, max_len, mode='test')

from transformers import BertModel
import torch.nn as nn

modal_name = "bert-base-chinese"
hidden_size = 768 # the output size of BERT
num_label = len(train_data_original.label_dict)
num_classes=3

class BERTLinearModel(nn.Module):
    def __init__(self):
        super(BERTLinearModel, self).__init__()
    
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier4NER = nn.Linear(hidden_size, num_label)
        self.classifier4SA = nn.Linear(hidden_size, num_classes)
    
    def forward(self, device, batch):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        position_ids = batch["position_ids"].to(device)
        bert_output = self.bert(input_ids, attention_mask=attention_mask, 
                      token_type_ids=token_type_ids, position_ids=position_ids)
        
        sequence_output, pooled_output = bert_output[0], bert_output[1]
    
        ner_logits = self.classifier4NER(sequence_output)
        sa_logits = self.classifier4SA(pooled_output)
        out = ner_logits, sa_logits
        return out

loss_fct = nn.CrossEntropyLoss()


def get_k_fold_data(k, i, X):
    assert k > 1
    fold_size = len(X) // k  
    
    X_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step):return a 'slice' object
        X_part = X[idx]
        if j == i: # take j_th fold as valid
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = ConcatDataset([X_train, X_part])
    return X_train,  X_valid
 

def k_fold(model, k, train_data_original, num_epochs=3,learning_rate=0.001, batch_size=5):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum ,valid_acc_sum = 0,0

    for i in range(k):
        # get the train_data and valid_data from original train_data
        train_data, val_data = get_k_fold_data(k, i, train_data_original) 
        
        train_ls, valid_ls = train(model, train_data, val_data, num_epochs, learning_rate, batch_size)
        
        # regard the last epoch's result as this train's final result
        print(
            f'''Fold: {i + 1}
          | Train Loss: {train_ls[-1][0]: .3f}
          | Train Accuracy: {train_ls[-1][1]: .3f}
          | Val Loss: {valid_ls[-1][0]: .3f}
          | Val Accuracy: {valid_ls[-1][1]: .3f}''')
        
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
        
    print(
            f'''Finally Result: 
          | train_loss_sum: {train_loss_sum/k: .3f}
          | train_acc_sum: {train_acc_sum/k: .3f}
          | valid_loss_sum: {valid_loss_sum/k: .3f}
          | valid_acc_sum: {valid_acc_sum/k: .3f}''')


from transformers import AdamW
from tqdm import tqdm

label_dict = train_data_original.label_dict
id2dict = {v: k for k, v in label_dict.items()} # reverse the dict_map: from k:v to v:k

def train(model, train_data, val_data, num_epochs, learning_rate, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_data,
              batch_size = batch_size,
              collate_fn = collate_func_train,
              shuffle = True)
    val_loader = DataLoader(dataset=val_data, 
                batch_size = batch_size, 
                collate_fn=collate_func_train)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    model.to(device)
    train_ls, val_ls = [], []
    for i_epoch in range(num_epochs):
        # ------ Train -----------
        model.train()
        total_acc_train, total_loss_train = 0, 0
        
        # try-except structure is used for 'tqdm' bug
        try:
            with tqdm(train_loader, desc="Iter_train:", ncols=100) as t:
                for batch in t:
                    labels_ids = batch["labels_ids"].to(device)
                    classes = batch["classes"].to(device)
                    ner_logit, sa_logit = model(device, batch)
                    
                    # calculate loss
                    ner_loss = loss_fct(ner_logit.view(-1,num_label), labels_ids.view(-1))
                    sa_loss = loss_fct(sa_logit, classes) #classes有点怪，跟num_label对不上
                    loss = ner_loss + sa_loss # for backward()
                    total_loss_train += loss.item()
                    
                    # calculate accuracy
                    ner_acc = (ner_logit.argmax(dim=-1) == labels_ids).sum()
                    sa_acc = (sa_logit.argmax(dim=-1) == classes).sum()
                    acc = ner_acc + sa_acc
                    total_acc_train += acc.item()

                    # model update 
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # update tqdm's progress bar using new loss value
                    t.set_description("Iter_train (loss=%5.3f)" % loss.item()) # .item(): higher precision
                    
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        
        # ------ Valid -----------
        # switch th eval mode for valid dataset
        model.eval()
        total_acc_val, total_loss_val = 0, 0

        try:
            with tqdm(val_loader, desc="Iter_valid", ncols=100) as t:
                for batch in t:
                    labels_ids = batch["labels_ids"].to(device)
                    classes = batch["classes"].to(device)
                    # no backward(), so no autograd, which will consume memory
                    with torch.no_grad():
                        ner_logit, sa_logit = model(device, batch)

                    ner_loss = loss_fct(ner_logit.view(-1,num_label), labels_ids.view(-1))
                    sa_loss = loss_fct(sa_logit, classes) 
                    loss = ner_loss + sa_loss 
                    total_loss_val += loss.item()

                    ner_acc = (ner_logit.argmax(dim=-1) == labels_ids).sum()
                    sa_acc = (sa_logit.argmax(dim=-1) == classes).sum()
                    acc = ner_acc + sa_acc
                    total_acc_val += acc.item()
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        
        train_loss_rate = total_loss_train / len(train_data)
        train_acc_rate = total_acc_train / len(train_data)
        val_loss_rate = total_loss_val / len(val_data)
        val_acc_rate = total_acc_val / len(val_data)

        train_ls.append((train_loss_rate, train_acc_rate))
        val_ls.append((val_loss_rate, val_acc_rate))
        
    # return all epoch's results
    return train_ls, val_ls



def predict(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_dict = test_data_original.label_dict
    id2dict = {v: k for k, v in label_dict.items()} # reverse the dict_map: from k:v to v:k
    test_loader = DataLoader(test_data_original,
                             batch_size = batch_size, 
                             collate_fn=collate_func_test)
    
    ner_predict = []
    sa_predict = []
    model.to(device)
    model.eval() 
    try:
        with tqdm(test_loader, desc="Iter_predict", ncols=100) as t:
            for batch in t:
                len_list = batch["len"].to(device)
                with torch.no_grad():
                    ner_logit, sa_logit = model(device, batch)
                ner = torch.argmax(ner_logit, dim=-1).cpu().numpy().tolist()
                sa = torch.argmax(sa_logit, dim=-1).cpu().numpy().tolist()
                
                for idy in range(len(ner)):
                    ner_seq = ner[idy][1:len_list[idy]+1] # remove [CLS] and [PAD] etc. in terms of the 'len_list'
                    ner_res = [id2dict[idx] for idx in ner_seq]
                    ner_predict.append(' '.join(ner_res))
                    
                sa_predict.extend(sa)
                
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
    
    return ner_predict, sa_predict


model = BERTLinearModel()
k_fold(model=model, k=10,train_data_original=train_data_original,
       num_epochs = 2, learning_rate = 1e-5, batch_size = 16)

#sava train parameters
torch.save(model.state_dict(), 'model.pth')

# load parameters of previous saved model
model.load_state_dict(torch.load('model.pth')) 
ner_predict, sa_predict = predict(model)

result_data=[]
for idx,(bio,cls) in enumerate(zip(ner_predict, sa_predict)):
    result_data.append([idx,bio,cls])

submit=pd.DataFrame(result_data,columns=['id','BIO_anno','class'])
submit.to_csv('submission.csv', index=False)
# submit.head(10)

