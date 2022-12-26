
import datetime

import numpy as np
import pandas as pd
from torch import nn

from earlystop import EarlyStopping
from linear_mix import *
from method import *
import random
from sklearn.metrics import accuracy_score, f1_score


class lstmNet(nn.Module):
    def __init__(self, TEXT, num_classess):
        super(lstmNet, self).__init__()
        self.TEXT = TEXT
   
        self.embedding = nn.Embedding(num_embeddings=len(TEXT.vocab),
                                      embedding_dim=300,
                                      padding_idx=1)
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad = False
       

        self.LSTM1=nn.LSTM(input_size=300,hidden_size=64,num_layers=2,batch_first=True,dropout=0.3,bidirectional=True)
        self.dense=nn.Sequential()

        self.dense.add_module("linear", nn.Linear(128, 20))
        self.dense.add_module("relu_3", nn.ReLU())
        self.dense.add_module("linear2", nn.Linear(20,num_classess))  
        self.dense.add_module("softmax", nn.Softmax(1))
        # self.norm=nn.LayerNorm(128)
    def Text(TEXT):
        self.TEXT = TEXT

    def forward(self, x):
        
        x = self.embedding(x)
        x = self.LSTM1(x)[0]
        # x=self.norm(x)
        x=torch.mean(x,dim=1)

        y = self.dense(x)
        #         y=torch.argmax(y,1)+1
        #         y=y.unsqueeze(1)
        return y

class cnnNet_mix(nn.Module):
    def __init__(self, TEXT,num_classess):
        super(cnnNet_mix, self).__init__()

   

        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_1",
            nn.Conv1d(in_channels=300, out_channels=128, kernel_size=5))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module("pool_1",
                             nn.AdaptiveMaxPool1d(1))  
        #         self.conv.add_module("relu_2",nn.ReLU())

        self.dense = nn.Sequential()

        self.dense.add_module("linear", nn.Linear(128, 20))
        self.dense.add_module("relu_3", nn.ReLU())
        self.dense.add_module("linear2", nn.Linear(20,num_classess))  
        self.dense.add_module("softmax", nn.Softmax(1))

    #         self.dense.add_module("linear",nn.Linear(2,1))
    #         self.dense.add_module("sigmoid",nn.Sigmoid())

    def Text(TEXT):
        pass

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze()
        y = self.dense(x)
 
        return y



class lstmNet_hidden(nn.Module):
    def __init__(self, TEXT,num_classess):
        super(lstmNet_hidden, self).__init__()

        self.LSTM1=nn.LSTM(input_size=300,hidden_size=64,num_layers=2,batch_first=True,dropout=0.3,bidirectional=True)
        self.dense=nn.Sequential()

        self.dense.add_module("linear", nn.Linear(128, 20))
        self.dense.add_module("relu_3", nn.ReLU())
        self.dense.add_module("linear2", nn.Linear(20,num_classess))  
        self.dense.add_module("softmax", nn.Softmax(1))
        # self.norm=nn.LayerNorm(128)
    def Text(TEXT):
        self.TEXT = TEXT

    def hidden(x,labels):
        x = self.LSTM1(x)[0]
        # x=self.norm(x)
        x=torch.mean(x,dim=1)
        new_felist=[]
        new_lalist=[]
        new_felist.append(x)
        new_lalist.append(x)
        for (fe_1,la_1) in zip(x, labels):
            

            PC=random.sample(range(len(x)),4)
            for j in PC:
                alpha=np.random.beta(1, 1)
                fe_2=x[j]
                la_2=labels[j]
                fe_new,la_new = in_mixup(fe_1 ,la_1,fe_2,la_2, alpha)    
                new_felist.append(fe_new.unsqueeze(0))
                new_lalist.append(la_new.unsqueeze(0))
        new_fe=torch.cat(new_felist,0)
        new_la=torch.cat(new_lalist,0)
        return forward(new_fe),new_la
    def forward(self, x):

        x = self.LSTM1(x)[0]
        # x=self.norm(x)
    


        y = self.dense(new_fe)
        #         y=torch.argmax(y,1)+1
        #         y=y.unsqueeze(1)
        return y



def train_step_hidden(model, features, labels, optimizer, metric_func):

   
    model.train()

    
    optimizer.zero_grad()

    

    predictions,lab_mix = model.hidden(features,labels)
    #     print(predictions,labels)
    log_prob = torch.nn.functional.log_softmax(predictions, dim=1)
    #     print(log_prob)
    loss = -torch.sum(log_prob * lab_mix) / len(lab_mix)  
    #     print(loss)
    metric = metric_func(predictions, labels)
  
    loss.backward()
    optimizer.step()

    return loss.item(),metric.item()