import datetime

import numpy as np
import pandas as pd
from torch import nn

from earlystop import EarlyStopping
from linear_mix import *
from method import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_mix(model, epochs, dl_train, dl_valid, log_step_freq,
                    optimizer, metric_func, TEXT, mix, alp):
    early_stopping = EarlyStopping(patience=30, verbose=True)
    metric_name = "auc"
    dfhistory = pd.DataFrame(columns=[
        "epoch", "loss", metric_name, "val_loss", "val_" + metric_name
    ])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("==========" * 8 + "%s" % nowtime)

    embedding = nn.Embedding(num_embeddings=len(TEXT.vocab),
                             embedding_dim=300,
                             padding_idx=1).to(device)  #
    embedding.weight.data.copy_(TEXT.vocab.vectors)
    embedding.weight.requires_grad = False

    for epoch in range(1, epochs + 1):

        
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):
            # torch.cuda.empty_cache()
            features = features.to(device)
            features = embedding(features).to(device)
            labels = labels.to(device)  ###改mixup

            t = len(features)
            # print(t)
            index = mix
            ind = int(t / index)
            for i in range(index):
                for j in range(i + 1, index):

                    fe_1 = features[i * ind:(i + 1) * ind]
                    la_1 = labels[i * ind:(i + 1) * ind]

                    fe_2 = features[j * ind:(j + 1) * ind]
                    la_2 = labels[j * ind:(j + 1) * ind]

                    for alpha in np.arange(alp, 1, alp):
                        fe_mix, la_mix = in_mixup(fe_1, la_1, fe_2, la_2,
                                                  alpha)
                        train_step_mix(model, fe_mix, la_mix, optimizer,
                                       metric_func)

            loss, metric = train_step_mix(model, features, labels, optimizer,
                                          metric_func)

          
            loss_sum += loss
            metric_sum += metric

            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))

        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            features = features.to(device)
            features = embedding(features).to(device)
            labels = labels.to(device)
            val_loss, val_metric = valid_step(model, features, labels,
                                              optimizer, metric_func)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

      
        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info

      
        early_stopping(val_loss_sum / val_step, model)
    
        if early_stopping.early_stop:
            print("Early stopping")
         
            break
        print(
            ("\nEPOCH = %d, loss = %.3f," + metric_name +
             "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
            % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "==========" * 8 + "%s" % nowtime)

    print("Finished Training...")

    return dfhistory


def train_step_mix(model, features, labels, optimizer, metric_func):

  
    model.train()


    optimizer.zero_grad()



    predictions = model(features)
    #     print(predictions,labels)
    log_prob = torch.nn.functional.log_softmax(predictions, dim=1)
    #     print(log_prob)
    loss = -torch.sum(log_prob * labels) / len(labels) 
    #     print(loss)
    metric = metric_func(predictions, labels)
    #     print(metric)

    loss.backward()
    optimizer.step()

    return loss.item(), metric.item()


class cnnNet_mix(nn.Module):
    def __init__(self, TEXT,num_classess):
        super(cnnNet_mix, self).__init__()

  

        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_1",
            nn.Conv1d(in_channels=300, out_channels=128, kernel_size=5))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module("pool_1",
                             nn.AdaptiveMaxPool1d(1))  # 
        #   

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
        #         y=torch.argmax(y,1)+1
        #         y=y.unsqueeze(1)
        return y



class lstmNet_mix(nn.Module):
    def __init__(self, TEXT,num_classess):
        super(lstmNet_mix, self).__init__()
        
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
        
        x = self.LSTM1(x)[0]
        # x=self.norm(x)
        x=torch.mean(x,dim=1)

        y = self.dense(x)
        #         y=torch.argmax(y,1)+1
        #         y=y.unsqueeze(1)
        return y