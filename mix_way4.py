import datetime

import numpy as np
import pandas as pd
from torch import nn
import random
from earlystop import EarlyStopping
from linear_mix import *
from method import *
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_mix4(model, epochs, dl_train, dl_valid, log_step_freq, optimizer, metric_func, TEXT,mix,alp,num_class=2):
    early_stopping = EarlyStopping(patience=40, verbose=True)
    metric_name = "auc"
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("==========" * 8 + "%s" % nowtime)

    embedding = nn.Embedding(num_embeddings=len(TEXT.vocab), embedding_dim=300, padding_idx=1).to(device) 
    embedding.weight.data.copy_(TEXT.vocab.vectors)
    embedding.weight.requires_grad = False



    for epoch in range(1, epochs + 1):

  
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):

            torch.cuda.empty_cache()


            features = features.to(device)
            features = embedding(features).to(device)
            labels = labels.to(device) 
            new_felist=[]
            new_lalist=[]
            
            t = len(features)
           
            if(t <10):
                continue
            new_felist.append(features)
            new_lalist.append(labels)  
            for i in range(t):
                PC=random.sample(range(t),min(mix,t))

                for j in PC:
                    #
                    if i==j :
                        continue

                    #     continue
                    fe_1 = features[i]
                    fe_2 = features[j]
                    la_1=labels[i]
                    la_2=labels[j]  
                    # for alpha in np.arange(-0.2,1.2,alp):
                    alpha=np.random.beta(alp,alp)
                        # alpha=0.1+0.1*o
                    #         print(alpha)
                    # if -0.04<alpha<0.04 or 0.96<alpha<1.04:
                    #     continue       
                    fe_new,la_new = new_global_mixup(fe_1 ,la_1,fe_2,la_2, features, labels,alpha,num_class)
                    # if(torch.sum(fe_new*fe_new)==0):
                    #     continue
                    new_felist.append(fe_new.unsqueeze(0))
                    new_lalist.append(la_new.unsqueeze(0))
                        
            new_fe=torch.cat(new_felist,0)
            new_la=torch.cat(new_lalist,0)
            # for s,i in enumerate (new_fe):
            #     if(torch.sum(i*i)==0):
            #         print(i,new_la[s],i.shape,new_la[s].shape)
            train_step_mix(model,new_fe, new_la, optimizer, metric_func)

            # print(features.shape,labels.shape)
            # print(model(features)[0])
            loss, metric = train_step_mix(model, features, labels, optimizer, metric_func)


            loss_sum += loss
            metric_sum += metric

            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") % (step, loss_sum / step, metric_sum / step))


        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            features = features.to(device)
            features = embedding(features).to(device)
            labels = labels.to(device)
            val_loss, val_metric = valid_step(model, features, labels, optimizer, metric_func)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info


        early_stopping(val_loss_sum / val_step, model)
    
        if early_stopping.early_stop:
            print("Early stopping")
        
            break
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "==========" * 8 + "%s" % nowtime)

    print("Finished Training...")

    return dfhistory

