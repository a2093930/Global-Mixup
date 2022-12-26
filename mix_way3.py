import datetime

import numpy as np
import pandas as pd
from torch import nn

from earlystop import EarlyStopping
from linear_mix import *
from method import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_mix3(model, epochs, dl_train, dl_valid, log_step_freq, optimizer, metric_func, TEXT,mix,alp,num_class=2):
    early_stopping = EarlyStopping(patience=30, verbose=True)
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

            t = len(features)
            # print(t)
            index = mix
            ind = int(t / index)
            
         
            for i in range(index):
          
                if i== index-1:
                    continue
                j=np.random.randint(i+1,index)
                fe_1 = features[i * ind : (i + 1) * ind]
                fe_2 = features[j * ind : (j + 1) * ind]
                la_1=labels[i*ind:(i+1)*ind]
                la_2=labels[j*ind:(j+1)*ind]   
                for alpha in np.arange(-0.2 ,1.2,alp):
                #         print(alpha)
                    if -0.04<alpha<0.04 or 0.96<alpha<1.04:
                        continue       
                    la_global,fe_global, la_mix,fe_mix = global_mixup(fe_1 ,la_1,fe_2,la_2, features, labels,alpha,num_class)
                    # print(fe_mix.shape)
                    # print(fe_1.shape[0],fe_mix,alpha)
                    if(fe_mix.shape[0]>1):
                        # print("s")
                        train_step_mix(model, fe_mix, la_mix, optimizer, metric_func)
                    # print(fe_global.shape,features.shape)
                    if(fe_global.shape[0]>1):
                        train_step_mix(model, fe_global, la_global, optimizer, metric_func)

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