import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from torchkeras import summary
import datetime
import string, re
import torchtext
# import pytorch_lightning as pl
from torchkeras import LightModel
import torchkeras
from sklearn.metrics import accuracy_score, f1_score
from earlystop import EarlyStopping

MAX_LEN = 128
MAX_WORDS = 10000
BATCH_SIZE = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filterLowFreqWords(arr, vocab):
    arr = [[x if x < MAX_WORDS else 0 for x in example] for example in arr]
    return arr


def load_da(path1, train1, valid1, test1, format1='csv'):

    #分词方法
    tokenizer = lambda x: re.sub('[%s]' % string.punctuation, "", x).split(
        " "
    )  #re。sub：替换x中的所有字符(string.punctuation)用""  '[%s]'%string.punctuation：用后面的字符串替换前面的
    #如果字符串以r开头表示正则表达式   如 r'[0-9]'
    #过滤掉低频词

    # train1 = train1 + '.' + format1
    # valid1 = valid1 + '.' + format1
    # test1 = test1 + '.' + format1
    #1,定义各个字段的预处理方法
    TEXT = torchtext.legacy.data.Field(sequential=True,
                                       tokenize=tokenizer,
                                       lower=True,
                                       fix_length=MAX_LEN,
                                       postprocessing=filterLowFreqWords)

    LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

    #2,构建表格型dataset
    #torchtext.data.TabularDataset可读取csv,tsv,json等格式
    ds_train, ds_valid, ds_test = torchtext.legacy.data.TabularDataset.splits(
        path=path1,
        train=train1,
        validation=valid1,
        test=test1,
        format=format1,
        fields=[('label', LABEL), ('text', TEXT)],
        skip_header=False)

    #     for train in ds_train:
    #         print(train.label)
    #         print(type(torch.tensor(int(train.label))))
    #         train.label=torch.nn.functional.one_hot(torch.tensor(int(train.label))-1, 2).float()
    #         train.label=int(train.label)
    #         print(train.label)
    #     for valid in ds_valid:
    #         valid.label=torch.nn.functional.one_hot(valid.label-1, 2).float()
    #vectors = Vectors(name='myveglove.6B.200d.txt')
    TEXT.build_vocab(ds_train, vectors='glove.6B.300d')

    #4,构建数据管道迭代器
    train_iter, valid_iter, test_iter = torchtext.legacy.data.Iterator.splits(
        (ds_train, ds_valid, ds_test),
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        shuffle=True,
        batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE))

    return train_iter, valid_iter, test_iter, TEXT


class DataLoader:
    def __init__(self, data_iter, num_classes, begin):
        self.data_iter = data_iter
        self.length = len(data_iter)
        self.num_class = num_classes
        self.begin = begin

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意：此处调整features为 batch first，并调整label的shape和dtype
        for batch in self.data_iter:
            yield (torch.transpose(batch.text, 0, 1),
                   (torch.nn.functional.one_hot(batch.label - self.begin,
                                                self.num_class).float())
                   )  #num_classes


#             torch.nn.functional.one_hot(valid.label-1, 2).float()
class Model(LightModel):

    #loss,and optional metrics
    def shared_step(self, batch) -> dict:
        x, y = batch
        prediction = self(x)
        loss = nn.BCELoss()(prediction, y)
        preds = torch.where(prediction > 0.5, torch.ones_like(prediction),
                            torch.zeros_like(prediction))
        acc = pl.metrics.functional.accuracy(preds, y)
        dic = {"loss": loss, "accuracy": acc}
        return dic

    #optimizer,and optional lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=0.02)
        return optimizer


class lstmNet(nn.Module):
    def __init__(self, TEXT, num_classess):
        super(lstmNet, self).__init__()
        self.TEXT = TEXT
        #设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings=len(TEXT.vocab),
                                      embedding_dim=300,
                                      padding_idx=1)  #填充padding的index
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad = False
        # 设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量

        self.LSTM1=nn.LSTM(input_size=300,hidden_size=64,num_layers=2,batch_first=True,dropout=0.3,bidirectional=True)
        self.dense=nn.Sequential()

        self.dense.add_module("linear", nn.Linear(128, 20))
        self.dense.add_module("relu_3", nn.ReLU())
        self.dense.add_module("linear2", nn.Linear(20,num_classess))  # 2=num_classess
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




class cnnNet(nn.Module):
    def __init__(self, TEXT, num_classess):
        super(cnnNet, self).__init__()
        self.TEXT = TEXT
        #设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings=len(TEXT.vocab),
                                      embedding_dim=300,
                                      padding_idx=1)  #填充padding的index
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad = False

        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_1",
            nn.Conv1d(in_channels=300, out_channels=128, kernel_size=5))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module("pool_1", nn.AdaptiveMaxPool1d(1))  #kernel 几个里取最大
        #         self.conv.add_module("relu_2",nn.ReLU())

        self.dense = nn.Sequential()

        self.dense.add_module("linear", nn.Linear(128, 20))
        self.dense.add_module("relu_3", nn.ReLU())
        self.dense.add_module("linear2",
                              nn.Linear(20, num_classess))  #2=num_classess
        self.dense.add_module("softmax", nn.Softmax(1))


#         self.dense.add_module("linear",nn.Linear(2,1))
#         self.dense.add_module("sigmoid",nn.Sigmoid())

    def Text(self, TEXT):

        self.TEXT = TEXT

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze()
        y = self.dense(x)
        #         y=torch.argmax(y,1)+1
        #         y=y.unsqueeze(1)
        return y


def train_step(model, features, labels, optimizer, metric_func):

    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    optimizer.zero_grad()

    # 正向传播求损失

    predictions = model(features)
    #     print(predictions,labels)
    log_prob = torch.nn.functional.log_softmax(predictions, dim=1)
    #     print(log_prob)
    loss = -torch.sum(log_prob * labels) / len(labels)  #交叉熵
    #     print(loss)
    metric = metric_func(predictions, labels)
    #     print(metric)
    # 反向传播求梯度
    loss.backward()
    optimizer.step()

    return loss.item(), metric.item()


def train_step_mix(model, features, labels, optimizer, metric_func):

    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    optimizer.zero_grad()

    # 正向传播求损失

    predictions = model(features)
    #     print(predictions,labels)
    log_prob = torch.nn.functional.log_softmax(predictions, dim=1)
    #     print(log_prob)
    loss = -torch.sum(log_prob * labels) / len(labels)  # 交叉熵
    #     print(loss)
    metric = metric_func(predictions, labels)
    #     print(metric)
    # 反向传播求梯度
    loss.backward()
    optimizer.step()

    return loss.item(), metric.item()


def valid_step(model, features, labels, optimizer, metric_func):

    # 预测模式，dropout层不发生作用
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        predictions = model(features)
        log_prob = torch.nn.functional.log_softmax(predictions, dim=1)
        #     print(log_prob)
        loss = -torch.sum(log_prob * labels) / len(labels)  #交叉熵
        #     print(loss)
        metric = metric_func(predictions, labels)

    return loss.item(), metric.item()

    ##############
    #######################################################################################
    ###############func


func_AUC = lambda y_pred, y_true: accuracy_score(
    np.argmax(y_true.data.cpu().numpy(), 1),
    np.argmax(y_pred.data.cpu().numpy(), 1))
func_F1_macro = lambda y_pred, y_true: f1_score(
    np.argmax(y_true.data.cpu().numpy(), 1),
    np.argmax(y_pred.data.cpu().numpy(), 1),
    average='macro')

func_F1_micro = lambda y_pred, y_true: f1_score(
    np.argmax(y_true.data.cpu().numpy(), 1),
    np.argmax(y_pred.data.cpu().numpy(), 1),
    average='micro')

func_F1_weight = lambda y_pred, y_true: f1_score(
    np.argmax(y_true.data.cpu().numpy(), 1),
    np.argmax(y_pred.data.cpu().numpy(), 1),
    average='weighted')


def test_step(model, features, labels, metric_func):

    # 预测模式，dropout层不发生作用
    model.eval()
    # 关闭梯度计算
    metric = []
    with torch.no_grad():
        predictions = model(features)
        #     print(log_prob)

        #     print(loss)
        metric_auc = metric_func(predictions, labels)
        metric_F1_macro = func_F1_macro(predictions, labels)
        metric_F1_micro = func_F1_micro(predictions, labels)
        metric_F1_weight = func_F1_weight(predictions, labels)
        metric.extend(
            [metric_auc, metric_F1_macro, metric_F1_micro, metric_F1_weight])

    return metric


def train_model(model,
                epochs,
                dl_train,
                dl_valid,
                log_step_freq,
                optimizer,
                metric_func,
                TEXT,
                mix,
                alp,
                num_class=2):
    early_stopping = EarlyStopping(patience=40, verbose=True)
    metric_name = 'auc'
    dfhistory = pd.DataFrame(columns=[
        "epoch", "loss", metric_name, "val_loss", "val_" + metric_name
    ])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)

    for epoch in range(1, epochs + 1):

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):
            features = features.to(device)
            labels = labels.to(device)  ###改mixup
            loss, metric = train_step(model, features, labels, optimizer,
                                      metric_func)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            features = features.to(device)
            labels = labels.to(device)
            val_loss, val_metric = valid_step(model, features, labels,
                                              optimizer, metric_func)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        # 打印epoch级别日志
        early_stopping(val_loss_sum / val_step, model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f")
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    print('Finished Training...')

    return dfhistory


def plot_metric(dfhistory, metric, name):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.switch_backend('agg')
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(name)
    # plt.show()


