from method import *
from linear_mix import *
import random
import csv
from mix import *
from mix_way2 import *
from mix_way3 import *
from mix_way4 import *
from hidden import *
import argparse
import pandas as pd
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_auc(dl_test, model, metric_func, TEXT):
    step = 1
    test_metric_sum = np.array([0, 0, 0, 0]).astype(float)
    for step, (features, labels) in enumerate(dl_test, 1):
        features = features.to(device)
        labels = labels.to(device)
        test_metric = np.array(test_step(model, features, labels, metric_func))

        test_metric_sum += test_metric


    return test_metric_sum / step


def run(name,
        path,
        train,
        valid,
        test,
        model_in=cnnNet,
        epochs=2000,
        format1='csv',
        fuc=train_model,
        fuc_t=test_auc,
        mix=16,
        alp=0.1,
        num_classess=2,
        begin=1):

    train_iter, valid_iter, test_iter, TEXT = load_da(path,
                                                      train,
                                                      valid,
                                                      test,
                                                      format1=format1)
    dl_train = DataLoader(train_iter, num_classess, begin)

    dl_valid = DataLoader(valid_iter, num_classess, begin)
    dl_test = DataLoader(test_iter, num_classess, begin)

    model = model_in(TEXT, num_classess).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    metric_func = lambda y_pred, y_true: accuracy_score(
        np.argmax(y_true.data.cpu().numpy(), 1),
        np.argmax(y_pred.data.cpu().numpy(), 1))
    dfhistory = fuc(model,
                    epochs,
                    dl_train,
                    dl_valid,
                    log_step_freq=50,
                    optimizer=optimizer,
                    metric_func=metric_func,
                    TEXT=TEXT,
                    mix=mix,
                    alp=alp,
                    num_class=num_classess)

    aucc = fuc_t(dl_test, model, metric_func, TEXT)

    return aucc



def test_auc_mix(dl_test, model, metric_func, TEXT):

    embedding = nn.Embedding(num_embeddings=len(TEXT.vocab),
                             embedding_dim=300,
                             padding_idx=1).to(device)  
    embedding.weight.data.copy_(TEXT.vocab.vectors)
    embedding.weight.requires_grad = False

    step = 1
    test_metric_sum = np.array([0, 0, 0, 0]).astype(float)
    for step, (features, labels) in enumerate(dl_test, 1):
        features = features.to(device)
        features = embedding(features).to(device)
        labels = labels.to(device)
        test_metric = np.array(test_step(model, features, labels, metric_func))

        test_metric_sum = test_metric_sum + test_metric

    return test_metric_sum / step  #


def file_split(path, name):
    dftrain_raw = pd.read_csv(path + '/' + name + '.csv', header=None)
    file_len = len(dftrain_raw)

    valid = '_valid'
    train = '_train'
    f = open(path + '/' + name + valid + '.csv', 'w', encoding='utf-8')
    d = open(path + '/' + name + train + '.csv', 'w', encoding='utf-8')

 

    l = random.sample(range(file_len), int(0.1 * file_len))

    t = [x for x in range(file_len)]
    for i in l:
        t.remove(i)
    print(file_len, len(l), len(t))

    csv_writer = csv.writer(f)
    for i in l:
        csv_writer.writerow(dftrain_raw.loc[i])
    csv_writer = csv.writer(d)
    for i in t:
        csv_writer.writerow(dftrain_raw.loc[i])


    f.close()
    d.close()
    return name, name + valid, name + train


def get_datalist(folders, ways, alphas, path):
    train_data = []
    for folder in folders:
        for way in ways:
            for alpha in alphas:
                filename = 'train_' + way + '_' + str(alpha)
                data_path = path + folder
                name, valid_name, train_name = file_split(data_path, filename)
                train_data.append([name, valid_name, train_name])
    return train_data


def train(dir, train, val, test, mix, alpha, classes, begins):
    file = open('' + datetime.datetime.now().strftime('%m-%d %H:%M') + '.txt',
                'w',
                encoding='utf-8',
                buffering=1)

    train_list = ['sst-1_train']
    val_list = ['sst-1_val']
    alph_list = [4, 8, 8, 1]
    ind_list = [4, 8, 16, 2]
    file.write(' =================' * 8 + '\n' + '\n' + train + '\t' + val +
               '\n')

    auc = run('base',
              dir,
              train,
              val,
              test,
              model_in=cnnNet_mix,
              epochs=500,
              fuc=train_model_mix4,
              fuc_t=test_auc_mix,
              mix=mix,
              alp=alpha,
              num_classess=classes,
              begin=begins)  
    print(auc)
    file.write('loss-fuc:auc mix=' + str(mix) + ',alpha=' + str(alpha) + '\t' +
               str(auc) + '\n')



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",
                    required=True,
                    type=str,
                    help="The input data dir")
    ap.add_argument("--input_train",
                    required=True,
                    type=str,
                    help="input file of train data. Should be .csv file")
    ap.add_argument("--input_val",
                    required=True,
                    type=str,
                    help="input file of val data. Should be .csv file")
    ap.add_argument("--input_test",
                    required=True,
                    type=str,
                    help="input file of test data. Should be .csv file")
    ap.add_argument("--num_mix",
                    required=True,
                    type=int,
                    help="number of augmented sentences per original sentence")
    ap.add_argument("--alpha",
                    required=True,
                    type=float,
                    help="Beta parameter")
    ap.add_argument("--classes",
                    required=True,
                    type=int,
                    help="number of target labels")
    ap.add_argument("--begins",
                    required=True,
                    type=int,
                    help="Minimum value of the target label")
    args = ap.parse_args()

    train(args.dir, args.input_train, args.input_val, args.input_test,
          args.num_mix, args.alpha, args.classes, args.begins)
