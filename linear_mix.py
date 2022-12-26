import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def in_mixup(vector_x, label_x, vector_y, label_y, alpha):
    vector_mix = alpha * vector_x + (1 - alpha) * vector_y
    label_mix = alpha * label_x + (1 - alpha) * label_y

    return vector_mix, label_mix


def global_mixup(vector_x,
                 label_x,
                 vector_y,
                 label_y,
                 data_matrix,
                 label_matrix,
                 alpha,
                 num_class=2):

    label_global_list = []
    label_mix_list = []
    vector_mix_list = []
    vector_global = []
    label_global = []

    vector_mix = (alpha)*vector_x + (1 - alpha) * vector_y  #一个多句矩阵
    #对应alpha的vector矩阵
    label_mix = min(max(alpha, 0), 1) * label_x + min(max(
        (1 - alpha), 0), 1) * label_y
    label_mix = label_mix.tolist()
    vector_mix = vector_mix.tolist()
    mix_pop_list = []
    for i, label in enumerate(label_mix):
        #             print(i,label_mix[i],max(label_mix[i]),vector_mix[i])
        if (max(label_mix[i]) > 0.85):
            vector_global.append(vector_mix[i])
            label_global.append(label_mix[i])
            mix_pop_list.append(i)
#对应的数据结果，可以直接返回
##part1--需要全部找的：global

    for i in mix_pop_list[::-1]:
        #             print(i)
        label_mix.pop(i)
        vector_mix.pop(i)

    label_mix = torch.tensor(label_mix).to(device)
    vector_mix = torch.tensor(vector_mix).to(device)
    #         print(vector_mix,label_mix)#不同类部分
    vector_global = torch.Tensor(vector_global).to(device)
    # print(vector_global)

    for j in range(vector_global.shape[0]):  #每个句子
        #         print(vector_global.shape[0])
        list_ind = []
        tmp = None
        for i in range(int(data_matrix.shape[0])):  #计算全局 距离
            #             print((data_matrix.shape[0]))
            tmp = data_matrix[i]
            dist_mat = tmp - vector_global[j]
            dist_word=torch.sum(dist_mat * dist_mat, dim=0)
            dist_vector = torch.sqrt(torch.sum(dist_word*dist_word))
            list_ind.append(dist_vector)
        list_ind = torch.Tensor(list_ind).to(device)
        list_ind = -torch.sqrt(list_ind)
        list_soft = torch.softmax(list_ind,
                                  0).reshape(int(label_matrix.shape[0]), 1)
        label_global_mix = sum(torch.mul(list_soft, label_matrix), 0)

        label_global_list.append(label_global_mix.reshape(1, num_class))
    #global
    if label_global_list != []:
        label_global = torch.cat(label_global_list, 0)
    else:
        label_global = torch.Tensor([]).to(device)

    return label_global, vector_global, label_mix, vector_mix



#分出两类label,A直接过网络，B过out_mixup
#cat 二者 并return 是否分开训练？

def new_global_mixup(vector_x,
                 label_x,
                 vector_y,
                 label_y,
                 data_matrix,
                 label_matrix,
                 alpha,
                 num_class=2):

    mix_pop=False
    vector_mix =alpha*vector_x + (1 - alpha) * vector_y  #一个多句矩阵
    #对应alpha的vector矩阵
    label_mix = min(max(alpha, 0), 1) * label_x + min(max(
        (1 - alpha), 0), 1) * label_y
    
    if (max(label_mix) > 0.9):
        vector_global=vector_mix
        # vector_global=vector_mix.to(device)
        mix_pop=True      
    if(mix_pop==False):
        return vector_mix,label_mix
    #         print(vector_mix,label_mix)#不同类部分
    # print(vector_global)
    #每个句子
        #         print(vector_global.shape[0])
    list_ind = []
    tmp = None
#     print(MAT)
    for i in range(int(data_matrix.shape[0])):  #计算全局 距离
        #             print((data_matrix.shape[0]))
#         print(tmp,data_matrix[i].shape)
        tmp = torch.sum(data_matrix[i]*vector_global)
        dist_mat = torch.sum(vector_global * vector_global)
        global_mat= torch.sum(data_matrix[i] * data_matrix[i])                        
        dist_vector =tmp/ (torch.sqrt(dist_mat*global_mat)+1e-4)
        list_ind.append(dist_vector.item())
#         MAT=1/dist_vector.item()
#     print(MAT)
#     list_ind = -torch.sqrt(torch.Tensor(list_ind))
#     list_ind = -torch.Tensor(list_ind)+MAT
#     print(list_ind)
#     print(len(list_ind))
#     list_ind=torch.cat(list_ind,0)
    list_ind=torch.Tensor(list_ind).to(device)
    list_ind=(list_ind.topk(10,dim=0))
    label_mat=label_matrix.index_select(0,list_ind.indices.squeeze())
    list_soft = torch.softmax(list_ind.values.squeeze()*5.7,0).reshape(int(label_mat.shape[0]), 1)
    # print(list_ind.values,list_soft,label_mat)
    label_global = sum(torch.mul(list_soft, label_mat), 0)
    # print(label_mix,label_global)

    return vector_global,label_global