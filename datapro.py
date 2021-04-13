import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset
from sklearn import preprocessing

import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas
import os
import numpy as np
def getcluster():
    c = []
    c1 = [5, 8, 13, 26, 27, 29, 31, 32, 33, 37, 39, 40, 42, 46, 50]
    c2 = [0, 10, 15, 16, 17, 21, 24, 30, 34, 43, 51]
    c3 = [4, 20, 25, 28, 38]
    c4 = [1, 2, 7, 18, 19, 35, 41, 44, 45, 49]
    c5 = [3, 6, 9, 11, 12, 14, 22, 23, 36, 47, 48]
    c.append(c1)
    c.append(c2)
    c.append(c3)
    c.append(c4)
    c.append(c5)
    return c
def normalized_data(data):
    mu = data.mean(axis = 0)
    std = data.std(axis = 0)

    std = std
    t = (data - mu)/std

    return t, mu , std

def timewindow(samples ,windowlength ):

    sample_time = []
    for i in range(samples.shape[0]-windowlength+1):
        i = i+windowlength
        term = samples[i-windowlength:i,:]
        sample_time.append(np.concatenate(term,axis=0))
    sample_time_np = np.array(sample_time).reshape(-1,52*windowlength)
    return sample_time_np
def readnormal():
    filepath = r'D:\TEDATA\TEST'
    dpath = 'd00_te.dat'
    dDatapath = os.path.join(filepath, dpath)
    with open(dDatapath, 'r') as fr:
        data = fr.read()
        data = np.fromstring(data, dtype=np.float32, sep='   ')
        data = data.reshape(-1, 52)

    samples_Nocomp = data
    stdsc_00 = StandardScaler()
    mms_nor = MinMaxScaler()
    samples_Nocomp = stdsc_00.fit_transform(data)
    return samples_Nocomp
def readFile(IS_TRAIN,batch_size=20, window = False,windowl=2):
    c =getcluster()
    mms_00 = MinMaxScaler()
    stdsc_00 = StandardScaler()

    DATA = []
    LABEL = []
    filepath = r'D:\TEDATA\TEST'
    # dpath = ['d10train.csv','d11.dat', 'd16train.csv', 'd18.dat']
    dpath =['d00train.csv', 'd01train.csv', 'd02train.csv', 'd04train.csv', 'd05train.csv', 'd06train.csv', 'd07train.csv', 'd08train.csv','d10train.csv', 'd11train.csv',
      'd12train.csv', 'd13train.csv', 'd14train.csv','d16train.csv', 'd17train.csv', 'd18train.csv', 'd19train.csv', 'd20train.csv']
    i = 0
    for Datapath in dpath:

        dDatapath = os.path.join(filepath, Datapath)

        if Datapath == 'd00train.csv':


            data = np.loadtxt(dDatapath, dtype=np.float32, delimiter=',')
            samples_Nocomp = data


        else:

            data = np.loadtxt(dDatapath, dtype=np.float32, delimiter=',')
            samples_Nocomp = data



        DATA.append(samples_Nocomp)

        label = i * np.ones(samples_Nocomp.shape[0])
        LABEL.append(label)
        i += 1

    data_TE_train = np.concatenate(np.array(DATA), axis=0)
    label_TE_train_s = np.concatenate(np.array(LABEL), axis=0)

    i=0
    DATA = []
    LABEL = []
    filepath = r'D:\TEDATA\TRAIN'

    dpath = ['d00test.csv', 'd01test.csv', 'd02test.csv', 'd04test.csv', 'd05test.csv','d06test.csv', 'd07test.csv','d08test.csv', 'd10test.csv','d11test.csv',
             'd12test.csv','d13test.csv','d14test.csv','d16test.csv','d17test.csv','d18test.csv','d19test.csv','d20test.csv']
    for Datapath in dpath:

        dDatapath = os.path.join(filepath,Datapath)

        if Datapath == 'd00test.csv':

            data = np.loadtxt(dDatapath, dtype=np.float32, delimiter=',')
            samples_Nocomp = data


        else:

            data = np.loadtxt(dDatapath, dtype=np.float32, delimiter=',')
            samples_Nocomp = data

        DATA.append(samples_Nocomp)
        label = i * np.ones(samples_Nocomp.shape[0])
        LABEL.append(label)
        i +=1
    data_TE_test = np.concatenate(np.array(DATA), axis=0)
    label_TE_test_s = np.concatenate(np.array(LABEL), axis=0)
    i = 0
    DATA = []
    LABEL = []
    filepath = r'D:\TEDATA\TRAIN'

    dpath = ['d00validation.csv', 'd01validation.csv', 'd02validation.csv', 'd04validation.csv', 'd05validation.csv', 'd06validation.csv', 'd07validation.csv',
             'd08validation.csv', 'd10validation.csv', 'd11validation.csv',
             'd12validation.csv', 'd13validation.csv', 'd14validation.csv', 'd16validation.csv', 'd17validation.csv', 'd18validation.csv', 'd19validation.csv',
             'd20validation.csv']
    for Datapath in dpath:

        dDatapath = os.path.join(filepath, Datapath)

        if Datapath == 'd00validation.csv':

            data = np.loadtxt(dDatapath, dtype=np.float32, delimiter=',')
            samples_Nocomp = data

        else:

            data = np.loadtxt(dDatapath, dtype=np.float32, delimiter=',')
            samples_Nocomp = data

            # samples_Nocomp = stdsc_00.transform(samples_Nocomp)
            # samples_Nocomp = mms_00.fit_transform(samples_Nocomp)

            # samples_Nocomp = samples_Nocomp[160:, :]
        DATA.append(samples_Nocomp)
        label = i * np.ones(samples_Nocomp.shape[0])
        LABEL.append(label)
        i += 1

    data_TE_valid = np.concatenate(np.array(DATA), axis=0)
    label_TE_valid_s = np.concatenate(np.array(LABEL), axis=0)
    stdsc = StandardScaler()
    mms = MinMaxScaler()
    # data_TE_train = mms.fit_transform(data_TE_train)

    data_TE_train = stdsc.fit_transform(data_TE_train)
    data_TE_test = stdsc.transform(data_TE_test)
    data_TE_valid = stdsc.transform(data_TE_valid)



    # data_TE_test = mms.transform(data_TE_test)




    if window==True:
        samplenum = []
        samplenum.append(0)
        orglendata = []
        data_w_train = []
        label_w_train =[]
        for i in range(len(dpath)):

            samplenum.append(np.sum(label_TE_train_s==i)+samplenum[i])
            orglendata.append(data_TE_train[samplenum[i]:samplenum[i+1]])
        for i in range(len(orglendata)):
            single = np.array(orglendata[i]).reshape(-1,52)
            Data = timewindow(single, windowlength=windowl)
            label = i * np.ones(Data.shape[0])
            data_w_train.append(Data)
            label_w_train.append(label)
        data_TE_train = np.concatenate(np.array(data_w_train), axis=0)
        # data_TE_train = stdsc_00.fit_transform(data_TE_train)
        label_TE_train = np.concatenate(np.array(label_w_train), axis=0)
        samplenum = []
        samplenum.append(0)
        orglendata = []
        data_w_test = []
        label_w_test = []
        for i in range(len(dpath)):
            samplenum.append(np.sum(label_TE_test_s == i) + samplenum[i])
            orglendata.append(data_TE_test[samplenum[i]:samplenum[i + 1]])
        for i in range(len(orglendata)):
            single = np.array(orglendata[i]).reshape(-1, 52)
            Data = timewindow(single, windowlength=windowl)
            label = i * np.ones(Data.shape[0])
            data_w_test.append(Data)
            label_w_test.append(label)
        data_TE_test = np.concatenate(np.array(data_w_test), axis=0)
        # data_TE_test = stdsc_00.fit_transform(data_TE_test)
        label_TE_test = np.concatenate(np.array(label_w_test), axis=0)
        samplenum = []
        samplenum.append(0)
        orglendata = []
        data_w_valid = []
        label_w_valid = []
        for i in range(len(dpath)):
            samplenum.append(np.sum(label_TE_valid_s == i) + samplenum[i])
            orglendata.append(data_TE_valid[samplenum[i]:samplenum[i + 1]])
        for i in range(len(orglendata)):
            single = np.array(orglendata[i]).reshape(-1, 52)
            Data = timewindow(single, windowlength=windowl)
            label = i * np.ones(Data.shape[0])
            data_w_valid.append(Data)
            label_w_valid.append(label)
        data_TE_valid = np.concatenate(np.array(data_w_valid), axis=0)
        # data_TE_test = stdsc_00.fit_transform(data_TE_test)
        label_TE_valid = np.concatenate(np.array(label_w_valid), axis=0)
    else:
        data_TE_train = data_TE_train
        label_TE_train = label_TE_train_s
        data_TE_test = data_TE_test
        label_TE_test = label_TE_test_s
        data_TE_valid = data_TE_valid
        label_TE_valid = label_TE_valid_s

    if IS_TRAIN == 'Train':
        data_TE_T = torch.tensor(data_TE_train).float()
        label_TE_T = torch.tensor(label_TE_train).long()
        dataloader = DataLoader(TensorDataset(data_TE_T, label_TE_T), batch_size=batch_size, shuffle=True,
                                num_workers=1)
        return data_TE_train,label_TE_train, dataloader
    elif IS_TRAIN == 'Test':
        data_TE_T = torch.tensor(data_TE_test).float()
        label_TE_T = torch.tensor(label_TE_test).long()
        dataloader = DataLoader(TensorDataset(data_TE_T, label_TE_T), batch_size=batch_size, shuffle=False,
                                num_workers=1)
        return data_TE_test, label_TE_test, dataloader
    elif IS_TRAIN == 'Valid':
        data_TE_T = torch.tensor(data_TE_valid).float()
        label_TE_T = torch.tensor(label_TE_valid).long()
        dataloader = DataLoader(TensorDataset(data_TE_T, label_TE_T), batch_size=batch_size, shuffle=False,
                                num_workers=1)
        return data_TE_valid, label_TE_valid, dataloader




# a = readFile(IS_TRAIN=False,batch_size=20,window=True,windowl=20)
