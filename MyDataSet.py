"""
自定义dataset类,返回一个映射到序号的样本
"""
import dataLoader 
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np

from torch.utils.data import Dataset,DataLoader
class MyDataSet(Dataset):
    def __init__(self,data_path,label_path,train=0) -> None:
        print("load data from %s"%data_path)
        if train:
            #加载数据(二层列表)和词表
            sentences,tags,vocab = dataLoader.loadData(data_path,get_vocab=True)
            labels = dataLoader.loadLabel(label_path)
            #把字和标签映射到词表中的序号
            MyDataSet.vocab2idx = {char: idx for idx, char in enumerate(vocab)}
            MyDataSet.label2idx = {char: idx for idx, char in enumerate(labels)}
            MyDataSet.vocab = vocab
            np.save("vocab2idx.npy",MyDataSet.vocab2idx)
            np.save("idx2label.npy",labels)
        else: #验证集
            sentences,tags,_ = dataLoader.loadData(data_path)
        self.sentences_idx = word2idx(sentences, MyDataSet.vocab2idx)
        self.tags_idx = word2idx(tags,MyDataSet.label2idx)

    def __len__(self):
        return len(self.sentences_idx)

    def __getitem__(self, index):
        return self.sentences_idx[index],self.tags_idx[index]

def word2idx(data,word2idx:dict):
    """
    把字根据word2idx字典映射到序号，返回两层的列表
    """
    index_result = []
    for line in data:
        index_result.append(torch.tensor(list(map(lambda x:word2idx.get(x,1),line)))) #词表中不存在的字序号为1(UNK)    
    return index_result

def idx2label(data,idx2label):
    """
    把预测的序号结果映射到label,返回两层列表
    """
    index_result = []
    for line in data:
        index_result.append(list(idx2label[line]))    
    return index_result

def collate_fn(batch):
    """
    将一个batch抽取出的样本padding, 参数是getitem函数返回的数据项的batch形成的列表.
    """
    (x,y) = zip(*batch) 
    #得到没有pad的数据的真实长度
    x_lens = [len(ele) for ele in x]
    x_pad = pad_sequence(x,batch_first=True,padding_value=0)#batch_first表明第一个维度是batch_size
    y_pad = pad_sequence(y,batch_first=True,padding_value=0)
    max_len = x_pad.size()[1]
    mask = generate_mask(max_len,x_lens)
    return x_pad,y_pad,x_lens,mask

def generate_mask(max_len,real_len):
    """
    生成mask
    """
    batch_size = len(real_len)
    mask = torch.zeros(batch_size,max_len,dtype=torch.uint8)
    for i,length in enumerate(real_len):
        mask[i,:length] = 1
    return mask

def getDataLoader(batch_size,train_path,valid_path,label_path):
    train_dataset = MyDataSet(train_path,label_path,train=1)
    valid_dataset = MyDataSet(valid_path,label_path,train=0)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False,collate_fn = collate_fn)
    return train_dataloader,valid_dataloader
