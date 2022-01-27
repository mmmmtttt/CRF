from gensim.models.keyedvectors import Vocab
import numpy as np
import re

def loadData(path,get_vocab = False):
    sentences = []
    tags = []
    vocab = ['<PAD>','<UNK>']
    with open(path,'r',encoding="UTF-8") as f:
        sentence = []
        tag = []
        for line in f:
            if line=="\n":
                if not sentence:#到文件结尾
                    break
                #一个句子的结尾
                sentences.append(''.join(sentence))
                tags.append(''.join(tag))
                sentence,tag = [],[]
            else:#句子未结束
                pair = line.split()
                sentence.append(pair[0])
                tag.append(pair[1])
                if get_vocab and pair[0] not in vocab: #从数据中获得字典
                    vocab.append(pair[0])
    return np.asarray(sentences),np.asarray(tags),vocab

def loadTemplate(path):
    templates = {'u':[],'b':[]}
    with open(path,'r',encoding="UTF-8") as f:
        pattern = re.compile(r'(?<=\[)-?\d+') #找到[之后的数字(距离当前位置的偏移量)
        for line in f:
            if line[0] == "U": #word和tag的对应关系
                templates['u'].append(list(map(int,pattern.findall(line))))
            elif line[0] == "B": #当前tag和上下文的tag的对应关系
                templates['b'].append(list(map(int,pattern.findall(line))))
    return templates

def loadLabel(path):
    with open(path,'r',encoding="UTF-8") as f:
        labels = f.read().split("\n")
    labels.remove('')
    return np.asarray(labels)

