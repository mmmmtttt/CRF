from gensim.matutils import pad
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchcrf import CRF
from MyDataSet import getDataLoader,MyDataSet,generate_mask,word2idx,idx2label
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# 文件
DATA_PATH = "train_dataset/dataset1/train.utf8"
LABEL_PATH = "train_dataset/dataset1/labels.utf8"
VALID_PATH = "train_dataset/dataset2/train.utf8"
CHECKPOINT_PATH = "bilstm_crf.pkl"
EMBEDDING_PATH = "embedding_matrix.pt" #embedding层的权重
WORD2VEC_PATH = "sgns.context.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5.bz2"

#模型参数
EMBED_DIM = 300
HIDDEN_DIM = 128
LABEL_CLASSES = 4
BATCH_SIZE=64

def create_embedding_matrix(word2vec_model:KeyedVectors, vocab, embed_dim=EMBED_DIM):
    """
    从word2vec模型中创建embedding层的权重，保存
    """
    embeddings_matrix = torch.zeros(len(vocab), embed_dim) #初始化词向量矩阵
    for i in range(2,len(vocab)): # 对pad和未知字默认是0
        char = vocab[i] #每个字
        embeddings_matrix[i,:] = torch.from_numpy(word2vec_model.get_vector(char))
    torch.save(embeddings_matrix,EMBEDDING_PATH)
    return embeddings_matrix

class Bi_LSTM_CRF(nn.Module):
    def __init__(self,embedding_weight,fine_tuning=False,embed_dim= EMBED_DIM,hidden_dim=HIDDEN_DIM,label_classes =LABEL_CLASSES):
        super(Bi_LSTM_CRF,self).__init__() #对继承自父类的属性进行初始化
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.label_classes = label_classes

        self.embedding = nn.Embedding.from_pretrained(embedding_weight,padding_idx=0) #预训练的词向量
        self.embedding.weight.requires_grad = fine_tuning #在训练过程中对词向量的权重是否进行微调
        self.dropout = nn.Dropout(0.2)
        self.bi_lstm = nn.LSTM(embed_dim,hidden_dim,num_layers=1,bidirectional=True,batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim *2, label_classes) #得到发射矩阵
        self.crf = CRF(label_classes,batch_first=True)
    
    def forward(self,x,x_lens,mask):
        """评估验证时调用，得到预测的序列"""
        x = self.embedding(x) #x已经被pad
        x = self.dropout(x)
        #压缩padding后的序列，让lstm处理变长序列
        x = pack_padded_sequence(x, x_lens, batch_first=True,enforce_sorted=False) 
        x,_ = self.bi_lstm(x)
        x,_ = pad_packed_sequence(x,batch_first=True)
        x = self.hidden2tag(x)
        x = self.crf.decode(emissions=x,mask=mask)
        return x

    def neg_log_likelihood(self,x,y,x_lens,mask):
        """训练用，负对数似然损失函数"""
        x = self.embedding(x) #x已经被pad
        x = self.dropout(x)
        #压缩padding后的序列，让lstm处理变长序列
        x = pack_padded_sequence(x,x_lens, batch_first=True,enforce_sorted=False) 
        x,_ = self.bi_lstm(x)
        x,_ = pad_packed_sequence(x,batch_first=True)
        x = self.hidden2tag(x)
        return - self.crf(x,y,mask=mask)
    
def train(epochs):
    train_dataloader,valid_dataloader = getDataLoader(BATCH_SIZE,DATA_PATH,VALID_PATH,LABEL_PATH)
    
    print("load pretrained embedding weight")
    if os.path.exists(EMBEDDING_PATH): #存在之前保存好的词向量权重矩阵
        embedding_weight = torch.load(EMBEDDING_PATH)
    else: #从模型中加载权重
        word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=False, unicode_errors='ignore')  
        embedding_weight = create_embedding_matrix(word2vec_model,MyDataSet.vocab,EMBED_DIM)
    
    print("start training")
    model = Bi_LSTM_CRF(embedding_weight,fine_tuning=True)
    optimizer=optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-5) #一阶动量+二阶动量
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7) #等间隔调整
    model.train()
    
    for epoch in range(epochs):
        print("---------------------epoch %s---------------------"%epoch)
        for batchIndex,(data,label,x_lens,mask) in enumerate(train_dataloader):
            data,label = Variable(data),Variable(label)
            # 训练
            optimizer.zero_grad() #上一次的梯度记录被清空
            loss = model.neg_log_likelihood(data,label,x_lens,mask)
            loss.backward() #自动计算梯度
            optimizer.step() #更新参数
            # 验证
            accuracy = valuate(model,valid_dataloader)
            print('epoch:%s,batch:%s,accuracy:%.6f'%(epoch,batchIndex,accuracy))
            # 保存模型
        torch.save(model.state_dict(),CHECKPOINT_PATH)
        scheduler.step()

def valuate(model,data_loader):
    """
    在验证集上测试准确率
    """
    model.eval() #切换evaluation模式
    correct = 0
    total = 0
    with torch.no_grad():#上下文管理器，不追踪梯度
        for data,label,x_lens,mask in data_loader:
            output = model(data,x_lens,mask) #调用forward函数
            for row in range(len(output)):
                same = [1 if ele==label[row,i] else 0 for i,ele in enumerate(output[row])]
                correct = correct+sum(same)
                total =total+len(same)
    model.train()
    return correct/total

def predict(model,sentences):
    """
    在测试集上得到标注结果
    sentences是list(str)
    返回list(str)
    """
    model.eval()
    word2idx_dict = np.load("vocab2idx.npy", allow_pickle=True).item()
    idx2label_list = np.load("idx2label.npy", allow_pickle=True)
    idx_x = word2idx(sentences,word2idx_dict) #字序列转成序号序列
    x_lens = [len(sentence) for sentence in sentences] #真实长度
    x_pad = pad_sequence(idx_x,batch_first=True,padding_value=0) #补成定长
    max_len = x_pad.size()[1]
    mask = generate_mask(max_len,x_lens) #生成mask
    idx_result = model(x_pad,x_lens,mask) #得到的结果是序号，且不定长
    label_result = idx2label(idx_result,idx2label_list)#序号转标签
    return list(map(lambda x:''.join(x),label_result)) #结果从list->str

def load_model():
    print("part2:load model from %s"%CHECKPOINT_PATH)
    embedding_weight = torch.load(EMBEDDING_PATH)
    model = Bi_LSTM_CRF(embedding_weight)
    state_dict = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(state_dict)
    return model

if __name__=="__main__":
    # train(epochs=10)
    pass