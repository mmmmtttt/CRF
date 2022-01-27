import tensorflow
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers.core import Lambda
import BiLSTM_CRF.dataLoader as dataLoader
import numpy as np
from gensim.models import KeyedVectors
import tensorflow.keras as keras
from crf import CRF
from tqdm import tqdm

tensorflow.compat.v1.enable_eager_execution() 

WORD2VEC_PATH = "sgns.context.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5.bz2"
TEMPLATEPATH = "train_dataset/dataset2/template.utf8"
LABELPATH = "train_dataset/dataset1/labels.utf8"
DATAPATH = "train_dataset/dataset1/train.utf8"
VALID_PATH= "train_dataset/dataset2/train.utf8"
CHECKPOINTPATH = "bilstm_crf.pkl"
tensorflow.autograph.set_verbosity(0) #忽略警告

def create_embedding_matrix(word2vec_model, vocab,embed_dim=300):
    """
    从word2vec模型中创建embedding层的权重
    """
    # embeddings_matrix = np.zeros((len(vocab), embed_dim)) #初始化词向量矩阵
    # for i in tqdm(range(2,len(vocab))): # 对pad和未知字默认是0
    #     char = vocab[i] #每个字
    #     embeddings_matrix[i] = word2vec_model.get_vector(char)
    # np.save("embeddingmatrix.npy",embeddings_matrix)
    embeddings_matrix = np.load("embeddingmatrix.npy")
    return embeddings_matrix

def word2idx(data,vocab2idx):
    index_result = []
    for line in data:
        index_result.append(list(map(lambda x:vocab2idx[x],line)))
    return index_result

class BiLSTM_CRF:
    def __init__(self) -> None:
        self.max_len = 0 #在读取文件时计算
        self.label_classes = 4

    def prepocess_data(self):
        """
        加载数据，预处理数据
        """
        print("preprocess data...")
        self.labels = dataLoader.loadLabel(LABELPATH)
        self.label_classes = len(self.labels)
        vocab = ['<PAD>','<UNK>'] #特殊词：PAD表示padding，UNK表示词表中没有
        #加载数据和词表
        train_x,train_y =dataLoader.loadData(DATAPATH,vocab)
        test_x,test_y = dataLoader.loadData(VALID_PATH,vocab)
        #计算最大的句子长度
        self.max_len = max(len(max(train_x,key=len)),len(max(max(test_x,key=len)))) #计算最大值
        #把字和标签映射到词表中的序号
        self.vocab2idx = {char: idx for idx, char in enumerate(vocab)}
        train_x = word2idx(train_x, self.vocab2idx)
        test_x = word2idx(test_x, self.vocab2idx)
        self.label2idx = {char: idx for idx, char in enumerate(self.labels)}
        train_y = word2idx(train_y, self.label2idx)
        test_y = word2idx(test_y, self.label2idx)
        #padding到最大的长度
        train_x = keras.preprocessing.sequence.pad_sequences(train_x,self.max_len)
        test_x = keras.preprocessing.sequence.pad_sequences(test_x,self.max_len)
        train_y = keras.preprocessing.sequence.pad_sequences(train_y,self.max_len)
        test_y = keras.preprocessing.sequence.pad_sequences(test_y,self.max_len)
        train_y = np.expand_dims(train_y,2)
        test_y = np.expand_dims(test_y,2)
        print("finish preprocessing data!")
        return train_x,train_y,test_x,test_y,vocab

    def create_model(self,vocab,embed_dim=300,lstm_units=128): 
        print("create model...")  
        # word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=False, unicode_errors='ignore')  
        word2vec_model=""
        embeddings_matrix = create_embedding_matrix(word2vec_model,vocab)
        print("finish load embedding weight...")  
        inputs = keras.layers.Input(shape=(self.max_len,), dtype='int32') 
        x = keras.layers.Embedding(
            len(vocab), embed_dim, 
            input_length=self.max_len, 
            embeddings_initializer=keras.initializers.Constant(embeddings_matrix),
            trainable=False, #固定预训练的词向量
            mask_zero=True #预处理时用0 padding，这部分padding不参与计算
            )(inputs)
        x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units,return_sequences=True))(x)
        self.crf = CRF(self.label_classes, sparse_target=True) #输出不是one-hot向量
        y = self.crf(x)
        self.model = keras.Model(inputs=inputs, outputs=y)
        self.model.summary()
        self.model.run_eagerly= True #调试用

    def train(self,train_x,train_y,test_x,test_y,batch_size=64,epochs=10):
        self.model.compile(optimizer=keras.optimizers.Adam(1e-5),loss=self.crf.loss,metrics=[self.crf.accuracy])
        self.model.fit(train_x, train_y, validation_data=[test_x,test_y], batch_size=batch_size, epochs=epochs)
        self.model.save(CHECKPOINTPATH)

if __name__=='__main__':   
    model = BiLSTM_CRF()
    train_x,train_y,test_x,test_y,vocab = model.prepocess_data()
    model.create_model(vocab)
    model.train(train_x,train_y,test_x,test_y)

