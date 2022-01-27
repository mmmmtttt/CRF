import os
from numpy.core.defchararray import index
import BiLSTM_CRF.dataLoader as dataLoader
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
import bert
from bert.tokenization.bert_tokenization import FullTokenizer
from crf import CRF

BERT_DIR = "./chinese_L-12_H-768_A-12"
MAX_LEN = 420
TEMPLATEPATH = "train_dataset/dataset2/template.utf8"
LABELPATH = "train_dataset/dataset1/labels.utf8"
DATAPATH = "train_dataset/dataset1/train.utf8"
VALID_PATH= "train_dataset/dataset2/train.utf8"
CHECKPOINTPATH = "bilstm_crf.pkl"

def bert_preprocess(data,tokenizer:FullTokenizer):
    """
    给data列表中的每句话增加[CLS][SEP]，转换成字典中的序号
    """
    result = []
    for sentence in data:
        tokens = tokenizer.tokenize(sentence) #分词
        tokens = ["[CLS]"]+tokens+["[SEP]"] #增加首尾标记
        token_ids = tokenizer.convert_tokens_to_ids(tokens) #转换成字典文件中的index
        result.append(token_ids)
    result = keras.preprocessing.sequence.pad_sequences(result)
    return result

def preprocess_label(labels):
    """
    将字母的label转换成序号的label,并加上padding
    """
    label_index = dataLoader.loadLabel(LABELPATH)
    label_dict = dict(zip(label_index,range(1,len(label_index)+1)))
    index_result = []
    for label in labels:
        index_result.append(np.array([0]+list(map(lambda x:label_dict[x],label))+[0]))
    index_result = keras.preprocessing.sequence.pad_sequences(index_result)
    index_result = np.expand_dims(index_result,2)
    return index_result

def create_model():    
    bert_params = bert.params_from_pretrained_ckpt(BERT_DIR)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
    input_ids = keras.layers.Input(shape=(MAX_LEN, ), dtype='int32')
    embedding = l_bert(input_ids)
    x = keras.layers.Masking(mask_value=0)(embedding)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True))(x)
    crf = CRF(4, sparse_target=True)
    outputs = crf(x)
    model = keras.Model(inputs=input_ids, outputs=outputs)
    model.summary()
    model.build(input_shape=(None, MAX_LEN))
    bert_ckpt_file = os.path.join(BERT_DIR, "bert_model.ckpt")
    bert.load_stock_weights(l_bert, bert_ckpt_file)
    model.compile(optimizer=keras.optimizers.Adam(1e-5),loss=crf.loss,metrics=[crf.accuracy])
    return model

if __name__=='__main__':   
    train_sentence,train_label = dataLoader.loadData(DATAPATH)
    train_label = preprocess_label(train_label)
    tokenizer = FullTokenizer(
        vocab_file=os.path.join(BERT_DIR, "vocab.txt")
    )
    processed_train = bert_preprocess(train_sentence,tokenizer)
    model = create_model()
    print(processed_train.shape)
    model.fit(processed_train, train_label, verbose=3,epochs=3)#validation_data=[test_data, test_label]


