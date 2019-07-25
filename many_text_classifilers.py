# ========================读取数据，生成可以直接训练的样本：==================

import pandas as pd
from utils import *

labeled_data = pd.read_excel('data.xlsx')
filled_labeled_data = labeled_data.fillna('其他')

labeled_files = list(filled_labeled_data.filename)
corpus = []
X_words = []
for file_name in labeled_files:
    with open('data/%s.txt'%file_name,encoding='utf-8') as f:
        text = f.read()
        text = remove_punctuations(text)
        text_words = jieba.lcut(text)
        corpus += text_words
        X_words.append(text_words)

vocab_size = 20000
maxlen = 800
tokenizer, word_index, freq_word_index = fit_corpus(corpus,vocab_size=vocab_size)
X = text2dix(tokenizer,X_words,maxlen=maxlen)
Y_dict = {}
for cate in ['sentiment','industry','gov','neirong']:
    Y_dict[cate] = label2idx(list(filled_labeled_data[cate]))

for key in Y_dict:
    print(Y_dict[key][1])


# ===========================搭建模型:=================================

import keras
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,GRU,Embedding,Input,Conv1D,MaxPooling1D
from keras.layers import Flatten,Dropout,Concatenate,concatenate,Lambda
import keras.backend as K
from keras.utils import to_categorical
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# 加载词向量模型：
wv_path = '../wv/wikibaikewv250'
print("Loading word2vec model, may take a few minutes......")
if ('wvmodel' not in vars()): # 避免重复加载  
    wvmodel = Word2Vec.load(wv_path)
wvdim = 250

embedding_matrix = create_embedding_matrix(wvmodel,vocab_size,wvdim,freq_word_index)


# ===========================各种模型！===================================
def LSTM_classify(X_train,X_test,Y_train,Y_test,num_classes):
    m = Sequential()
    m.add(Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix]))
    m.add(LSTM(250))
    m.add(Dense(num_classes,activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     m.summary()
    m.fit(X_train,to_categorical(Y_train),batch_size=32,epochs=10)
    results = m.evaluate(X_test,to_categorical(Y_test))
    del m
    return results

def CNN_classify(X_train,X_test,Y_train,Y_test,num_filters,filter_sizes,num_classes,batch_size=32,epochs=10):
    input = Input(shape=(maxlen,))
    emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix])(input)
    feature_maps = []
    for fs in filter_sizes:
        cnn = Conv1D(num_filters,fs,padding='valid',strides=1,activation='relu')(emb)
        maxpool = MaxPooling1D(pool_size=maxlen-fs+1,strides=1)(cnn) # 800-5+1, 这样最后只得到一个值
        feature_maps.append(maxpool)
    feature_all = Concatenate()(feature_maps)
    flat = Flatten()(feature_all)
    droped_flat = Dropout(0.5)(flat)
    output = Dense(num_classes,activation='softmax')(droped_flat)
    m = Model(inputs=input,outputs=output)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     m.summary()
    m.fit(X_train,to_categorical(Y_train),batch_size=batch_size,epochs=epochs)
    Y_pred = np.argmax(m.predict(X_test),axis=1)
    accuracy = accuracy_score(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred,average='weighted')
    recall = recall_score(Y_test,Y_pred,average='weighted')
    f1 = f1_score(Y_test,Y_pred,average='weighted')
    results = [accuracy,precision,recall,f1]
    del m
    return results

def LEAM_cnn_classify(X_train,X_test,Y_train,Y_test,num_filters,filter_sizes,num_classes,batch_size=32,epochs=10):
    # 计算attention score时需要用到的余弦相似度函数：
    def tf_cosine_similarity(X,Y):
        product = tf.matmul(X,tf.transpose(Y,perm=[0,2,1])) #(C,V) tranpose要指定perm参数，使之batch维度不动
        X_norm = tf.norm(X,axis=2) # 由于这里是三维的，第一位是batch，所以axis应该是最后一维.返回的是二维的
        X_norm = tf.reshape(X_norm,(-1,X_norm.shape[1],1))
        Y_norm = tf.norm(Y,axis=2)
        Y_norm = tf.reshape(Y_norm,(-1,Y_norm.shape[1],1))
        norm = tf.matmul(X_norm,tf.transpose(Y_norm,perm=[0,2,1]))
        cs_matrix = product/norm
        return cs_matrix

    text_input = Input(shape=(maxlen,),name='text_input')
    label_input = Input(shape=(num_classes,),name='label_input')
    text_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
    label_emb = Embedding(num_classes,wvdim,input_length=num_classes,name='label_emb')(label_input) # (C,wvdim)
    att_base = Lambda(lambda pair:tf_cosine_similarity(pair[0],pair[1]),name='att_base')([label_emb,text_emb])
    att_score = MaxPooling1D(pool_size=num_classes,name='att_score')(att_base)
    att_weight = Lambda(lambda x:K.softmax(x),name='att_weight')(att_score)
    att_emb = Lambda(lambda pair:tf.multiply(tf.reshape(pair[0],(-1,pair[0].shape[2],1)),pair[1]),name='att_emb')([att_weight,text_emb])
    
    feature_maps = []
    for fs in filter_sizes:
        cnn = Conv1D(num_filters,fs,padding='valid',strides=1,activation='relu')(att_emb)
        maxpool = MaxPooling1D(pool_size=maxlen-fs+1,strides=1)(cnn) # 800-5+1, 这样最后只得到一个值
        feature_maps.append(maxpool)
    feature_all = Concatenate()(feature_maps)
    flat = Flatten()(feature_all)
    droped_flat = Dropout(0.7)(flat)
    output = Dense(num_classes,activation='softmax')(droped_flat)

    m = Model(inputs=[text_input,label_input],outputs=output)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #输入的y需要构造一下，每一个输入文本都有一个固定的y输入
    #比如有13类的话，那么可以直接构造维度（num_samples,13）
    L_train = np.array([np.array(range(num_classes)) for i in range(len(X_train))])
    L_test = np.array([np.array(range(num_classes)) for i in range(len(X_test))])
    m.fit([X_train,L_train],to_categorical(Y_train),batch_size=batch_size,epochs=epochs)
    
    Y_pred = np.argmax(m.predict([X_test,L_test]),axis=1)
    accuracy = accuracy_score(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred,average='weighted')
    recall = recall_score(Y_test,Y_pred,average='weighted')
    f1 = f1_score(Y_test,Y_pred,average='weighted')
    results = [accuracy,precision,recall,f1]
    del m
    return results

def LEAM_lstm_classify(X_train,X_test,Y_train,Y_test,num_filters,filter_sizes,num_classes,batch_size=32,epochs=10):
    def tf_cosine_similarity(X,Y):
        product = tf.matmul(X,tf.transpose(Y,perm=[0,2,1])) #(C,V) tranpose要指定perm参数，使之batch维度不动
        X_norm = tf.norm(X,axis=2) # 由于这里是三维的，第一位是batch，所以axis应该是最后一维.返回的是二维的
        X_norm = tf.reshape(X_norm,(-1,X_norm.shape[1],1))
        Y_norm = tf.norm(Y,axis=2)
        Y_norm = tf.reshape(Y_norm,(-1,Y_norm.shape[1],1))
        norm = tf.matmul(X_norm,tf.transpose(Y_norm,perm=[0,2,1]))
        cs_matrix = product/norm
        return cs_matrix

    text_input = Input(shape=(maxlen,),name='text_input')
    label_input = Input(shape=(num_classes,),name='label_input')
    text_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
    label_emb = Embedding(num_classes,wvdim,input_length=num_classes,name='label_emb')(label_input) # (C,wvdim)
    att_base = Lambda(lambda pair:tf_cosine_similarity(pair[0],pair[1]),name='att_base')([label_emb,text_emb])
    att_score = MaxPooling1D(pool_size=num_classes,name='att_score')(att_base)
    att_weight = Lambda(lambda x:K.softmax(x),name='att_weight')(att_score)
    att_emb = Lambda(lambda pair:tf.multiply(tf.reshape(pair[0],(-1,pair[0].shape[2],1)),pair[1]),name='att_emb')([att_weight,text_emb])
    
    lstm = LSTM(250)(att_emb)
    output = Dense(num_classes,activation='softmax')(lstm)

    m = Model(inputs=[text_input,label_input],outputs=output)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #输入的y需要构造一下，每一个输入文本都有一个固定的y输入
    #比如有13类的话，那么可以直接构造维度（num_samples,13）
    L_train = np.array([np.array(range(num_classes)) for i in range(len(X_train))])
    L_test = np.array([np.array(range(num_classes)) for i in range(len(X_test))])
    m.fit([X_train,L_train],to_categorical(Y_train),batch_size=batch_size,epochs=epochs)
    
    Y_pred = np.argmax(m.predict([X_test,L_test]),axis=1)
    accuracy = accuracy_score(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred,average='weighted')
    recall = recall_score(Y_test,Y_pred,average='weighted')
    f1 = f1_score(Y_test,Y_pred,average='weighted')
    results = [accuracy,precision,recall,f1]
    del m
    return results
