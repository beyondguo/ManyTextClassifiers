# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:54:21 2019
@author: GBY
"""

from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import jieba
import re
import time
import copy

# ===================preprocessing:==============================
def remove_punctuations(text):
    return re.sub('[，。：；’‘“”？！、,.!?\'\"\n\t]','',text)


def fit_corpus(corpus,vocab_size=None):
    """
    corpus 为分好词的语料库
    """
    print("Start fitting the corpus......")
    t = Tokenizer(vocab_size) # 要使得文本向量化时省略掉低频词，就要设置这个参数
    tik = time.time()
    t.fit_on_texts(corpus) # 在所有的评论数据集上训练，得到统计信息
    tok = time.time()
    word_index = t.word_index # 不受vocab_size的影响
    print('all_vocab_size',len(word_index))
    print("Fitting time: ",(tok-tik),'s')
    freq_word_index = {}
    if vocab_size is not None:
        print("Creating freq-word_index...")
        x = list(t.word_counts.items())
        s = sorted(x,key=lambda p:p[1],reverse=True)
        freq_word_index = copy.deepcopy(word_index) # 防止原来的字典也被改变了
        for item in s[vocab_size:]:
            freq_word_index.pop(item[0])
        print("Finished!")
    return t,word_index,freq_word_index


def text2dix(tokenizer,text,maxlen):
    """
    text 是一个列表，每个元素为一个文档的分词
    """
    print("Start vectorizing the sentences.......")
    X = tokenizer.texts_to_sequences(text) # 受vocab_size的影响
    print("Start padding......")
    pad_X = pad_sequences(X,maxlen=maxlen,padding='post')
    print("Finished!")
    return pad_X


def create_embedding_matrix(wvmodel,vocab_size,emb_dim,word_index):
    """
    vocab_size 为词汇表大小，一般为词向量的词汇量
    emb_dim 为词向量维度
    word_index 为词和其index对应的查询词典
    """
    embedding_matrix = np.random.uniform(size=(vocab_size+1,emb_dim)) # +1是要留一个给index=0
    print("Transfering to the embedding matrix......")
    # sorted_small_index = sorted(list(small_word_index.items()),key=lambda x:x[1])
    for word,index in word_index.items():
        try:
            word_vector = wvmodel[word]
            embedding_matrix[index] = word_vector
        except Exception as e:
            print(e,"Use random embedding instead.")
    print("Finished!")
    print("Embedding matrix shape:\n",embedding_matrix.shape)
    return embedding_matrix


def label2idx(label_list):
    label_dict = {}
    unique_labels = list(set(label_list))
    for i,each in enumerate(unique_labels):
        label_dict[each] = i
    new_label_list = []
    for label in label_list:
        new_label_list.append(label_dict[label])
    return new_label_list,label_dict

