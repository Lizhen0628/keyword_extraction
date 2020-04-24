# -*- coding: utf-8 -*-
# @Time    : 2020/4/21 6:37 下午
# @Author  : lizhen
# @FileName: tf-idf.py
# @Description:

import os
import math
import jieba
from collections import Counter

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


def get_documents(filepath):  # 遍历文件夹中的所有文件，返回文件list
    arr = []
    for root, dirs, files in os.walk(filepath):
        for fn in files:
            arr.append(os.path.join(root, fn))
    return arr


def get_stopwords(path):  # 获取停用词表
    stop_words = []
    with open(path) as f:
        for line in f.readlines():
            stop_words.append(line.strip())
    return stop_words


def get_corpus(filelist, stop_words_list):  # 建立语料库
    """
    对filelist 中的每篇文档进行分词，并去停用词，然后把分词后的文档添加到语料库列表中
    :param filelist:
    :param stop_words_list:
    :return:
    """
    document_list = []  # 分词后的文档列表
    for document_path in filelist:
        with open(str(document_path)) as f:
            document = ''
            for line in f.readlines():
                document += line.strip()
            # 分词并去停用词
            document_words = [word for word in jieba.cut(document) if word not in stop_words_list]
            document_list.append(document_words)
    return document_list


def wordinfilecount(word, corpus):  # 统计包含该词的文档数
    count = 0  # 计数器
    for document in corpus:
        # 只要文档出现该词，这计数器加1
        if word in document:
            count = count + 1
        else:
            continue
    return count


def tf_idf(document, corpus):  # 计算TF-IDF,并返回字典
    """

    :param document: 计算document里面每个词的tfidf值，document为文本分词后的形式，
    如:[6 月 19 日 2012 年度 中国 爱心 城市 公益活动 新闻 发布会 在京举行]
    如果是对一篇文档进行关键词提取，则需要对文档进行分句，把每句话看成一个document，corpus则存放的是整篇文档分词后的所有句子（句子为分词后的结果）。

    :param corpus:  corpus为所有问当分词后的列表：[document1,document2,document3,...]
    :return:dict类型，按照tfidf值从大到小排序： orderdict[word] = tfidf_value
    """
    word_tfidf = {}
    # 计算词频
    freq_words = Counter(document)
    for word in freq_words:
        # 计算TF：某个词在文章中出现的次数/文章总词数
        tf = freq_words[word] / len(document)

        # 计算IDF：log(语料库的文档总数/(包含该词的文档数+1))
        idf = math.log(len(corpus) / (wordinfilecount(word, corpus) + 1))

        # 计算每个词的TFIDF值
        tfidf = tf * idf  # 计算TF-IDF
        word_tfidf[word] = tfidf

    orderdic = sorted(word_tfidf.items(), key=lambda item: item[1], reverse=True)  # 给字典排序
    return orderdic


def main():
    stop_words_path = r'stop_words.txt'  # 停用词表路径
    stop_words = get_stopwords(stop_words_path)  # 获取停用词表列表

    documents_dir = 'data'
    filelist = get_documents(documents_dir)  # 获取文件列表

    corpus = get_corpus(filelist, stop_words)  # 建立语料库

    for idx,document in enumerate(corpus):
        word_tfidf = tf_idf(document, corpus)  # 计算TF-IDF
        # 输出前十关键词
        print('document {},top 10 key words{}'.format(idx+1,word_tfidf[:10]))
    print('---------------- scikit learn ------------------------')
    # 对语料进行稍微处理
    corpus = [*map(lambda x:" ".join(x), corpus)]
    tfidf_model = TfidfVectorizer()
    tfidf_matrix = tfidf_model.fit_transform(corpus)  # 计算每个词的tfidf值
    words = tfidf_model.get_feature_names()# 所有词的集合
    for i in range(len(corpus)):
        word_tfidf = {}
        for j in range(len(words)):
            word_tfidf[words[j]] = tfidf_matrix[i, j]
        word_tfidf = sorted(word_tfidf.items(),key=lambda item:item[1],reverse=True)
        print('document {},top 10 key words{}'.format(i+1, word_tfidf[:10]))



if __name__ == '__main__':
    main()