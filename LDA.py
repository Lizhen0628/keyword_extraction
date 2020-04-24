# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 6:58 下午
# @Author  : lizhen
# @FileName: LDA.py
# @Description:
import gensim
import math
import jieba
import jieba.posseg as posseg
from jieba import analyse
from gensim import corpora, models
import functools
import numpy as np
import os


# 停用词表加载方法
# 停用词表存储路径，每一行为一个词，按行读取进行加载
# 进行编码转换确保匹配准确率
def get_stopword_list():
    stopword_path = 'stop_words.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stopword_path, encoding='utf8').readlines()]
    return stopword_list


# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        # 不进行词性标注分词
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注分词
        seg_list = posseg.cut(sentence)

    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    # 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if word not in stopword_list and len(word) > 1:
            filter_list.append(word)
    return filter_list


# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos=False, data_dir='data'):
    # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    documents = os.listdir(data_dir)
    doc_list = []
    for document in documents:

        for line in open(os.path.join(data_dir,document), 'r', encoding='utf8'):
            content = line.strip()
            seg_list = seg_to_list(content, pos)
            filter_list = word_filter(seg_list, pos)
            doc_list.append(filter_list)

    return doc_list






# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LDA', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化 (token_id,freq)
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]  # (token_id,freq)
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.tfidf_corpus = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载胡模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    # 向量化
    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        print("vec_list", vec_list)
        return vec_list

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        # 2及变1及结构
        for doc in doc_list:
            # extend he append 方法有何异同 容易出错
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    # 得到数据集的主题 - 词分布
    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            singlist = [word]
            # 计算每个词胡加权向量
            word_corpus = self.tfidf_model[self.dictionary.doc2bow(singlist)]
            # 计算每个词de主题向量
            word_topic = self.model[word_corpus]
            wordtopic_dic[word] = word_topic

        return wordtopic_dic

    def train_lsi(self):
        lsi = models.LsiModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        # 文档的加权向量
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        # 文档主题 向量
        senttopic = self.model[sentcorpus]

        # senttopic [(0, 0.03457821), (1, 0.034260772), (2, 0.8970413), (3, 0.034119748)]
        # 余弦相似度计算

        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            # 还是计算每个再本文档中的词  和文档的相识度
            if k not in word_list:
                continue
            #
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=lambda item:item[1], reverse=True)[:self.keyword_num]:
            print(k, v)
        print()




#
def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/ ", end='')
    print()


# test
if __name__ == '__main__':
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'

    pos = False
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print('LDA模型结果：')
    topic_extract(filter_list, 'LDA', pos)


# LDA模型结果：
# doc_list>>>>>>>>>>> [['南都讯', '记者', '刘凡', '周昌', '日票', '深圳', '地铁', '车厢', '...]..]
# 先构建词空间self.dictionary  Dictionary(4064 unique tokens: ['上将', '专门', '乘客', '仪式', '体验']...)
# 使用BOW模型向量化corpus [[(0, 1), (1, 1), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), ...
# dictionary变化前 ['南都讯', '记者', '刘凡', '周昌', '日票', '深圳', '地铁', '车厢', '坐票',..
# dictionary变化后 ['四联', '注音版', '前瞻性', '饲料', '成正比', '省城', '正值', '养白',
# self.wordtopic_dic 得到数据集的主题-词分布 {'四联': [(0, 0.12684152), (1, 0.12683797), (2, 0.12792856),
# (3, 0.61839193)], '注音版': [(0, 0.61396044), (1, 0.12880294), (2, 0.12856448), ....
# 晋江市/ 年度/ 频道/ 民政部/ 大会/ 陈健倩/ 许嘉璐/ 重庆市/ 人大常委会/ 巡视员/

# LDA 和LSI LSA步骤总结
# 数据集处理
# 1先构建词空间  Dictionary(4064 unique tokens: ['上将', '专门', '乘客', '仪式', '体验']...)
# 2使用BOW模型向量化   corpus [[(0, 1), (1, 1), (2, 2), (3, 1),。。。
# 3对每个词，根据tf-idf进行加权，得到加权后的向量表示
#
# 根据数据集获得模型
# 4得到数据集的主题-词分布  model (得到每个词的向量）（文档转列表 再转集合去重，再转列表）{'白血病': [(0, 0.1273009), (1, 0.6181468), (2, 0.12732704), (3, 0.12722531)], '婴儿': [。。。
# 5求文档的分布:词》向量》tf/idf加权》同第4步得到文档的分布向量 [(0, 0.033984687), (1, 0.033736005), (2, 0.8978361), (3, 0.03444325)]
# 6.计算余弦距离得到结果




