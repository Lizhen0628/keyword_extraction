# -*- coding: utf-8 -*-
# @Time    : 2020/4/22 1:38 下午
# @Author  : lizhen
# @FileName: textrank.py
# @Description:

from collections import defaultdict
import jieba.posseg as pseg
import sys

class textrank_graph:
    def __init__(self):
        self.graph = defaultdict(list) # key:[(),(),(),...] 如：是 [('是', '全国', 1), ('是', '调查', 1), ('是', '失业率', 1), ('是', '城镇', 1)]
        self.d = 0.85  # d是阻尼系数，一般设置为0.85
        self.min_diff = 1e-5  # 设定收敛阈值

    # 添加节点之间的边
    def addEdge(self, start, end, weight):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    # 节点排序
    def rank(self):
        # 一共有14个节点
        print(len(self.graph))
        # 默认初始化权重
        weight_deault = 1.0 / (len(self.graph) or 1.0)
        # nodeweight_dict, 存储节点的权重
        nodeweight_dict = defaultdict(float)
        # outsum，存储节点的出度权重
        outsum_node_dict = defaultdict(float)
        # 根据图中的边，更新节点权重
        for node, out_edge in self.graph.items():
            # 是 [('是', '全国', 1), ('是', '调查', 1), ('是', '失业率', 1), ('是', '城镇', 1)]
            nodeweight_dict[node] = weight_deault # 初始化节点权重
            outsum_node_dict[node] = sum((edge[2] for edge in out_edge), 0.0) # 统计node节点的出度
        # 初始状态下的textrank重要性权重
        sorted_keys = sorted(self.graph.keys())
        # 设定迭代次数，
        step_dict = [0]
        for step in range(1, 1000):
            for node in sorted_keys:
                s = 0
                # 计算公式：(edge_weight/outsum_node_dict[edge_node])*node_weight[edge_node]
                for e in self.graph[node]:
                    s += e[2] / outsum_node_dict[e[1]] * nodeweight_dict[e[1]]
                # 计算公式：(1-d) + d*s
                nodeweight_dict[node] = (1 - self.d) + self.d * s
            step_dict.append(sum(nodeweight_dict.values()))

            if abs(step_dict[step] - step_dict[step - 1]) <= self.min_diff:
                break

        # 利用Z-score进行权重归一化，也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]之间。
        # 先设定最大值与最小值均为系统存储的最大值和最小值
        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])
        for w in nodeweight_dict.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in nodeweight_dict.items(): # 归一化
            nodeweight_dict[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return nodeweight_dict



class TextRank:
    def __init__(self):
        self.candi_pos = ['n', 'v', 'a'] # 关键词的词性：名词，动词，形容词
        self.span = 5 # 窗口大小

    def extract_keywords(self, text, num_keywords):
        g = textrank_graph()
        cm = defaultdict(int)
        word_list = [[word.word, word.flag] for word in pseg.cut(text)] # 使用jieba分词并且对词性进行标注
        for i, word in enumerate(word_list): # 该循环用于统计在窗口范围内，词的共现次数
            if word[1][0] in self.candi_pos and len(word[0]) > 1: #
                for j in range(i + 1, i + self.span):
                    if j >= len(word_list):# 防止下标越界
                        break
                    if word_list[j][1][0] not in self.candi_pos or len(word_list[j][0]) < 2: # 排除词性不在关键词词性列表中的词或者词长度小于2的词
                        continue
                    pair = tuple((word[0], word_list[j][0]))
                    cm[(pair)] += 1

        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        nodes_rank = sorted(nodes_rank.items(), key=lambda asd:asd[1], reverse=True)

        return nodes_rank[:num_keywords]

def test():
    text = '''（原标题：央视独家采访：陕西榆林产妇坠楼事件在场人员还原事情经过）
    央视新闻客户端11月24日消息，2017年8月31日晚，在陕西省榆林市第一医院绥德院区，产妇马茸茸在待产时，从医院五楼坠亡。事发后，医院方面表示，由于家属多次拒绝剖宫产，最终导致产妇难忍疼痛跳楼。但是产妇家属却声称，曾向医生多次提出剖宫产被拒绝。
    事情经过究竟如何，曾引起舆论纷纷，而随着时间的推移，更多的反思也留给了我们，只有解决了这起事件中暴露出的一些问题，比如患者的医疗选择权，人们对剖宫产和顺产的认识问题等，这样的悲剧才不会再次发生。央视记者找到了等待产妇的家属，主治医生，病区主任，以及当时的两位助产师，一位实习医生，希望通过他们的讲述，更准确地还原事情经过。
    产妇待产时坠亡，事件有何疑点。公安机关经过调查，排除他杀可能，初步认定马茸茸为跳楼自杀身亡。马茸茸为何会在医院待产期间跳楼身亡，这让所有人的目光都聚焦到了榆林第一医院，这家在当地人心目中数一数二的大医院。
    就这起事件来说，如何保障患者和家属的知情权，如何让患者和医生能够多一份实质化的沟通？这就需要与之相关的法律法规更加的细化、人性化并且充满温度。用这种温度来消除孕妇对未知的恐惧，来保障医患双方的权益，迎接新生儿平安健康地来到这个世界。'''
    textranker = TextRank()
    keywords = textranker.extract_keywords(text, 10)

    for keyword in keywords:
        print(keyword)
    '''
    ('产妇', 1.0)
    ('医院', 0.5913681024247537)
    ('家属', 0.5429117450097523)
    ('事件', 0.5252165334872677)
    ('剖宫产', 0.4323518137698726)
    ('患者', 0.42213201850447274)
    ('榆林', 0.3458613813882902)
    ('温度', 0.3433894045919456)
    ('跳楼', 0.3253241303426245)
    ('事情', 0.30329273312129706)
    '''
test()
