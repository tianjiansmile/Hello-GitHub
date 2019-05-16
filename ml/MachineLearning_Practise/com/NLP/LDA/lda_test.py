from gensim import corpora, models, similarities
import gensim
import jieba
import numpy as np
import pandas as pd
import re
from com.NLP.jieba import jieba_test
import pymongo
# import nltk
# from nltk import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download()
import string
from com.risk_score.feature_extact import setting

client = pymongo.MongoClient(setting.host)
db = client.call_test
coll = db.call_info

doc1 = "吉林省长春市万通高层A14栋610室"
doc2 = "广东省广州市海珠区凤源里22号"
doc3 = "安徽省宣城市安徽省,宣城市,宣州区|美都新城25栋903"
doc4 = "福建省龙岩市长汀县河田镇下修坊村水古岭路8号"
doc5 = "广西壮族自治区柳州市柳北区北雀路十七区26 栋1单元201号"

# doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
# doc2 = "My father spends a lot of time driving my sister around to dance practice."
# doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
# doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
# doc5 = "Health experts say that Sugar is not good for your lifestyle."

# 整合文档数据
doc_complete = [doc1, doc2, doc3, doc4, doc5]

def mongo_read():
    results = coll.find({}, {'_id': 0, 'emergencer': 0,'calls': 0,'contacts': 0,'phone': 0,'carr_phone': 0 }).limit(10)
    for d in results:
        if d:
            idnum = d.get('id_num')
            addresses = d.get('addresses')

            l_a = addresses.get('L')
            if l_a:
                print(l_a)
                texts = jieba_test.jieba_split(l_a[0])
                dictionary = corpora.Dictionary(texts)
                corpus = [dictionary.doc2bow(text) for text in texts]
                print(corpus)

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(text, cut_all=False)
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)
    print(seg_list)

    return seg_list
if __name__ == '__main__':
    # mongo_read()

    stop = set(['is','not'])
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    doc_clean = [jieba_split(doc) for doc in doc_complete]

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary(doc_clean)

    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # 使用 gensim 来创建 LDA 模型对象
    Lda = models.ldamodel.LdaModel

    # 在 DT 矩阵上运行和训练 LDA 模型
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)

    print(ldamodel.print_topics(num_topics=3, num_words=3))