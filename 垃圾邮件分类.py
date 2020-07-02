# import csv


# def read_file():
#     file_path = r'D://PycharmProjects//naive_bayes//data//SMSSpamCollection'
#     sms = open(file_path, encoding='utf-8')
#     csv_reader = csv.reader(sms, delimiter='\t')
#     for r in csv_reader:
#         print(r)
#     sms.close()
#
#
# if __name__ == '__main__':
#     read_file()

'''
查看版本
'''
# import nltk
#
# print(nltk.__doc__)

'''
传统方法实现
'''
# # 利用列表、字典、集合等操作进行词频统计
# sep = '.,:;?!-_'
# exclude = {'a', 'the', 'and', 'i', 'you', 'in'}
#
#
# def gettxt():
#     txt = open(r'D://PycharmProjects//naive_bayes//data//test.txt', 'r').read().lower()  # 大小写
#     for ch in sep:
#         txt = txt.replace(ch, '')  # 标点符号
#     return txt
#
#
# bigstr = gettxt()  # 获取待统计字符串
# biglist = bigstr.split()  # 英文分词列表
# bigdict = {}
# for word in biglist:
#     bigdict[word] = bigdict.get(word, 0) + 1  # 词频统计字典
# for word in exclude:
#     del(bigdict[word])  # 无意义词
# bigitems = list(bigdict.items())
# bigitems.sort(key=lambda x: x[1], reverse=True)   # 按词频排序
# for i in range(10):
#     w, c = bigitems[i]
#     print('{0:>10}：{1:<5}'.format(w, c))  # TOP10

# import nltk
#
# text = "I've been searching for the right words to thank you for this breather. I promise i wont take your help for " \
#         "granted and will fulfil my promise. You have been wonderful and a blessing at all times."
#
# sents = nltk.sent_tokenize(text)
# sents
#
# print(nltk.word_tokenize(sents[0]))
# print(text.split())


'''
去掉停用词
'''
# from nltk.corpus import stopwords
# import nltk
#
# stops = stopwords.words('english')
# print("英文停用词：\n", stops)
# print("英文停用词数量：", len(stops))


# text = "I've been searching for the right words to thank you for this breather. I promise i wont take your help for " \
#         "granted and will fulfil my promise. You have been wonderful and a blessing at all times."
'''
方法一
'''
# tokens = []
# for sent in nltk.sent_tokenize(text):
#     for word in nltk.word_tokenize(sent):
#         tokens.append(word)
# print(tokens)
'''
方法二
'''
# tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
# print("分词后的句子：\n", tokens)
# print("总共有", len(tokens), "个单词")

# tokens = [token for token in tokens if token not in stops]
# print("去除停用词后的句子：\n", tokens)
# print("总共有", len(tokens), "个单词")

'''
词性标注
'''
# print(nltk.pos_tag(tokens))
'''
词性转换
'''
# from nltk.stem import WordNetLemmatizer
#
# lemmatizer = WordNetLemmatizer()
# lemmatizer.lemmatize('leaves')
# lemmatizer.lemmatize('better')
# lemmatizer.lemmatize('better', pos='a')
# lemmatizer.lemmatize('made', pos='v')


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import nltk
import csv
import numpy as np


def get_wordnet_pos(treebank_tag):  # 根据词性，生成还原参数pos
    """
    根据词性，生成还原参数 pos
    """
    if treebank_tag.startswith('J'):  # 形容词
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):  # 动词
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):  # 名词
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):  # 副词
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN


def preprocessing(text):
    """
    预处理
    """
    # text = text.decode("utf-8")
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]  # 分词
    stops = stopwords.words('english')  # 使用英文的停用词表
    tokens = [token for token in tokens if token not in stops]  # 去除停用词

    tokens = [token.lower() for token in tokens if len(token) >= 3]  # 大小写，短词
    lmtzr = WordNetLemmatizer()
    tag = nltk.pos_tag(tokens)  # 词性
    tokens = [lmtzr.lemmatize(token, pos=get_wordnet_pos(tag[i][1])) for i, token in enumerate(tokens)]  # 词性还原
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def create_dataset():
    """
    导入数据
    """
    file_path = r'D://PycharmProjects//naive_bayes//data//SMSSpamCollection'
    sms = open(file_path, encoding='utf-8')
    sms_data = []
    sms_label = []
    csv_reader = csv.reader(sms, delimiter='\t')
    for line in csv_reader:
        sms_label.append(line[0])  # 提取出标签
        sms_data.append(preprocessing(line[1]))  # 提取出特征
    sms.close()
    # print("数据集标签：\n", sms_label)
    # print("数据集特征：\n", sms_data)
    return sms_data, sms_label


def revert_mail(x_train, X_train, model):
    """
    向量还原成邮件
    """
    s = X_train.toarray()[0]
    print("=====================================================")
    print("第一封邮件向量表示为：", s)
    # 该函数输入一个矩阵，返回扁平化后矩阵中非零元素的位置（index）
    a = np.flatnonzero(X_train.toarray()[0])  # 非零元素的位置（index）
    print("向量的非零元素的值：", s[a])
    b = model.vocabulary_  # 词汇表
    key_list = []
    for key, value in b.items():
        if value in a:
            key_list.append(key)  # key非0元素对应的单词
    print("向量非零元素对应的单词：", key_list)
    print("向量化之前的邮件：", x_train[0])


def split_dataset(data, label):
    """
    划分数据集
    """
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)
    tfidf2 = TfidfVectorizer()
    X_train = tfidf2.fit_transform(x_train)  # X_train用fit_transform生成词汇表
    X_test = tfidf2.transform(x_test)  # X_test要与X_train词汇表相同，因此在X_train进行fit_transform基础上进行transform操作
    revert_mail(x_train, X_train, tfidf2)

    return X_train, X_test, y_train, y_test


def mnb_model(x_train, x_test, y_train):
    """
    模型构建（根据数据特点选择多项式分布）
    """
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_mnb = mnb.predict(x_test)
    return y_mnb


def class_report(y_mnb, y_test):
    """
    模型评价：混淆矩阵
    """
    conf_matrix = confusion_matrix(y_test, y_mnb)
    print("=====================================================")
    print("混淆矩阵：\n", conf_matrix)
    cr = classification_report(y_test, y_mnb)
    print("=====================================================")
    print("分类报告：\n", cr)
    print("模型准确率：", (conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix))


if __name__ == '__main__':
    sms_data, sms_label = create_dataset()
    X_train, X_test, y_train, y_test = split_dataset(sms_data, sms_label)
    y_mnb = mnb_model(X_train, X_test, y_train)
    class_report(y_mnb, y_test)



"""
文本特征提取：
把文本数据转化成特征向量的过程，比较常用的文本特征表示法为词袋法
词集：0、1
词袋模型：
不考虑词语出现的顺序怕，每个出现过的词汇单独作为一列特征，这些不重复的特征词汇集合为词表[room desk]10000
每一个文本都可以在很长的词表上统计出一个很多列的特征向量[2, 0, 0, 0, 0, 0, 0, 0]10000
如果每个文本都出现的词汇，一般被标记为停用词不计入特征向量

主要有两个api来实现CountVectorizer和TfidfVectorizer
CountVectorizer：只考虑词汇在我本本中出现的频率
TfidfVectorizer：（1）除了考量某词汇在本文本中出现的频率，还关注包含这个词的其他文本的数量30 the 5000
                 （2）能够削减高频没有意义的词汇出现带来的影响，挖掘更有意义的特征
"""
"""
TF-IDF 概念
是一种统计方法，用以评估一个词对于一个语料库中一份文件的重要程度。
词的重要性随着在文件中出现的次数正比增加，同时随着它在语料库其他文件中出现的频率反比下降。
就是说一个词在某一个文档中出现次数比较多，其他文档没有出现，说明该词对该份文档分类很重要。然而如果其他文档也出现比较多，说明该词区分性不大，就用IDF来降低该词的权重。
"""
