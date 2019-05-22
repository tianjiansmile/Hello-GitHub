#  coding: utf-8
from gensim import corpora, models, similarities
from nltk.stem.wordnet import WordNetLemmatizer
import jieba
import json

history_dict = {}
# 历史表现
def order_dict():
    file = 'D:/特征提取/历史特征数据/value_list_0.txt'
    file1 = 'D:/特征提取/历史特征数据/value_list_1.txt'
    file2 = 'D:/特征提取/历史特征数据/value_list_2.txt'
    file3 = 'D:/特征提取/历史特征数据/value_list_3.txt'

    file4 = 'D:/特征提取/历史特征数据/value_list_4.txt'
    with open(file ,'r') as rf:
        for line in rf.readlines():
            line = eval(line)
            idnum = line[0]
            apply_sum_all = line[3]
            approve_sum_all = line[6]
            overdue_sum_all = line[9]
            loanamount_sum_all = line[12]
            maxOverdue_sum_all = line[15]
            # print(line)
            # print(idnum,apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all)
            temp_list = [apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all]
            # print(temp_list)
            history_dict[idnum] = temp_list

    with open(file1 ,'r') as rf:
        for line in rf.readlines():
            line = eval(line)
            idnum = line[0]
            apply_sum_all = line[3]
            approve_sum_all = line[6]
            overdue_sum_all = line[9]
            loanamount_sum_all = line[12]
            maxOverdue_sum_all = line[15]
            # print(line)
            # print(idnum,apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all)
            temp_list = [apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all]
            # print(temp_list)
            history_dict[idnum] = temp_list

    with open(file2 ,'r') as rf:
        for line in rf.readlines():
            line = eval(line)
            idnum = line[0]
            apply_sum_all = line[3]
            approve_sum_all = line[6]
            overdue_sum_all = line[9]
            loanamount_sum_all = line[12]
            maxOverdue_sum_all = line[15]
            # print(line)
            # print(idnum,apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all)
            temp_list = [apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all]
            # print(temp_list)
            history_dict[idnum] = temp_list
    #
    with open(file3 ,'r') as rf:
        for line in rf.readlines():
            line = eval(line)
            idnum = line[0]
            apply_sum_all = line[3]
            approve_sum_all = line[6]
            overdue_sum_all = line[9]
            loanamount_sum_all = line[12]
            maxOverdue_sum_all = line[15]
            # print(line)
            # print(idnum,apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all)
            temp_list = [apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all]
            # print(temp_list)
            history_dict[idnum] = temp_list
    #
    with open(file4 ,'r') as rf:
        for line in rf.readlines():
            line = eval(line)
            idnum = line[0]
            apply_sum_all = line[3]
            approve_sum_all = line[6]
            overdue_sum_all = line[9]
            loanamount_sum_all = line[12]
            maxOverdue_sum_all = line[15]
            # print(line)
            # print(idnum,apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all)
            temp_list = [apply_sum_all,approve_sum_all,overdue_sum_all,loanamount_sum_all,maxOverdue_sum_all]
            # print(temp_list)
            history_dict[idnum] = temp_list

def app_keyword_analysis():
    app_dict = {}
    file = 'D:/特征提取/app列表/appData_list_0.txt'
    with open(file, 'r', encoding='UTF-8') as rf:
        count = 0
        lines = rf.readlines()
        for line in lines:
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.split(',')
            for a in apps:
                count += 1
                # print(a)
                check = app_dict.get(a)
                if check is None:
                    app_dict[a] = 1
                else:
                    app_dict[a] += 1

    print(count, len(app_dict))

    black_list = ['一键新机', 'NZT', '饿鱼一键新机', '007改机',
                  '361一键新机', 'V8一键新机', 'AWZ爱伪装', '易天行', '机器猫', 'ROG一键新机'
                                                               '安卓版NZT', '暗王者AWZ', '芝麻变机宝',
                  '大牛模拟器', '神乐科技', '瞬移科技app', '任我行', '天下游', '天下任我行',
                  '神行者', '云峰科技', '芝麻软件', 'IP修改器', 'IP Tools', '天马助贷', '金招网络科技']
    # print(app_dict)
    for app in black_list:
        temp = app_dict.get(app)
        print(app, temp)


app_dict = {}
# 将历史表现映射到app列表
def app_history_mapping():
    file = 'D:/特征提取/app列表/appData_list_0.txt'
    file1 = 'D:/特征提取/app列表/appData_list_1.txt'
    file2 = 'D:/特征提取/app列表/appData_list_2.txt'
    file3 = 'D:/特征提取/app列表/appData_list_3.txt'
    file4 = 'D:/特征提取/app列表/appData_list_4.txt'
    user_count = 0
    catch = 0
    with open(file, 'r', encoding='UTF-8') as rf:
        lines = rf.readlines()
        for line in lines:
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.split(',')

            his_data = history_dict.get(idnum)
            # print(idnum, his_data)
            user_count +=1
            if his_data:
                catch +=1
                handle_mapping(apps,his_data)

    with open(file1, 'r', encoding='UTF-8') as rf:
        lines = rf.readlines()
        for line in lines:
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.split(',')

            his_data = history_dict.get(idnum)
            # print(idnum, his_data)
            user_count += 1
            if his_data:
                catch += 1
                handle_mapping(apps, his_data)

    with open(file2, 'r', encoding='UTF-8') as rf:
        lines = rf.readlines()
        for line in lines:
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.split(',')

            his_data = history_dict.get(idnum)
            # print(idnum, his_data)
            user_count += 1
            if his_data:
                catch += 1
                handle_mapping(apps, his_data)

    with open(file3, 'r', encoding='UTF-8') as rf:
        lines = rf.readlines()
        for line in lines:
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.split(',')

            his_data = history_dict.get(idnum)
            # print(idnum, his_data)
            user_count += 1
            if his_data:
                catch += 1
                handle_mapping(apps, his_data)

    with open(file4, 'r', encoding='UTF-8') as rf:
        lines = rf.readlines()
        for line in lines:
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.split(',')

            his_data = history_dict.get(idnum)
            # print(idnum, his_data)
            user_count += 1
            if his_data:
                catch += 1
                handle_mapping(apps, his_data)

    print(user_count,catch,len(app_dict))

    # print(app_dict)

    for k in app_dict:
        check = app_dict[k]
        count = check['count']
        apply = check['apply']
        approve = check['approve']
        overdue = check['overdue']
        loan_amount = check['loan_amount']
        pd0 = check['pd0']
        pd7 = check['pd7']
        pd10 = check['pd10']
        pd14 = check['pd14']
        M1 = check['M1']
        M2 = check['M2']
        M3 = check['M3']


        # 平均指标
        check['avg_apply'] = round(apply / count,2)
        check['avg_approve'] = round(approve / count, 2)
        check['avg_overdue'] = round(overdue / count, 2)
        check['avg_loan_amount'] = round(loan_amount / count, 2)
        check['avg_pd0'] = round(pd0 / count, 2)
        check['avg_pd7'] = round(pd7 / count, 2)
        check['avg_pd10'] = round(pd10 / count, 2)
        check['avg_pd14'] = round(pd14 / count, 2)
        check['avg_M1'] = round(M1 / count, 2)
        check['avg_M2'] = round(M2 / count, 2)
        check['avg_M3'] = round(M3 / count, 2)

        check['approve_rate'] = round(approve / apply, 2)
        if approve != 0:
            check['overdue_rate'] = round(overdue / approve, 2)
        else:
            check['overdue_rate'] = 0.00

        if overdue !=0:
            check['pd7_rate'] = round(pd7 / overdue, 2)
            check['pd10_rate'] = round(pd10 / overdue, 2)
            check['pd14_rate'] = round(pd14 / overdue, 2)

            check['M1_rate'] = round(M1 / overdue, 2)
            check['M2_rate'] = round(M2 / overdue, 2)
            check['M3_rate'] = round(M3 / overdue, 2)
        else:
            check['pd7_rate'] = 0.00
            check['pd10_rate'] = 0.00
            check['pd14_rate'] = 0.00

            check['M1_rate'] = 0.00
            check['M2_rate'] = 0.00
            check['M3_rate'] = 0.00




    with open("record.json", "w",encoding='utf-8') as f:
        json.dump(app_dict, f,ensure_ascii=False)
    print("加载入文件完成...")


def handle_mapping(apps,his_data):
    apply = his_data[0]
    approve = his_data[1]
    overdue = his_data[2]
    loan_amount = his_data[3]
    max_overdue = his_data[4]
    pd0 = 0
    pd7 = 0
    pd10 = 0
    pd14 = 0
    M1 = 0
    M2 = 0
    M3 = 0
    if max_overdue > 0:
        pd0 = 1
        if max_overdue >= 7:
            pd7 = 1
            if max_overdue >= 10:
                pd10 = 1

            if max_overdue >= 14:
                pd14 = 1

            if max_overdue >= 30:
                M1 = 1

            if max_overdue >= 60:
                M2 = 1

            if max_overdue >= 90:
                M3 = 1


    for a in apps:
        # count += 1
        # print(a)
        check = app_dict.get(a)
        if check is None:
            check = {'apply':apply,'approve':approve,'overdue':overdue
                           ,'loan_amount':loan_amount,'pd0':pd0,'pd7':pd7,
                     'pd10':pd10,'pd14':pd14,
                     'M1':M1,'M2':M2,'M3':M3,'count':1}
            app_dict[a] = check
        else:
            check['apply'] += apply
            check['approve'] += approve
            check['overdue'] += overdue
            check['loan_amount'] += loan_amount
            check['pd0'] += pd0
            check['pd7'] += pd7
            check['pd10'] += pd10
            check['pd14'] += pd14

            check['M1'] += M1
            check['M2'] += M2
            check['M3'] += M3

            check['count'] += 1


def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(text, cut_all=False)
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)
    # print(seg_list)

    return seg_list

def app_lda_analysis():
    doc_complete = []
    file = 'D:/特征提取/app列表/appData_list_0.txt'
    with open(file, 'r', encoding='UTF-8') as rf:
        count = 0
        lines = rf.readlines()
        for line in lines:
            count+=1
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.replace('","','')
            apps = apps.replace('["', '')
            apps = apps.replace('"]', '')
            # apps = apps.split(',')
            print(apps)
            print(type(jieba_split(apps)))
            doc_complete.append(jieba_split(apps))

            if count ==1000:
                break
            # for a in apps:
            #     count += 1
            #     print(a)

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary(doc_complete)
    #
    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_complete]

    # 使用 gensim 来创建 LDA 模型对象
    Lda = models.ldamodel.LdaModel

    # 在 DT 矩阵上运行和训练 LDA 模型
    ldamodel = Lda(doc_term_matrix, num_topics=10, id2word=dictionary, passes=50)

    print(ldamodel.print_topics(num_topics=10, num_words=10))

def app_check():
    with open("record.json", 'r',encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
        for k in load_dict:
            pass
        
    print(load_dict)

if __name__ == '__main__':
    import time
    starttime = time.time()

    # 1 app关键词统计
    # app_keyword_analysis()

    # 2 主题分类在app数据上的应用
    # app_lda_analysis()

    # 加载历史数据
    # order_dict()
    # 3 将历史表现映射到app列表
    # app_history_mapping()

    # 对一些关键词app进行统计
    app_check()

    endtime = time.time()
    print(' cost time: ', endtime - starttime)
