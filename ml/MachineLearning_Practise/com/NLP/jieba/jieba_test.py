#!-*- coding:utf8-*-
import jieba
from com.NLP.jieba import prov_dict
import pymongo
import operator;
from com.risk_score.feature_extact import setting

client = pymongo.MongoClient(setting.host)
db = client.call_test
coll = db.call_info

# 结巴中文分词涉及到的算法包括：
# (1) 基于Trie树结构实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图（DAG)；
# (2) 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合；
# (3) 对于未登录词，采用了基于汉字成词能力的HMM模型，使用了Viterbi算法。
# 结巴中文分词支持的三种分词模式包括：
# (1) 精确模式：试图将句子最精确地切开，适合文本分析；
# (2) 全模式：把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义问题；
# (3) 搜索引擎模式：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
#

def test():
    # 全模式
    text = "吉林省长春市万通高层A14栋610室"
    seg_list = jieba.cut(text, cut_all=True)
    print(u"[全模式]: ", "/ ".join(seg_list))

    # 精确模式
    seg_list = jieba.cut(text, cut_all=False)
    print(u"[精确模式]: ", "/ ".join(seg_list))

    # 默认是精确模式
    seg_list = jieba.cut(text)
    print(u"[默认模式]: ", "/ ".join(seg_list))

    # 搜索引擎模式
    seg_list = jieba.cut_for_search(text)
    print(u"[搜索引擎模式]: ", "/ ".join(seg_list))

def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(text, cut_all=False)
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)
    # print(seg_list)

    return seg_list

# 省字典
province_dict = {}
# 市字典
city_dict = {}
# 县字典
county_dict = {}
# 构造省市县数据字典
def read_prov():
    prov = prov_dict.all.get('prov')

    # # 省字典
    # province_dict = {}
    # # 市字典
    # city_dict = {}
    # # 县字典
    # county_dict = {}
    for p in prov:
        # print(p.get('province'))
        province_dict[p.get('province')] = 0
        city_dict[p.get('province')] = {}
        county_dict[p.get('province')] = {}

        city = p.get('city')
        # print(city)
        for c in city:
            curr = city_dict[p.get('province')]
            city_name = c.get('name')
            county = c.get('county')
            # print(city_name)
            curr[city_name] = 0

            curr1 = county_dict[p.get('province')]
            curr1[city_name] = {}
            for o in county:
                oc = curr1[city_name]
                oc[o] = 0
        #
        # print(city_dict)
        # print(county_dict)

def mongo_read():
    results = coll.find({}, {'_id': 0, 'emergencer': 0,'calls': 0,'contacts': 0,'phone': 0,'carr_phone': 0 }).limit(100)
    for d in results:
        if d:
            idnum = d.get('id_num')
            addresses = d.get('addresses')
            l_a = addresses.get('L')
            if l_a:
                a = l_a[0]
                a = a.replace(',','')
                a = a.replace('|', '')
                a = a.replace('-', '')
                # print(l_a)
                seg_list = jieba_split(a)
                p = seg_list[0]
                c = seg_list[1]
                cou = seg_list[2]

                temp = county_dict.get(p)
                if temp:
                    temp = temp.get(c)
                    print(c)
                    if temp:
                        temp = temp.get(cou)
                        print(cou, temp)
                for s in range(len(seg_list)):
                    if seg_list[s].strip() is not ',' and seg_list[s].strip() is not '':

                        # print(seg_list[s])
                        temp = seg_list[s]
                        if s == 0:
                            prov = county_dict.get(temp)
                            # if prov:

                            print(temp)

# 类似将广州映射为广州省
def address_map():
    prov = prov_dict.all.get('prov')
    map_dict = {}
    for p in prov:
        # print(p.get('province'))
        province = p.get('province')
        map_dict[province] = 0

        city = p.get('city')
        for c in city:
            # print(c)
            city_name = c.get('name')
            county = c.get('county')
            # print(city_name)
            map_dict[city_name] = 0
            for o in county:
                # print(o)
                map_dict[o] = 0
        #
        # print(city_dict)
    # print(map_dict)

    final_map = {}
    for item in map_dict:
        if len(item) > 2:
            temp = item[:-1]
        else:
            temp = item
        # print(item,temp)

        final_map[temp] = item

    # print(final_map)

    return final_map


# 将分词后自计数，并构造字典
def mongo_read1():
    all_dict = {}
    null_count = 0
    risk_count = 0
    results = coll.find({}, {'_id': 0, 'emergencer': 0,'calls': 0,'contacts': 0,'phone': 0,'carr_phone': 0 }).limit(12000000)
    addresses_map = address_map()
    for d in results:
        try:
            if d:
                idnum = d.get('id_num')
                addresses = d.get('addresses')
                l_a = addresses.get('L')
                if l_a:

                    null_count+=1
                    temp = l_a[0]
                    temp = temp.replace(',','')
                    temp = temp.replace('|', '')
                    temp = temp.replace('-', '')
                    temp = temp.replace(' ', '')
                    temp = temp.strip()
                    # print(temp)
                    seg_list = jieba_split(temp)
                    # print(seg_list)
                    # 统计个地区词频
                    # area_count(seg_list, addresses_map, all_dict)

                    risk = history_dict.get(idnum)
                    if risk:
                        risk_count += 1
                        risk_analysis(seg_list, addresses_map, all_dict,idnum,risk)

        except Exception as e:
            # print(e)
            continue

    print('历史订单数量',len(history_dict),'地址不空',null_count,'有历史表现',risk_count)
    # print(all_dict)

    final_dict = {}
    for k in all_dict:
        apply = all_dict[k][0]
        approve = all_dict[k][1]
        overdue = all_dict[k][2]
        loanamount = all_dict[k][3]
        person_count = all_dict[k][4]

        # 通过率
        approve_rate = 0
        if apply != 0:
            approve_rate = round(float(approve / apply),2)

        # 逾期率
        overdue_rate = 0
        if approve != 0:
            overdue_rate = round(float(overdue / approve),2)

        # 平均放款金额
        avg_loanamount = 0
        if person_count != 0:
            avg_loanamount = round(float(loanamount / person_count),2)

        # 平均申请次数
        avg_apply = 0
        avg_apply = round(float(apply / person_count),2)

        # print(k,'通过率',approve_rate,'逾期率',overdue_rate,'平均放款金额',avg_loanamount,'平均申请',avg_apply)
        final_dict[k] = (approve_rate,overdue_rate,avg_loanamount,avg_apply,person_count)

    print(final_dict)


    # print(sorted(all_dict.items(),key=operator.itemgetter(1),reverse=True))

# 各地区词频统计
def area_count(seg_list,addresses_map,all_dict):
    a = seg_list[0]

    short = addresses_map.get(a)
    if short:
        a = short
    if all_dict.get(a) != None:
        all_dict[a] += 1
    else:
        all_dict[a] = 1

    # print(all_dict.get(a))
    # print(all_dict.get('广东省'))

    b = seg_list[1]
    short = addresses_map.get(b)
    if short:
        b = short
    if all_dict.get(b) != None:
        all_dict[b] += 1
    else:
        all_dict[b] = 1

    c = seg_list[2]
    short = addresses_map.get(c)
    if short:
        c = short
    if all_dict.get(c) != None:
        all_dict[c] += 1
    else:
        all_dict[c] = 1

    d = seg_list[3]
    short = addresses_map.get(d)
    if short:
        d = short
    if all_dict.get(d) != None:
        all_dict[d] += 1
    else:
        all_dict[d] = 1

def risk_analysis(seg_list, addresses_map, all_dict,idnum,risk):

    if len(seg_list) < 3:
        a = seg_list[0]

        #
        if risk:
            short = addresses_map.get(a)
            if short:
                a = short
            if all_dict.get(a) != None:
                all_dict[a][0] += risk[0]
                all_dict[a][1] += risk[1]
                all_dict[a][2] += risk[2]
                all_dict[a][3] += risk[3]
                all_dict[a][4] += 1
            else:
                all_dict[a] = [0,0,0,0,0]
                all_dict[a][0] += risk[0]
                all_dict[a][1] += risk[1]
                all_dict[a][2] += risk[2]
                all_dict[a][3] += risk[3]
                all_dict[a][4] += 1

            # print(all_dict.get(a))
            # print(all_dict.get('广东省'))

            b = seg_list[1]
            short = addresses_map.get(b)
            if short:
                b = short
            if all_dict.get(b) != None:
                all_dict[b] += risk[0]
            else:
                all_dict[b] = [0,0, 0, 0, 0]
                all_dict[b][0] += risk[0]
                all_dict[b][1] += risk[1]
                all_dict[b][2] += risk[2]
                all_dict[b][3] += risk[3]
                all_dict[b][4] += 1

    elif len(seg_list) == 3:
        a = seg_list[0]
        if risk:
            short = addresses_map.get(a)
            if short:
                a = short
            if all_dict.get(a) != None:
                all_dict[a][0] += risk[0]
                all_dict[a][1] += risk[1]
                all_dict[a][2] += risk[2]
                all_dict[a][3] += risk[3]
                all_dict[a][4] += 1
            else:
                all_dict[a] = [0, 0, 0, 0, 0]
                all_dict[a][0] += risk[0]
                all_dict[a][1] += risk[1]
                all_dict[a][2] += risk[2]
                all_dict[a][3] += risk[3]
                all_dict[a][4] += 1

            # print(all_dict.get(a))
            # print(all_dict.get('广东省'))

            b = seg_list[1]
            short = addresses_map.get(b)
            if short:
                b = short
            if all_dict.get(b) != None:
                all_dict[b] += risk[0]
            else:
                all_dict[b] = [0, 0, 0, 0, 0]
                all_dict[b][0] += risk[0]
                all_dict[b][1] += risk[1]
                all_dict[b][2] += risk[2]
                all_dict[b][3] += risk[3]
                all_dict[b][4] += 1

            c = seg_list[2]
            short = addresses_map.get(c)
            if short:
                c = short
            if all_dict.get(c) != None:
                all_dict[c][0] += risk[0]
                all_dict[c][1] += risk[1]
                all_dict[c][2] += risk[2]
                all_dict[c][3] += risk[3]
                all_dict[c][4] += 1
            else:
                all_dict[c] = [0, 0, 0, 0, 0]
                all_dict[c][0] += risk[0]
                all_dict[c][1] += risk[1]
                all_dict[c][2] += risk[2]
                all_dict[c][3] += risk[3]
                all_dict[c][4] += 1

    else:
        a = seg_list[0]
        if risk:
            short = addresses_map.get(a)
            if short:
                a = short
            if all_dict.get(a) != None:
                all_dict[a][0] += risk[0]
                all_dict[a][1] += risk[1]
                all_dict[a][2] += risk[2]
                all_dict[a][3] += risk[3]
                all_dict[a][4] += 1
            else:
                all_dict[a] = [0, 0, 0, 0, 0]
                all_dict[a][0] += risk[0]
                all_dict[a][1] += risk[1]
                all_dict[a][2] += risk[2]
                all_dict[a][3] += risk[3]
                all_dict[a][4] += 1

            # print(all_dict.get(a))
            # print(all_dict.get('广东省'))

            b = seg_list[1]
            short = addresses_map.get(b)
            if short:
                b = short
            if all_dict.get(b) != None:
                all_dict[b] += risk[0]
            else:
                all_dict[b] = [0, 0, 0, 0, 0]
                all_dict[b][0] += risk[0]
                all_dict[b][1] += risk[1]
                all_dict[b][2] += risk[2]
                all_dict[b][3] += risk[3]
                all_dict[b][4] += 1

            c = seg_list[2]
            short = addresses_map.get(c)
            if short:
                c = short
            if all_dict.get(c) != None:
                all_dict[c][0] += risk[0]
                all_dict[c][1] += risk[1]
                all_dict[c][2] += risk[2]
                all_dict[c][3] += risk[3]
                all_dict[c][4] += 1
            else:
                all_dict[c] = [0, 0, 0, 0, 0]
                all_dict[c][0] += risk[0]
                all_dict[c][1] += risk[1]
                all_dict[c][2] += risk[2]
                all_dict[c][3] += risk[3]
                all_dict[c][4] += 1

            d = seg_list[3]
            short = addresses_map.get(d)
            if short:
                d = short
            if all_dict.get(d) != None:
                all_dict[d][0] += risk[0]
                all_dict[d][1] += risk[1]
                all_dict[d][2] += risk[2]
                all_dict[d][3] += risk[3]
                all_dict[d][4] += 1
            else:
                all_dict[d] = [0, 0, 0, 0, 0]
                all_dict[d][0] += risk[0]
                all_dict[d][1] += risk[1]
                all_dict[d][2] += risk[2]
                all_dict[d][3] += risk[3]
                all_dict[d][4] += 1

    # print(all_dict)

# {'idnum': '',
#  'apply_pdl_all': 0.0, 'apply_int_all': 0.0, 'apply_sum_all': 0.0,
#  'approve_pdl_all': 0.0, 'approve_int_all': 0.0,'approve_sum_all': 0.0,
#  'overdue_pdl_all': 0.0, 'overdue_int_all': 0.0, 'overdue_sum_all': 0.0,
#  'loanamount_pdl_all': 0.0, 'loanamount_int_all': 0.0, 'loanamount_sum_all': 0.0,
#  'maxOverdue_pdl_all': -99999.0, 'maxOverdue_int_all': -99999.0, 'maxOverdue_sum_all': -99999.0,
#                  }
def order_dict():
    file = 'D:/特征提取/历史特征数据/value_list_0.txt'
    file1 = 'D:/特征提取/历史特征数据/value_list_1.txt'
    file2 = 'D:/特征提取/历史特征数据/value_list_2.txt'
    file3 = 'D:/特征提取/历史特征数据/value_list_3.txt'

    file4 = 'D:/特征提取/历史特征数据/value_list_4.txt'
    history_dict = {}
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

    return history_dict
if __name__ == '__main__':
    import time

    starttime = time.time()

    # test()
    # read_prov()

    history_dict = order_dict()
    mongo_read1()

    # address_map()

    endtime = time.time()
    print(' cost time: ', endtime - starttime)
