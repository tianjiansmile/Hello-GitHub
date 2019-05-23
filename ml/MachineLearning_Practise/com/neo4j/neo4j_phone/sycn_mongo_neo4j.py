# -*- coding:utf-8 -*-
import pymysql
from com.risk_score.feature_extact import setting
import pymongo
import hashlib
# 跑全量通话数据，通讯录，紧急联系人数据，撞出用户之间关系: 通话关系，通讯录关系，共用号码，共用紧急联系人，互为紧急联系人
#   1 读入全量用户，电话，构建字典


client = pymongo.MongoClient(setting.host)
db = client.call_test
coll = db.call_info

id_phone_dict = {}

def getJinpanOrders(data_index,wf):
    read_con = pymysql.connect(host=setting.mysql_host, user=setting.mysql_user,
                               password=setting.mysql_pass, database=setting.musql_db + data_index, port=3306,
                               charset='utf8')
    cursor = read_con.cursor()
    sql = "select distinct ub.identity_no,up.phone_num from user_basics_info ub left join user_phone_info up on ub.user_id=up.user_id"
    cursor.execute(sql)
    result = cursor.fetchall()
    for re in result:
        # print(re)
        phone = re[1]
        idnum = re[0]
        if phone and idnum:
            wf.write(idnum + ', ' + phone + '\n')



    read_con.close()
    # for order in orders:
    #     file.write(order['merchant_id'] + ',' + order['loan_order_no'] + '\n')
    # file.close()

def write_file(file):
    with open(file, 'w') as wf:
        for i in range(10):
            getJinpanOrders(str(i),wf)

def md5(src):
    m = hashlib.md5()
    m.update(str.encode(src))
    return m.hexdigest().lower()

def id_phone_build(file):
    count = 0
    count1 = 0
    with open(file, 'r') as rf:
        result= rf.readlines()
        for re in result:
            count +=1
            re = re.replace('\n','')
            re = re.replace(' ', '')
            line = re.split(',')
            phone = line[1]
            idnum = line[0]

            if phone :
                temp = id_phone_dict.get(phone)

                if temp:  # 如果可以查到这个人，而且身份证不同，这是共用号码的情况
                    if temp != idnum:
                        # print(temp,idnum,phone)
                        count1+=1
                else:  # 不存在则存入
                    if idnum:
                        id_phone_dict[phone] = idnum

            # print(phone, id_phone_dict.get(phone))
        print('用户量',count,'dict len', len(id_phone_dict),'共用量',count1)

# 全量用户转化为节点文件，准备导入
# node.csv 格式
# nid:ID,is_black,overdue,:LABEL
# 9927bf72b7c00c1f86a7bb50b158916e,-1,0,person
# rels.csv 格式
# :START_ID,:END_ID,:TYPE,time,call_len
# 789899f7c52d4371f5974e5381e4bbc2,d1fc1a3f27986de00791b296c03c40a3,same_phone,0,0
def node_file(file):
    count = 0
    with open(file, 'r') as rf:
        with open('D:/develop/data/network/whole_data/md5node.csv', 'w') as nwf:
            result= rf.readlines()
            nwf.write('nid:ID,is_black,overdue,:LABEL' + '\n')
            for re in result:
                re = re.replace('\n','')
                line = re.split(',')
                phone = line[1]
                idnum = line[0]

                md5_id = md5(idnum)
                nwf.write(md5_id + ',-1,0,person'+ '\n')

                count += 1
                print(count)
                # if count == 10:
                #     break

    # with open('D:/develop/data/network/md5_rels.csv', 'r') as rf:
    #     result = rf.readlines()
    #     for re in result:
    #         count +=1
    #         if count ==10:
    #             break
    #         print(re)

# 共用电话的数据写入文件
def same_phone_file(file):
    count = 0
    count1 = 0
    with open(file, 'r') as rf:
        with open('D:/develop/data/network/whole_data/same_phone.txt', 'w') as nwf:
            result = rf.readlines()
            for re in result:
                count += 1
                re = re.replace('\n', '')
                line = re.split(',')
                phone = line[1]
                idnum = line[0]

                if phone:
                    temp = id_phone_dict.get(phone)

                    if temp:  # 如果可以查到这个人，而且身份证不同，这是共用号码的情况
                        if temp != idnum:
                            print(temp, idnum, phone,count1)
                            count1 += 1
                            md5_id = md5(idnum)
                            md5_temp = md5(temp)
                            nwf.write(md5_id + ','+md5_temp+',same_phone,0,0' + '\n')
                    else:  # 不存在则存入
                        if idnum:
                            id_phone_dict[phone] = idnum
            # print(line)

        print('用户量', count, 'dict len', len(id_phone_dict), '共用量', count1)

def loop_mongo(rel_file):
    # results = coll.find({'id_num': '342501198611124046'},{'_id':0,'addresses':0,}).limit(10)
    results = coll.find().limit(10).batch_size(500)
    with open(rel_file, 'a') as wf:
        for data_dict in results:
            if data_dict:
                idnum = data_dict.get('id_num')
                emergencer = data_dict.get('emergencer')
                calls = data_dict.get('calls')
                contacts = data_dict.get('contacts')
                if idnum:
                    if calls:
                        loop_call(calls,idnum,wf)
                    # print(contacts)
                    if contacts:
                        loop_contact(contacts,idnum,wf)

                    # print(emergencer)
                    if emergencer:
                        loop_emergence(emergencer, idnum,wf)

def loop_call(calls,s_id,wf):
    for call in calls:
        phone = call.get('phone')
        call_len = call.get('call_len')
        times = call.get('times')
        contact_name = call.get('contact_name')
        if phone and len(phone) == 11:
            t_id = id_phone_dict.get(phone)
            if t_id: # 撞到了
                print(s_id,t_id,contact_name)

                md5_id = md5(s_id)
                md5_temp = md5(t_id)
                wf.write(md5_id + ',' + md5_temp + ',call,'+ str(times) + ','+str(call_len)+ '\n')

def loop_contact(contacts,s_id,wf):
    for call in contacts:
        # print(call,contacts.get(call))
        phone = call
        contact_name = contacts.get(call)
        if phone and len(phone) == 11:
            t_id = id_phone_dict.get(phone)
            if t_id: # 撞到了
                print(s_id,t_id,contact_name)
                md5_id = md5(s_id)
                md5_temp = md5(t_id)
                wf.write(md5_id + ',' + md5_temp + ',contact,0,0' + '\n')

def loop_emergence(emergence,s_id,wf):
    for phone in emergence.keys():
        # print(phone)
        if phone and len(phone) == 11:
            t_id = id_phone_dict.get(phone)
            if t_id:  # 撞到了
                print(s_id, t_id, phone)
                md5_id = md5(s_id)
                md5_temp = md5(t_id)
                wf.write(md5_id + ',' + md5_temp + ',emergencer,0,0' + '\n')

if __name__ == '__main__':
    import time
    starttime = time.time()
    file = 'D:/develop/data/network/whole_data/id_phone.txt'
    # 1 读入全量数据到文件，为字典做准备
    # write_file(file)

    # 2 读取全量数据，构造字典,
    id_phone_build(file)
    # 并且将共用电话的情况写入文件
    # same_phone_file(file)

    # 3 构造的全量字典可以先将全量用户转化为节点文件先存下来
    # node_file(file)


    rel_file = 'D:/develop/data/network/whole_data/import/md5rels.csv'
    # 4 开始读mongodb数据,撞数据，将撞到的数据写下来
    loop_mongo(rel_file)

    endtime = time.time()
    print(' cost time: ', endtime - starttime)

