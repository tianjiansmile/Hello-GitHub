import json
import requests
import re
import pymysql
import csv

# 主要实现对用户的通话数据做融合处理---- 数据融合---- 用户纯电话同构网络
class Phone:
    def __init__(self,phone,call_record):
        self.phone_d = phone[0]
        self.call_record_d = call_record

    def phone(self):
        phone_d = self.phone_d
        if phone_d != None:
            print(len(phone_d),phone_d[0])


    # 通过记录电话提取
    def call_extact(self):

        try:
            call_history = {}
            for callrec in self.call_record_d:
                if callrec != None:
                        if callrec != None and callrec.strip() != '':
                            ca = json.loads(callrec)
                            for c in ca:
                                p = c.get('phone')
                                # 通话时长
                                use_time = c.get('use_time')
                                # 呼叫类型
                                type = c.get('type')
                                if call_history.get(p) == None:
                                    call_history[p] =1
                                else:
                                    call_history[p] +=1
            print('call_history: ',call_history)
            print(len(call_history.keys()))
            self.call_history = call_history

        except Exception as e:
            raise TypeError('Exception') from e

    # 写入csv文件
    def write_to_csv(self,id):
        path = 'D:/neo4j/neo4j-desktop-data/neo4jDatabases/database-7780a393-231c-4389-8b43-1bfc06f66183/installation-3.4.9/import/'

        phone = self.phone_d[0]

        with open(path+id+'node.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # 标记是否为借款人
            writer.writerow(['id','is_loaner'])
            writer.writerow([phone,1])
            for row in self.call_history.keys():
                print(row,-1)
                writer.writerow([row,-1])

        with open(path+id+'rel.csv', 'w', newline='') as f:
            phone = self.phone_d[0]
            print(self.call_history)
            writer = csv.writer(f)
            writer.writerow(['from_id','property1','property2','to_id'])
            for row in self.call_history.keys():
                print(phone,'called',self.call_history[row],row)
                writer.writerow([phone,'called',self.call_history[row],row])





def mysql_connect():
    db = pymysql.connect(host="139.224.118.4", user="root",
                         password="zhouao.123", db="user_db", port=8066)

    cur = db.cursor()
    id = '510681199710165312'
    phone_sql = "select distinct phone_num from user_phone_info up LEFT JOIN " \
          "user_basics_info ub on ub.user_id=up.user_id where ub.identity_no = %s"

    call_sql = "select user_data from user_call_record up LEFT JOIN user_basics_info ub on ub.user_id=up.user_id " \
               "where ub.identity_no = %s"



    cur.execute(phone_sql,id)  # 像sql语句传递参数
    p_list = cur.fetchall()

    cur.execute(call_sql,id)
    c_list = cur.fetchall()
    cc = []
    for i in range(len(c_list)):
        cc.append(c_list[i][0])

    phone = Phone(p_list,cc)
    phone.phone()
    phone.call_extact()
    phone.write_to_csv(id)

    db.close()

if __name__ == '__main__':
    mysql_connect()