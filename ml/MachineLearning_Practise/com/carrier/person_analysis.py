# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import json
import os
import sys
import operator
from mpl_toolkits.mplot3d import Axes3D

# 主要用于提取用户变量，对运营商统计报告做数据项分析
class variable_person:

    def __init__(self,filepath,filename):
        with open(filepath+filename, 'r',encoding = 'utf-8') as load_f:
            data = json.load(load_f)

        if data != None:
            data_json = json.dumps(data)
            self.json_data = json.loads(data_json, strict=False)  # json全量数据
            self.setBasicDate()


    def setBasicDate(self):
        data = self.json_data
        if data.get('JSON_INFO') != None:
            data = data.get('JSON_INFO')

        self.varls = []
        self.main_service = data.get('main_service')
        self.main_service_compute()

        self.contact_region = data.get('contact_region')
        self.contact_region_compute()

        self.user_info_check = data.get('user_info_check')
        self.user_info_compute()

        self.behavior_check = data.get('behavior_check')
        self.behavior_check_compute()

    # 提取变量 （main_service）主要服务商通话总次数 main_service_count 以及其他相关变量
    def main_service_compute(self):
        main_service = self.main_service
        data_type = self.type_check(main_service)
        main_service_count = 0
        uncom_count = 0
        bank_count = 0
        special_count = 0
        airplan_count = 0
        if data_type == 'list':
            for item in main_service:
                if item.get('total_service_cnt')!=None:
                    total_service_cnt = item.get('total_service_cnt')
                    main_service_count = main_service_count+total_service_cnt
                    if item.get('company_type') == '通信服务机构':
                        uncom_count = uncom_count + total_service_cnt
                    elif item.get('company_type') == '银行':
                        bank_count = bank_count + total_service_cnt
                    elif item.get('company_type') == '特种服务':
                        special_count = special_count + total_service_cnt
                    elif item.get('company_type') == '铁路航空':
                        airplan_count = airplan_count + total_service_cnt
        elif data_type == 'dict':
            print('not handle yet')

        # print(main_service_count,uncom_count,bank_count,special_count,airplan_count)
        self.varls.append(main_service_count)
        self.varls.append(uncom_count)
        self.varls.append(bank_count)
        self.varls.append(special_count)
        self.varls.append(airplan_count)

    # 联系地区总数统计
    def contact_region_compute(self):
        contact_region = self.contact_region
        data_type = self.type_check(contact_region)
        contact_region_count = 0
        if data_type == 'list':
            # print(contact_region)
            contact_region_count = len(contact_region)
        elif data_type == 'dict':
            print('not handle yet')

        self.varls.append(contact_region_count)

    #  联系电话灰名单得分 电话号码注册过的相关企业数量 查询过该用户的相关企业数量
    def user_info_compute(self):
        user_info_check = self.user_info_check
        data_type = self.type_check(user_info_check)
        phone_gray_score = 0
        contacts_class1_blacklist = 0

        if data_type == 'list':
            for item in user_info_check:
                check_black_info = item.get('check_black_info')
                if check_black_info != None:
                    phone_gray_score = check_black_info.get('phone_gray_score')

                check_search_info = item.get('check_search_info')   # 需要判断一下相关企业的类型  CASH_LOAN
                if check_search_info != None:
                    register_org_cnt = check_search_info.get('register_org_cnt')
                    searched_org_cnt = check_search_info.get('searched_org_cnt')

        elif data_type == 'dict':
            check_black_info = user_info_check.get('check_black_info')
            if check_black_info != None:
                phone_gray_score = check_black_info.get('phone_gray_score')

            check_search_info = user_info_check.get('check_search_info')
            if check_search_info != None:
                register_org_cnt = check_search_info.get('register_org_cnt')
                searched_org_cnt = check_search_info.get('searched_org_cnt')


        self.varls.append(phone_gray_score)
        self.varls.append(register_org_cnt)
        self.varls.append(searched_org_cnt)

    def loan_evidence_compute(self,evidence):
        # 主叫
        call_count = 0
        # 被呼叫
        called_count = 0
        if evidence != None:
            if evidence.find('[总计]') == 0:
                call_local = evidence.index('主叫')
                called_local = evidence.index('被叫')
                call_count = int(evidence[call_local+2:call_local+4].replace('次',''))
                called_count = int(evidence[called_local+2:called_local+4].replace('次',''))
            elif evidence == '未找到贷款类相关号码':
                pass
            elif evidence.find('联系列表') == 0:
                ev_list = evidence.split('：')
                ev_loan_list = ev_list[1].split('，')
                for loan in ev_loan_list:
                    call_local = loan.index('主叫')
                    called_local = loan.index('被叫')
                    call_count += int(loan[call_local + 2:call_local + 4].replace('次', ''))
                    called_count += int(loan[called_local + 2:called_local + 4].replace('次', ''))
            else:
                print('kidding me....................................')

            print(call_count,called_count,evidence)

        return call_count,called_count

    # 通话行为监测统计
    def behavior_check_compute(self):
        behavior_check = self.behavior_check
        data_type = self.type_check(behavior_check)
        behavior_check_score = 0
        contact_each_other_result = 0
        contact_110_result = 0
        contact_lawyer_result = 0
        contact_court_result = 0
        contact_loan_result = 0
        contact_bank_result = 0
        contact_credit_card_result = 0
        contact_night_result = 0

        if data_type == 'list':
            # print(behavior_check)
            for item in behavior_check:
                score = item.get('score')
                behavior_check_score = behavior_check_score + score
                if item.get('check_point_cn') == '互通过电话的号码数量':
                    contact_each_other_result = item.get('result')
                elif item.get('check_point_cn') == '与110电话通话情况':
                    contact_110_result = item.get('result')
                elif item.get('check_point_cn') == '与律师电话通话情况':
                    contact_lawyer_result = item.get('result')
                elif item.get('check_point_cn') == '与法院电话通话情况':
                    contact_court_result = item.get('result')
                elif item.get('check_point_cn') == '与贷款类号码联系情况':
                    contact_loan_result = item.get('result')
                    evidence = item.get('evidence')
                    call_count, called_count = self.loan_evidence_compute(evidence)
                elif item.get('check_point_cn') == '与银行类号码联系情况':
                    contact_bank_result = item.get('result')
                elif item.get('check_point_cn') == '与信用卡类号码联系情况':
                    contact_credit_card_result = item.get('result')
                elif item.get('check_point_cn') == '夜间活动情况':
                    contact_night_result = item.get('result')




        elif data_type == 'dict':
            print('not handle yet')

        self.varls.append(behavior_check_score)
        self.varls.append(contact_each_other_result)
        self.varls.append(contact_110_result)
        self.varls.append(contact_lawyer_result)
        self.varls.append(contact_court_result)
        self.varls.append(contact_loan_result)
        self.varls.append(contact_bank_result)
        self.varls.append(contact_credit_card_result)
        self.varls.append(contact_night_result)
        # 最后加两个特征
        self.varls.append(call_count)
        self.varls.append(called_count)


    def type_check(self,item):
        data_type = 'list'
        if type(item).__name__ == 'list':
            data_type = 'list'
        elif type(item).__name__ == 'dict':
            data_type = 'dict'
        elif type(item).__name__ == 'str':
            data_type = 'str'
        elif type(item).__name__ == 'int':
            data_type = 'int'
        else:
            # item_name = self.namestr(item, locals())
            # print('data undefined')
            data_type = 'undefined'

        return data_type

# 提取原始数据到文件
def loop_file(carr_path,list):
    with open('D:/Develop/test/carrier/report_var/report_add_call_val.txt', 'w') as rf:
        pre = ['54','12','42','0','0','65','68','15']
        for risk in list:
            try:
                judge = variable_person(carr_path, risk[0] + '.json')

                rid = risk[0]
                score = risk[1]
                risk_status = risk[2]
                varl = judge.varls
                varl.append(score)
                varl.append(risk_status)
                # 写入所有变量
                print(risk[0], varl)
                val_line = ''
                for i in range(len(varl)):
                    if i < len(varl)-1:
                        val_line += str(varl[i]) + ','
                    else:
                        val_line += str(varl[i])

                print(val_line)
                if not operator.eq(pre[:7],judge.varls[:7]):
                    pass
                    rf.write(val_line+'\n')

                pre = judge.varls




            except Exception as e:
                print("解析json出错...%", risk[0])
                # logging.exception("解析json出错...%", filename)
                # raise TypeError('bad json') from e
            continue

# 提取数据结果到文件
def read_risk_data(risk_path):
    list = []
    with open(risk_path, 'r') as f:
        for line in f.readlines():
            # 去掉换行符
            line = line.strip('\n')
            temp = line.split(',')
            rid = temp[0]
            score = temp[1]
            risk_status = temp[2]
            score = score.lstrip()
            risk_status = risk_status.lstrip()

            list.append((rid, score,risk_status))

    return list

# 对数据进行清洗转换
def data_handler(file):
    with open(file, 'r') as f:
        tran_val = 0
        # 矩阵行数
        array_lines = f.readlines()
        number_lines = len(array_lines)
        # 初始化一个特征矩阵,先取三个典型特征看看
        data_matr = zeros((number_lines,7))
        # 结果向量
        classLabelVector = []

        index = 0
        for line in array_lines:
            # 去掉换行符
            line = line.strip('\n')
            temp = line.split(',')
            # 完成转换
            temp_tran = val_tranform(temp)
            print(temp_tran)

            # phone_gray_score 6，register_org_cnt 7，searched_org_cnt 8 behavior_check_score 9
            # contact_loan 14   call_count 18  called_count 19    做一个简单的数据分析
            data_matr, classLabelVector = val_filter(temp_tran,data_matr,classLabelVector,index)

            index += 1


    return data_matr,classLabelVector

# 特征数据转换补全
def val_tranform(temp):
    # 8-15位置的数据特征是文本数据，需要转换为数值
    contact_each_other = temp[10]
    tran_val = contact_each_other_trans(contact_each_other, 1)
    if tran_val == -1:
        tran_val = 2
    temp[10] = tran_val

    contact_110 = temp[11]
    tran_val = contact_each_other_trans(contact_110, 2)
    if tran_val == -1:
        tran_val = 0
    temp[11] = tran_val
    contact_lawyer = temp[12]
    tran_val = contact_each_other_trans(contact_lawyer, 2)
    if tran_val == -1:
        tran_val = 0
    temp[12] = tran_val
    contact_court = temp[13]
    tran_val = contact_each_other_trans(contact_court, 2)
    if tran_val == -1:
        tran_val = 0
    temp[13] = tran_val

    contact_loan = temp[14]
    tran_val = contact_each_other_trans(contact_loan, 3)
    if tran_val == -1:
        tran_val = 1
    temp[14] = tran_val
    contact_bank = temp[15]
    tran_val = contact_each_other_trans(contact_bank, 3)
    if tran_val == -1:
        tran_val = 1
    temp[15] = tran_val
    contact_credit_card = temp[16]
    tran_val = contact_each_other_trans(contact_credit_card, 3)
    if tran_val == -1:
        tran_val = 1
    temp[16] = tran_val

    contact_night = temp[17]
    tran_val = contact_each_other_trans(contact_night, 4)
    if tran_val == -1:
        tran_val = 1
    temp[17] = tran_val

    return temp


# 对缺失类数据进行补充，原则是取数量最多的数据作为缺失数据的值， parm:-1  return:整数
# contact_each_other_trans '1':96  '2':1188  '3':140     lost_tran:2
# contact_110   0:1186 2:113 3:14                        lost_tran:0
# contact_lawyer 1301 12 0                                lost_tran:0
# contact_court  1312 1 0                                lost_tran:0

# contact_loan   1:793 2:141 3:88                         lost_tran:1
# contact_bank   721 261 148                              lost_tran:1
# contact_credit_card 748 212 119                         lost_tran:1
# contact_night  1270 42 1                                lost_tran:1
def lost_value_tran(val,num):
    pass

# phone_gray_score 6，register_org_cnt 7，searched_org_cnt 8 behavior_check_score 9
# contact_loan 14   call_count 18  called_count 19    做一个简单的数据分析
def val_filter(temp_tran,data_matr,classLabelVector,index):
    data_matr[index, 0] = int(temp_tran[6])
    data_matr[index, 1] = int(temp_tran[7])
    data_matr[index, 2] = int(temp_tran[8])
    data_matr[index, 3] = int(temp_tran[9])
    data_matr[index, 4] = int(temp_tran[14])
    data_matr[index, 5] = int(temp_tran[18])
    data_matr[index, 6] = int(temp_tran[19])
    result = temp_tran[21]
    if result == 'N':
        classLabelVector.append(1)
    elif result == 'Y':
        classLabelVector.append(5)

    print(temp_tran[6], temp_tran[7], temp_tran[8], temp_tran[9], temp_tran[14], temp_tran[18], temp_tran[19],temp_tran[21])

    return data_matr, classLabelVector


# 对文本类数据进行数据转换 ,parm:文本数据  return:整数
def contact_each_other_trans(val,num):
    val = val.rstrip()
    tran_val = 0
    contact_each_other_list = {'数量正常（10 - 100）':2,'数量正常(10-100)':2,'数量众多（100以上，不含100）':3,'数量众多(100以上，不含100)':3,
                    '数量稀少(10以内，不含10)':1,'数量稀少（10以内，不含10）':1,
                               '0':-1}

    contact_lay_list = {'无通话记录':0,'偶尔通话（三次以内，包括三次）':2,'多次通话（三次以上）':3,
                        '0':-1}

    contact_loan_list = {'很少被联系（有该号码记录，且不符合上述情况）':1,
                         '偶尔被联系（联系次数在5次以上，包含5次，且主动呼叫占比 20% - 50%之间，包含20%）':2,
                         '0':-1,
                         '无该类号码记录':0,'经常被联系（联系次数在5次以上，包含5次，且主动呼叫占比大于50%，包含50%）':3}

    contact_night_list = {'0':-1,'很少夜间活动（低于20%)':1,'偶尔夜间活动（20% - 50%， 包含20%）':2,'频繁夜间活动（夜间通话比例大于50%，包含50%）':3}

    if num == 1:
        if contact_each_other_list.get(val)!=None:
            tran_val = contact_each_other_list.get(val)
        else:
            print('no value')
    elif num == 2:
        if contact_lay_list.get(val)!=None:
            tran_val = contact_lay_list.get(val)
        else:
            print('no value')
    elif num == 3:
        if contact_loan_list.get(val)!=None:
            tran_val = contact_loan_list.get(val)
        else:
            print('no value')
    elif num == 4:
        if contact_night_list.get(val)!=None:
            tran_val = contact_night_list.get(val)
        else:
            print('no value')

    return tran_val

# phone_gray_score 0，register_org_cnt 1，searched_org_cnt 2 behavior_check_score 3
            # contact_loan 4   call_count 5  called_count 6    做一个简单的数据分析
def drowMap(matr,classVector):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 设置X轴标签
    plt.xlabel('phone gray score')
    # 设置Y轴标签
    plt.ylabel('register_org_cnt')

    # 这里将喜欢不喜欢这一向量数据映射到了点的颜色上，用来区分不同情况
    # 最大的圈圈是黄色来表示的，
    # phone_gray_score 0，register_org_cnt 1，searched_org_cnt 2 behavior_check_score 3
    # contact_loan 4   call_count 5  called_count 6    做一个简单的数据分析
    # ax.scatter(matr[:, 0], matr[:, 1], 20.0 * array(classVector), 20.0 * array(classVector))
    # ax.scatter(matr[:, 0], matr[:, 6], 20.0 * array(classVector), 20.0 * array(classVector))
    ax.scatter(matr[:, 0], matr[:, 6], 8.0 * array(classVector), 8.0 * array(classVector))

    plt.show()

#  画出3D图像
def draw3DMap(matr,classVector):
    fig = plt.figure()
    ax = Axes3D(fig)
    #  分别设置三个坐标轴代表的数据，把数据的类别 1 2 3 映射到散点数据点的大小size，和 数据点的颜色
    # phone_gray_score 0，register_org_cnt 1，searched_org_cnt 2 behavior_check_score 3
    # contact_loan 4   call_count 5  called_count 6

    # plt.xlim()
    # ax.scatter(matr[:, 0], matr[:, 1], matr[:, 4], matr[:, 2], 8.0 * array(classVector), 8.0 * array(classVector), depthshade=True)
    # plt.ylim(0, 50)
    # ax.scatter(matr[:, 0], matr[:, 1], matr[:, 5], matr[:, 2], 10.0 * array(classVector), 10.0 * array(classVector), depthshade=True)
    # ax.scatter(matr[:, 0], matr[:, 5], matr[:, 6], matr[:, 2], 10.0 * array(classVector), 10.0 * array(classVector), depthshade=True)
    # ax.scatter(matr[:, 4], matr[:, 5], matr[:, 6], matr[:, 2], 15.0 * array(classVector), 15.0 * array(classVector), depthshade=True)

    plt.xlim(0,50);plt.ylim(0,50)
    ax.scatter(matr[:, 1], matr[:, 5], matr[:, 6], matr[:, 2], 15.0 * array(classVector), 15.0 * array(classVector), depthshade=True)

    plt.show()

# 根据特征画出折线图
def drawLineMap(matr):
    x = linspace(0, len(matr), len(matr))
    # 绘制y=2x+1函数的图像
    y1 = matr[:, 0]
    y2 = matr[:, 1]
    plt.title('Result Analysis')
    plt.plot(x, y1, color='green', label='order')
    plt.plot(x, y2, color='skyblue', label='repay')

    plt.xlabel('date ')
    plt.ylabel('data different')
    plt.show()
    # python 一个折线图绘制多个曲线

if __name__ == '__main__':
    carr_path = 'D:/Develop/test/carrier/carr_data/'
    risk_path = 'D:/Develop/test/carrier/risk_score/risk_data_new.txt'
    report_path = 'D:/Develop/test/carrier/report_var/report_add_call_val.txt'

    # 提取特征 到文件
    # list = read_risk_data(risk_path)
    # print(list)
    # loop_file(carr_path,list)

    # 分析
    data_matr, classLabelVector = data_handler(report_path)
    # # 画出散点图
    print(data_matr)
    print(classLabelVector)
    # drowMap(data_matr, classLabelVector)
    #
    draw3DMap(data_matr, classLabelVector)

