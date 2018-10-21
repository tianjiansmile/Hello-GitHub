# -*- coding: UTF-8 -*-
# from numpy import *
# import matplotlib.pyplot as plt
import json

# 主要用于提取用户变量
class variable_person:

    def __init__(self,filepath,filename):
        with open(filepath+filename, 'r',encoding = 'utf-8') as load_f:
            self.json_data = json.load(load_f)


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

    #  联系电话灰名单得分
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

        elif data_type == 'dict':
            check_black_info = user_info_check.get('check_black_info')
            if check_black_info != None:
                phone_gray_score = check_black_info.get('phone_gray_score')


        self.varls.append(phone_gray_score)

    def loan_evidence_compute(self,evidence):
        if evidence != None:
            pass
            # print(evidence)

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
                    self.loan_evidence_compute(evidence)
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


    def type_check(self,data):
        if isinstance(data,dict):
            return 'dict'
        elif isinstance(data,list):
            return 'list'
        elif isinstance(data,str):
            return 'str'
        elif isinstance(data,int):
            return 'int'
        else:
            return 'undefined'

if __name__ == '__main__':
    path = 'D:/spark/report/'
    order_list = ['13659439086195589125','13659439145922478087','13659439154646630417','13659439166726225923','13659439169410580480',
                  '13659439215715696654','13659439221084405760','13659439224439848964','13659439231150735361','13659439241217064963','13659439252625571847'
                  ,'13659439257323192324','13659439263362990080','13659439271416053764','13659439272087142402','13659439274100408341',
                  '13659439274771496969','13659439290877624328','13659439292219801600','13659439298259599375','13659439299601776647',
                  '13659439300943953936','13659439302957219859','13659439308997017616','13659439318392258569','13659439358657576965',
                  '13659439421739909132','13659439443214745604','13659439455294341121','13659439455294341139','13659439475427000322',
                  '13659439512336875529','13659439561326346244','13659439566695055367']
    for order in order_list:
        judge = variable_person(path,order+'.txt')
        judge.setBasicDate()

        print(order,judge.varls)
        # print(len(order_list))