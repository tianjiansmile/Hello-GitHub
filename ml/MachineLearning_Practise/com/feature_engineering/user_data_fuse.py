
import json
import requests
import re

# 主要实现对用户的所有订单数据和征信数据做融合处理---- 数据融合---- 用户纯用户同构网络
class Person:
    def __init__(self,data):
        self.jsondata = data

    # 初始化各个条目数据
    def set_basicNode(self):
        # get() 方法优于 jsondata[] 其默认key可以不存在，并且查询速度更快
        self.name = self.jsondata.get('name')
        self.idAddr = self.jsondata.get('idAddr')
        self.idNum = self.jsondata.get('idNum')
        self.birthday = self.jsondata.get('birthday')

        self.phone_d = []
        self.addresss_d = []
        self.order_d = []
        self.bankCards_d = []
        self.scores_d = []  ##
        self.contacts_d = []
        self.callRecords_d = []
        self.smses_d = []
        self.company_d = []
        self.devices_d = []
        self.apps_d = []
        self.repayments_d = []
        self.carriers_d = []

        self.set_orderNode(self.jsondata.get('orders'))


    # 过滤此用户所有基础数据，将订单对应的征信信息单独领出来
    def set_orderNode(self,orders):
        if orders != None:
            for ord in orders:
                phones = ord.get('phones')
                self.phone_d.append(phones)
                addresss = ord.get('addresss')
                self.addresss_d.append(addresss)
                order = ord.get('order')
                self.order_d.append(order)
                bankCards = ord.get('bankCards')
                self.bankCards_d.append(bankCards)
                scores = ord.get('scores')
                self.scores_d.append(scores)
                contacts = ord.get('contacts')
                self.contacts_d.append(contacts)
                callRecords = ord.get('callRecords')
                self.callRecords_d.append(callRecords)
                smses = ord.get('smses')
                self.smses_d.append(smses)
                company = ord.get('company')
                self.company_d.append(company)
                devices = ord.get('devices')
                self.devices_d.append(devices)
                apps = ord.get('apps')
                self.apps_d.append(apps)
                repayments = ord.get('repayments')
                self.repayments_d.append(repayments)

                carriers = ord.get('carrier')
                self.carriers_d.append(carriers)

    # 个人电话提取
    def phone_extact(self):
        phones = set()
        [phones.add(p.get('phoneNum')) for p in self.phone_d if p.get('phoneNum') !=None]
        self.phones = phones
        print('phone:',phones)

    # 个人地址提取
    def address_extact(self):
        l_address = set()
        w_address = set()
        g_address = set()
        for add in self.addresss_d:
            liveAddr = add.get('liveAddr')
            workAddr = add.get('workAddr')
            gpsAddr = add.get('gpsAddr')
            if liveAddr!=None and liveAddr.get('addrDetail') !=None:
                l_address.add(liveAddr.get('addrDetail'))
            if workAddr !=None and workAddr.get('addrDetail') !=None:
                w_address.add(workAddr.get('addrDetail'))
            if gpsAddr != None and gpsAddr.get('addrDetail') !=None:
                g_address.add(gpsAddr.get('addrDetail'))

        self.l_address = l_address
        self.w_address = w_address
        self.g_address = g_address
        print(l_address,w_address,g_address)


    # 个人订单特征提取
    def order_extact(self):
        apply_times = len(self.order_d)
        loan_times = 0
        for ord in self.order_d:
            loanStatus = ord.get('loanStatus')
            if loanStatus == 'Y':
                loan_times+=1
            modelScore = ord.get('modelScore')

        self.apply_times = apply_times
        self.loan_times = loan_times
        print('order: ',apply_times,loan_times)


    # 个人银行卡提取
    def bankcard_extact(self):
        card_num = set()
        for card in self.bankCards_d:
            if card != None:
                [card_num.add(c.get('cardNum')) for c in card if c.get('cardNum')]

        print('bank card',card_num)
        self.card_num = card_num

    # 通讯录紧急联系人提取
    def contacts_extact(self):
        try:
            # 提纯所有紧急联系人号码
            emerge_rel = set()
            # 提纯所有通讯录号码
            contact_set = set()
            for con in self.contacts_d:
                emergencyContacts = con.get('emergencyContacts')
                # print(emergencyContacts)
                if emergencyContacts !=None:
                    for em in emergencyContacts:
                        phone = em.get('phone')

                        # 都是联系人关系
                        relation = em.get('relation')
                        contact = em.get('contact')
                        if phone != None:
                            if relation!=None:
                                emerge_rel.add((relation,phone))
                            if contact!=None:
                                emerge_rel.add((contact, phone))

                            if relation==None and contact==None:
                                emerge_rel.add(('None', phone))


                contacts = con.get('contacts')
                if contacts != None:
                    # [contact_set.add(cont.get('phone').replace(' ',''))
                     for cont in contacts:
                        if cont.get('phone')!=None:
                            ph = cont.get('phone')
                            if type(ph) is str:
                                contact_set.add(cont.get('phone').replace(' ', ''))
                            elif type(ph) is list:
                                contact_set.add(cont.get('phone')[0].replace(' ', ''))
        except Exception as e:
            print(contacts)
            raise TypeError('Exception') from e

        print('emerge_rel: ',emerge_rel)
        print('contact_set',contact_set)
        self.emerge_rel = emerge_rel
        self.contact_set = contact_set

    # 解析运营商原始报告，主要解析通话记录和短信记录，不同渠道的运营商报告结构不一样，
    # 一个用户可能有多个订单，每一个订单都有一个报告，原则上如果用户通话记录获取不到的话
    # 就解析运营商报告来获取，只取一个报告做解析
    def carrier_origin_extact(self):
        pass

    # 通话记录提取
    def callRecords_extact(self):
        try:
            call_history = {}
            for callrec in self.callRecords_d:
                # print(callrec)
                if callrec != None:
                    for call in callrec:
                        ca = call.get('userData')
                        if ca != None and ca.strip() != '':
                            ca = json.loads(ca)
                            for c in ca:
                                p = c.get('phone')
                                # 通话时长
                                use_time = c.get('use_time')
                                # 呼叫类型
                                type = c.get('type')
                                if call_history.get(p) == None:
                                    call_history[p] = [(use_time,type)]
                                else:
                                    call_history[p].append((use_time,type))
            print('call_history: ',call_history)
            self.call_history = call_history

        except Exception as e:
            print('error ',self.callRecords_d)
            raise TypeError('Exception') from e

    # 短信记录提取
    def smses_extact(self):

        for smses in self.smses_d:
            if smses != None:
                print(smses)

    # 公司信息提取
    def company_extact(self):
        companys = set()
        for company in self.company_d:
            if company != None:
                compyName = company.get('compyName')
                compyAddr = company.get('compyAddr')
                companys.add((compyName,compyAddr))
        print('company: ',companys)
        self.companys = companys

    # app 提取
    def apps_extact(self):
        for apps in self.apps_d:
            if apps != None:
                print(apps)

    # 设备信息提取
    def devices_extact(self):
        dev_no = set()
        for dev in self.devices_d:
            if dev != None:
                # print(dev)
                for d in dev:
                    deviceNo = d.get('deviceNo')
                    if deviceNo != None:
                        dev_no.add(deviceNo)

        print('设备信息：',dev_no)
        self.dev_no = dev_no

    # 贷后数据提取
    def repayments_extact(self):
        overdueDays = []
        for repay in self.repayments_d:
            if repay != None:
                for re in repay:
                    overdueDay = re.get('overdueDay')
                    if overdueDay != '0':
                        overdueDays.append(overdueDay)

        print('over due:',overdueDays)
        self.overdueDays = overdueDays

    def set_extact(self):
        self.phone_extact()
        self.address_extact()
        self.bankcard_extact()
        self.order_extact()
        self.contacts_extact()
        self.callRecords_extact()
        self.smses_extact()
        self.company_extact()
        self.devices_extact()
        self.apps_extact()
        self.repayments_extact()

def pull_request(id_list):

    for id in id_list:
        if id !=None and id !='None':
            res = requests.get('http://**:8001/proxy/user/orders?identityNo=%s' % (id))
            json_data = res.json()
            person_fuse(json_data.get('data'))

def get_user_id(file):
    id_list = []
    with open(file, 'r') as rf:
        for line in rf.readlines():
            # 去掉换行符
            id = line.strip('\n')
            id_list.append(id)

    return id_list

# 个人数据融合
def person_fuse(json_data):
    p = Person(json_data)
    p.set_basicNode()
    p.set_extact()

if __name__ == '__main__':
    file = 'D:/Develop/test/neo4j_data/user_id/user_201809_ids.txt'
    id_list = get_user_id(file)
    test = id_list[100:102]
    print(test)
    pull_request(test)