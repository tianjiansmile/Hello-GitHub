#!-*- coding:utf8-*-
from com.risk_score.feature_extact import setting
import pymongo

client = pymongo.MongoClient(setting.host)
db = client.call_test
coll = db.resource_info


def mongo_read():
    results = coll.find({},{'_id': 0}).limit(5000)
    device_dict = {}
    ip_dict = {}
    wifi_dict = {}
    imei_dict = {}
    for d in results:
        if d:
            for a in d:
                print(a)
                resource = d[a]
                deviceId = resource.get('deviceId')
                for d in deviceId:
                    check = device_dict.get(d)
                    if check == None:
                        device_dict[d] = 1
                    else:
                        device_dict[d] += 1

                ip = resource.get('ip')
                for d in ip:
                    check = ip_dict.get(d)
                    if check == None:
                        ip_dict[d] = 1
                    else:
                        ip_dict[d] += 1

                wifiIp = resource.get('wifiIp')
                for d in wifiIp:
                    check = wifi_dict.get(d)
                    if check == None:
                        wifi_dict[d] = 1
                    else:
                        wifi_dict[d] += 1

                imei = resource.get('imei')
                for d in imei:
                    check = imei_dict.get(d)
                    if check == None:
                        imei_dict[d] = 1
                    else:
                        imei_dict[d] += 1

                # print(d[a])

    # print(device_dict)
    # print(sorted(device_dict.items(), key=operator.itemgetter(1), reverse=True))
    # print(imei_dict)
    # print(sorted(imei_dict.items(), key=operator.itemgetter(1), reverse=True))

    # print(sorted(wifi_dict.items(), key=operator.itemgetter(1), reverse=True))

    print(sorted(ip_dict.items(), key=operator.itemgetter(1), reverse=True))


if __name__ == '__main__':
    import operator

    # 产看用户资源是否有共用关系，从结果看是确实存在的
    mongo_read()