#  coding: utf-8

if __name__ == '__main__':
    app_dict = {}
    file = 'D:/特征提取/app列表/appData_list_0.txt'
    with open(file,'r',encoding='UTF-8') as rf:
        count = 0
        lines = rf.readlines()
        for line in lines:
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.split(',')
            for a in apps:
                count +=1
                # print(a)
                check = app_dict.get(a)
                if check is None:
                    app_dict[a] = 1
                else:
                    app_dict[a] += 1

    print(count,len(app_dict))
    print(app_dict)