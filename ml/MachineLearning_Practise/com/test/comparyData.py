# coding: utf-8

orderlist = []
ordercount = 0
panbaocount = 0
diffcount = 0

# 读入order数据
def readOrder(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            temp = line.split(',')
            uid = temp[0]
            pid = temp[1]
            orderlist.append((temp[0], temp[1]))
            global ordercount
            ordercount = ordercount + 1

        # print uid,pid
def readPanbao(path,finalfile):
    with open(path, 'r') as f:
        for line in f.readlines():
            temp = line.split(',')
            uid = temp[0]
            pid = temp[1]
            internaluid = temp[2]
            internalpid = temp[3]

            global panbaocount
            panbaocount = panbaocount + 1

            match = (uid,pid)
            # print match,uid,pid,internaluid,internalpid

            if match not in orderlist:
                print( internaluid+', '+internalpid)
                global diffcount
                diffcount = diffcount + 1
                finalfile.write(internaluid+', '+internalpid)

readOrder('D:/spark/order.txt')

txtName = "D:/spark/neededData.txt"
finalfile = open(txtName, "a+")

readPanbao('D:/spark/panbao.txt',finalfile)

finalfile.close()

print (ordercount,diffcount,panbaocount)