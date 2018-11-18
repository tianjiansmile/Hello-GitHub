# -*- coding: UTF-8 -*-
import pymysql

# 数据类型检查
def type_check(item):
    data_type = 'list'
    if type(item).__name__ == 'list':
        data_type = 'list'
    elif type(item).__name__ == 'dict':
        data_type = 'dict'
    elif type(item).__name__ == 'str':
        data_type = 'str'
    elif type(item).__name__ == 'int':
        data_type = 'int'
    elif type(item).__name__ == 'class':
        data_type = 'class'
    else:
        data_type = 'undefined'

    return data_type
