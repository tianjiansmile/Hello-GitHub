# -*- coding: UTF-8 -*-

import urllib2
import mysql.connector
import bs4
from bs4 import BeautifulSoup
from urlparse import urljoin
import re
from numpy import *


l1 = [[1,2,3],[4,5,6]]
m1 = matrix(l1)

# 生成一个随机矩阵
m2 = random.rand(4,4)

# m1的逆矩阵
invt_m1 = m1.I

my_eye = m1 * invt_m1

# 四阶单位矩阵
eye = eye(2)

print m1,invt_m1

print(my_eye - eye)


