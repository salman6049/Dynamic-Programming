# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:13:16 2019

@author: sigar
"""
# o(nlon)
def merge(l, m):
    result = []
    i = j = 0
    total = len(l) + len(m)
    while len(result) != total:
        if len(l) == i:
            result += m[j:]
            break
        elif len(m) == j:
            result += l[i:]
            break
        elif l[i] < m[j]:
            result.append(l[i])
            i += 1
        else:
            result.append(m[j])
            j += 1
    return result

import random

def merge(x,y):
    merged_list =[]
    i =j =0
    Total = len(x) + len(y)
    while len(merged_list) != Total:
        if len(x) == i:
            merged_list += y[j:]
            break
        elif len(y) == j:
            merged_list += x[i:]
            break
        elif x[i] < y[j] :
            merged_list.append(x[i])
            i += 1
        else:
            merged_list.append(y[j])
            j +=1
    return merged_list

a = [5,1,6,3]
b = [0,-1,11,9]

merge(a,b)