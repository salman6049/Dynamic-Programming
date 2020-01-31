# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:07:59 2020

@author: sigar
"""

#!/bin/python3

import os
import sys

#
# Complete the ashtonString function below.
def subString(s, n): 
    l = []
    for i in range(n): 
        for len in range(i+1,n+1): 
            l.append(s[i: len]); 
            #print(l)
    return l

def ashtonString(s, k):
    mylist = subString(s,len(s)); 
    mylist = list(dict.fromkeys(mylist))
    mylist.sort()
    s = ""
    s = s.join(mylist);
    #print(s)
    return s[k-1]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        s = input()

        k = int(input())

        res = ashtonString(s, k)

        fptr.write(str(res) + '\n')

    fptr.close()
