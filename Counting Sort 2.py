# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:17:51 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the countingSort function below.
def countingSort(arr):
    l = [0]*100
    for i in arr:
        l[i]+=1
    j = 0
    for i in range(min(arr),max(arr)+1):
        n = l[i]
        while(n>0):
            arr[j] = i
            j += 1
            n -= 1
    return arr
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = countingSort(arr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
