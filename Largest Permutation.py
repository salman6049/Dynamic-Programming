# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:35:29 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the largestPermutation function below.
def largestPermutation(k, arr):
    compare = sorted(arr,reverse = True)
    if arr == compare:
        return arr
    lookup = {n:i for i,n in enumerate(arr)}
    i = 0
    while i < len(arr) and k:
        if arr[i] == compare[i]:
            i += 1
            continue
        get, largeidx = arr[i], lookup[compare[i]] 
        arr[i],arr[largeidx],lookup[get], lookup[compare[i]] = compare[i], get,largeidx,i
        if arr == compare:
            return arr
        k -= 1
        i += 1
    return arr
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    arr = list(map(int, input().rstrip().split()))

    result = largestPermutation(k, arr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
