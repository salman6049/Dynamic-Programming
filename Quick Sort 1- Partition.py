# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:12:14 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the quickSort function below.
def quickSort(arr):
    if len(arr)<2:
        return arr
    else:
        pivot=arr[0]
        left=[i for i in arr[1:] if i<=pivot]
        right=[i for i in arr[1:] if i>pivot]
        return quickSort(left)+[pivot]+quickSort(right)
        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = quickSort(arr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
