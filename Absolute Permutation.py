# -*- coding: utf-8 -*-
"""
Created on April 2018

@author: Salman
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the absolutePermutation function below.
def absolutePermutation(n, k):
    if k == 0:
        return [i+1 for i in range(n)]
    elif n % (2*k) != 0 or 2*k > n: 
        return [-1]
    return [(i+1)+(1 if (i//k)%2==0 else -1)*k for i in range(n)]
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        nk = input().split()

        n = int(nk[0])

        k = int(nk[1])

        result = absolutePermutation(n, k)

        fptr.write(' '.join(map(str, result)))
        fptr.write('\n')

    fptr.close()
