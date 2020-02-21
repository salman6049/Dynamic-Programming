# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:41:21 2020

@author: sigar
"""

#!/bin/python

import math
import os
import random
import re
import sys

#
# Complete the 'solve' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER h
#  2. INTEGER_ARRAY wallPoints
#  3. INTEGER_ARRAY lengths
#

def solve(h, wallPoints, lengths):
    qw=[]
    for i in range(len(wallPoints)):
        qw.append(wallPoints[i]- int(lengths[i]/4)-h)
    ans=max(qw)
    if ans<0:
        return 0
    else:
        return ans
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = raw_input().rstrip().split()

    n = int(first_multiple_input[0])

    h = int(first_multiple_input[1])

    wallPoints = map(int, raw_input().rstrip().split())

    lengths = map(int, raw_input().rstrip().split())

    answer = solve(h, wallPoints, lengths)

    fptr.write(str(answer) + '\n')

    fptr.close()
