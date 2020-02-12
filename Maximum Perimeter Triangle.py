# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:30:57 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the maximumPerimeterTriangle function below.
def maximumPerimeterTriangle(sticks):
    sticks.sort(reverse=True)
    for a, b, c in zip(sticks, sticks[1:], sticks[2:]):
        if c + b > a:
            return c, b, a
    return (-1,)
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    sticks = list(map(int, input().rstrip().split()))

    result = maximumPerimeterTriangle(sticks)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
