# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:33:13 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the toys function below.
def toys(w):
    w = {weight: True for weight in w}
    containers = 0
    max_w = max(w)
    i = 0
    while i <= max_w:
        if w.get(i):
            containers += 1
            i += 4
        i += 1
    return containers

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    w = list(map(int, input().rstrip().split()))

    result = toys(w)

    fptr.write(str(result) + '\n')

    fptr.close()
