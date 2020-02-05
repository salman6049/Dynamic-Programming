# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:12:28 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys
from functools import reduce
from operator import and_

def gemstones(arr):
    return len(reduce(and_,map(set,arr)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = []

    for _ in range(n):
        arr_item = input()
        arr.append(arr_item)

    result = gemstones(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
