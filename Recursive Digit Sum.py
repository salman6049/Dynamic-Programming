# -*- coding: utf-8 -*-
"""
Created on October 2017

@author: Salman
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def digitSum(n, k):
    x = sum(map(int, list(n)))
    if k==1:
        return ds(x)
    else:
        return ds(x*k)
    
def ds(x):
    if x<10:
        return x
    else:
        return ds(sum(map(int, list(str(x)))))

if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [str(n), int(k)]
    result = digitSum(n, k)
    print(result)
