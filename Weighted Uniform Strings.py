# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 08:12:31 2020

@author: Salman
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the weightedUniformStrings function below.
s = input().strip()
n = int(input().strip())
seq=[]
for num in range(1,27):
    seq.append(0)
for x in sorted(set(s)):
    i = 1;
    while x * i in s:
        i += 1
    seq[ord(x)-97]=i-1 
finale=set()                 #using set
for index,every in enumerate(seq):
    for sval in range(every):
        finale.add((index+1)*(sval+1))     #using set
for a0 in range(n):
    x = int(input().strip())
    print("Yes" if x in finale else "No")