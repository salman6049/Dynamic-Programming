# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:08:04 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the separateNumbers function below.
def separateNumbers(s):
    # Complete this function
    if s[0] == s:
        print('NO')
        return
    for i in range(1, len(s)):
        mystack = []
        mystack.append(s[:i])
        while len(''.join(mystack)) < len(s):
            mystack.append(str(int(mystack[-1]) + 1))
        if ''.join(mystack) == s:
            print('YES', mystack[0])
            break
        if i == len(s) - 1:
            print('NO')

if __name__ == "__main__":
    q = int(input().strip())
    for a0 in range(q):
        s = input().strip()
        separateNumbers(s)