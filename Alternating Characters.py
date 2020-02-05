# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:14:15 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the alternatingCharacters function below.
def alternatingCharacters(s):
        return len(s) - len(re.sub('BB+', 'B', re.sub('AA+', 'A', s)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = alternatingCharacters(s)

        fptr.write(str(result) + '\n')

    fptr.close()
