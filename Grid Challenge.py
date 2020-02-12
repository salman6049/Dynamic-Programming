# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:25:01 2020

@author: sigar
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the gridChallenge function below.

def gridChallenge(grid):
    sorted_grid  = sorted(grid[0])
    for a in grid:
        a_list = sorted(a)
        for i, c in enumerate(a_list):
            if c < sorted_grid[i]:
                return "NO"
        sorted_grid = a_list
    return 'YES'
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        n = int(input())

        grid = []

        for _ in range(n):
            grid_item = input()
            grid.append(grid_item)

        result = gridChallenge(grid)

        fptr.write(result + '\n')

    fptr.close()
