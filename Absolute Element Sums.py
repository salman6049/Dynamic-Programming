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

# Complete the playingWithNumbers function below.
def playingWithNumbers(arr, queries):
    LEN = 4003  # 2000 - (-2000) + 1 + 2
    LAST_INDEX = LEN-1
    SHIFT = 2001  # 2000 + 1
    counts = [0]*LEN
    for a in arr:
        new_a = a+SHIFT
        counts[new_a] += 1

    incremental_counts = [0]*len(counts)
    incremental_sum = [0]*len(counts)
    incremental_sumn = [0]*len(counts)  # summing negatives
    for i in range(len(counts)):  
        # here is a hack:
        # Incremental_counts[i-1] == incremental_counts[-1] when i == 0,
        # but last element of incremental_counts is initialized with 0
        # It's exactly what we need, so we do not need 'if'-check for border condition
        incremental_counts[i] = counts[i] + incremental_counts[i-1]
        incremental_sum[i] = counts[i]*i + incremental_sum[i-1]
        incremental_sumn[i] = counts[i]*(LAST_INDEX-i) + incremental_sumn[i-1]

    results = []
    result_cache = {}
    tot_q = 0
    for q in queries:
        tot_q += q
        if tot_q not in result_cache:
            shift = SHIFT-tot_q
            index = max(min(SHIFT-tot_q, LAST_INDEX), 0)
            result = (
                (
                    incremental_sum[-1] - incremental_sum[index]
                    - shift*(incremental_counts[-1]-incremental_counts[index])  
                )
                + (
                    incremental_sumn[index]
                    - (LAST_INDEX-shift)*(incremental_counts[index])             
                )
            )
            result_cache[tot_q] = result
        results.append(result_cache[tot_q])
    return results
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    q = int(input())

    queries = list(map(int, input().rstrip().split()))

    result = playingWithNumbers(arr, queries)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
