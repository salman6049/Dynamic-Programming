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

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    counter = 1
    while(counter < n):
        j = counter
        while(j > 0 and arr[j-1] > arr[j]):            
            arr[j-1], arr[j] = arr[j], arr[j-1]
            j -= 1
            
        print(*arr)
        counter+=1  

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
