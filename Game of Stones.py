# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:21:22 2020

@author: Salman
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the gameOfStones function below.
n = int(input().strip())
for i in range(n):    
    option = int(input().strip())
    print("Second" if option % 7 in [0, 1] else "First")