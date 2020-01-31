# -*- coding: utf-8 -*-
"""
Created on March 2019

@author: Salman
"""
#!/bin/python

import math
import os
import random
import re
import sys

def passwordCracker(passwords, loginAttempt):
    # Write your code here
    MEMOIZED_RECORD = set()
    pd = {p:len(p) for p in passwords}        

    def recurPasswordCracker(pd, s, solution):               
        if s in pd:                                    
            solution.append(s)
            return solution
        if s in MEMOIZED_RECORD:
            return False
        if len(s) == 1:
            return False
        for k in pd:
            if s[:pd[k]] == k:                                                             
                MEMOIZED_RECORD.add(s)
                loopSolution = recurPasswordCracker(pd, s[pd[k]:], solution + [k])
                if loopSolution:
                    return loopSolution
        return False        

    result = recurPasswordCracker(pd, loginAttempt, [])    
    if not result:
        return 'WRONG PASSWORD'
    else:
        return ' '.join(p for p in result)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(raw_input().strip())

    for t_itr in xrange(t):
        n = int(raw_input().strip())

        passwords = raw_input().rstrip().split()

        loginAttempt = raw_input()

        result = passwordCracker(passwords, loginAttempt)

        fptr.write(result + '\n')

    fptr.close()
