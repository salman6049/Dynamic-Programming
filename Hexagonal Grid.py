# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:38:07 2020

@author: Salman
"""

def isBlocked(n, first,second):
    blocked_cells = sum(first) + sum(second)
    if blocked_cells % 2 == 1:
        return True
    # I DONT KNOW WHY THE TESTS CLAIM THIS IS VALID!
    if blocked_cells == 2*n:
        return False
    sum_so_far = 0
    for i in xrange(0,n):
        sum_so_far += (1 - first[i])
        if i > 0 and first[i] and second[i-1]:
            if sum_so_far % 2 == 1:
                return True
            sum_so_far = 0
        if first[i] and second[i] and sum_so_far % 2 == 1:
            return True
        sum_so_far += (1 - second[i])
    return False

T = int(raw_input().strip())
for t in xrange(T):
    n = int(raw_input().strip())
    first,second = [map(int,list(raw_input().strip())) for _ in xrange(2)]
    if isBlocked(n,first,second):
        print "NO"
    else:
        print "YES"
