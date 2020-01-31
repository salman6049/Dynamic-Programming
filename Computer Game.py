# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: Salman
"""

def computerGame(a,b):
    n=0
    for i in range(2,min(a,b)+1):
        if b%i==0 and a%i==0:
            n+=1
    if n==0:
        return 1
    else:
        return 0
n=int(input())
a=[int(i) for i in input().split()]
b=[int(i) for i in input().split()]
p=0
for i in a:
    for j in b:
        if computerGame(i,j)==0:
            p+=1
            break
print(p)