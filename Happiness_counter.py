# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:38:39 2017

@author: sigar
"""

def happiness_counter():
    n = int(input('n:'))
    m = int(input('m:'))
    print(n)
    print(m)
    happiness = 0

    A = set()
    B = set()
    C = []
    C = input().strip().split(' ')
#    while len(C) <n:
#        temp =int(input('value C:'))
#        C.append(temp)

    print(C)
    A = int(input().strip().split(' ')) 
    A = set(A)
#    while len(A) <m:
#        temp =int(input('value A:'))
#        A.add(temp)
    #print(A)
    B = int(input().strip().split(' '))
    B = set(B)
#    while len(B) <m:
#        temp =int(input('value B:'))
#        B.add(temp)
    #print(B)

           
    
    for i in C:
        if i in A:
            happiness +=1    
        if i in B:
            happiness -= 1
    print (happiness)

happiness_counter()

