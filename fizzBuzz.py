# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:46:23 2019

@author: 20786136
"""

def fizzBuzz(n):
    """
    :type n: int
    :rtype: List[str]
    """
    List = []
    for i in range(1,n+1):
        if i % 3 ==0 and i% 5 ==0:
            List.append('FizBuzz')
        elif i % 3 == 0:
            List.append('Fizz')
        elif i%5 == 0:
            List.append('Buzz')
        else:
            List.append(i)
    return List    
fizzBuzz(15)
    