# -*- coding: utf-8 -*-
"""
Created on Feb 2018

@author: Salman
"""

# function to print triplets with 0 sum  
def findTriplets(arr, n): 
    found = False
    for i in range(n - 1): 
  
        # Find all pairs with sum  
        # equals to "-arr[i]"  
        s = set() 
        for j in range(i + 1, n): 
            x = -(arr[i] + arr[j]) 
            if x in s: 
                print(x, arr[i], arr[j]) 
                found = True
            else: 
                s.add(arr[j]) 
    if found == False: 
        print("No Triplet Found") 