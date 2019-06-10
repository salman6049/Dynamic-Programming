# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:12:02 2019

@author: 20786136
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