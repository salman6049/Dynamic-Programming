# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:07:07 2019

@author: 20786136
"""

class Solution(object):
    def maxProfit(self, prices):
        n = len(prices)
        if n <= 1:
            return 0
        forward_profit,backward_profit = [0]*n,[0]*n
        buy = prices[0]
        for i in range(1,n):
            buy = min(prices[i],buy)
            forward_profit[i] = max(forward_profit[i-1],prices[i]-buy)
            
        sell = prices[-1]    
        for i in range(n-2,-1,-1):
            sell = max(sell,prices[i])
            backward_profit[i] = max(backward_profit[i+1],sell-prices[i]) 
        maxprofit = 0    
        for i in range(n):
            maxprofit = max(maxprofit,forward_profit[i]+backward_profit[i])
            
        return maxprofit
        