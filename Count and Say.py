# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:37:44 2019

@author: 20786136
"""

class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n == 1:
            return "1"
        
        pre = self.countAndSay(n-1)
        count = 1
        ans = []
        for i in range(len(pre)-1):
            if pre[i] == pre[i+1]:
                count += 1
            else:
                ans.append(str(count))
                ans.append(pre[i])
                count = 1
        ans.append(str(count))
        ans.append(pre[len(pre)-1])

        s = ''.join(ans)
        return s