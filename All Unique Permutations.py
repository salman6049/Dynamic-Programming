# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:50:17 2019

@author: 20786136
"""

class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(N, r, out):
            if not N:
                out += [r]
            else:
                visited = []
                for i, n in enumerate(N):
                    if n not in visited:
                        visited += [n]
                        dfs(N[:i] + N[i+1:], r + [n], out)
        out = []
        dfs(nums, [], out)
        return out