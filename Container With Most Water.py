# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:12:04 2019

@author: 20786136
"""


class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0 
        right = len(height) - 1
        area = -1
        while left < right:
            area = max(area, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return area