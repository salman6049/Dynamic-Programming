# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:20:54 2019

@author: 20786136
"""

"""Solution
Traverse both lists. One by one pick nodes of both lists and add the values. 
If sum is more than 10 then make carry as 1 and reduce sum. 
If one list has more elements than the other then consider remaining values of this list as 0. 
Following is the implementation of this approach.
"""


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        u_place=0
        num1=0;num2=0
        while l1!=None :
            num1+=l1.val*10**(u_place)
            l1=l1.next
            u_place+=1
        u_place=0
        while l2!=None:
            num2+=l2.val*10**(u_place)
            l2=l2.next
            u_place+=1
        res=num1+num2
        d=res
        dummyRoot = ListNode(0)
        ptr = dummyRoot
        if d==0:
            return ptr
        while d!=0:
            rem=int(d%10)
            ptr.next = ListNode(rem)
            ptr=ptr.next
            d=d//10
        ptr=dummyRoot.next
        return ptr