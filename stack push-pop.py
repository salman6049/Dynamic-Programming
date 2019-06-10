# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:36:51 2018

@author: sigar
"""

import numpy as np

X = np.array([84.04467948,60.,52.42447842,39.13555678])
Y = np.array([78.86529444,52.42447842,38.74910101,21.99846595])

dist = np.abs(X[:, np.newaxis] - Y)

#############################################3
class SpecialStack(object):

    def createStack(self):
        stack = []
        return stack

    def pushElement(self, stack , auxiallaryStack, element):


        if len(auxiallaryStack) == 0 :
            auxiallaryStack.append(element)
        else:
            if element < auxiallaryStack[-1] and len(auxiallaryStack) != 0:
                auxiallaryStack.append(element)
            elif element > auxiallaryStack[-1] and len(auxiallaryStack) != 0:
                auxiallaryStack.append(auxiallaryStack[-1])
            else:
                auxiallaryStack.append(element)
        stack.append(element)

    def popElement(self,stack, auxillarStack):
        if not stack:
            print ("Stack underflow , not able to pop")
        else:
            auxillarStack.pop
            return stack.pop

    def getMin(self, stack , auxillaryStack):
        return auxillaryStack[-1]

    def printStackElements(self, stack):
        for index in range(len(stack)-1 , -1 , -1):
            print (stack[index])

    def implementor(self):
        stack = self.createStack()
        auxiallyStack = self.createStack()

        self.pushElement(stack, auxiallyStack ,18)
        self.pushElement(stack, auxiallyStack ,19)
        self.pushElement(stack, auxiallyStack ,29)
        self.pushElement(stack, auxiallyStack ,15)
        self.pushElement(stack, auxiallyStack, 16)

        self.printStackElements(stack)
        print( "\n")

        print ("Minimum is " ,self.getMin(stack, auxiallyStack))

if __name__=="__main__":
    SpecialStack().implementor()