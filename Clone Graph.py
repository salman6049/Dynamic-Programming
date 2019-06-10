# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:52:00 2019

@author: 20786136
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':

        cp = collections.defaultdict(lambda: Node(0, []))
        nodes = [node]
        seen = set()
        while nodes:
            n = nodes.pop()
            cp[n].val = n.val
            cp[n].neighbors = [cp[x] for x in n.neighbors]
            nodes.extend(x for x in n.neighbors if x not in seen)
            seen.add(n)
        return cp[node]