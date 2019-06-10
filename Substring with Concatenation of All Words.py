# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:10:11 2019

@author: 20786136
"""

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        
        import collections
        if not words:
            return []
        ret = []
        word_len = len(words[0])
        sentence_len = word_len * len(words)
        s_len = len(s)
        dic = collections.Counter(words)

        for i in range(word_len):
            d = collections.Counter()
            left, right = i, i
            while right < s_len - word_len + 1:
                word = s[right: right + word_len]
                right += word_len
                if word not in dic:
                    d.clear()
                    left = right
                    continue
                d[word] += 1
                while d[word] > dic[word]:
                    d[s[left: left + word_len]] -= 1
                    left += word_len

                if left + sentence_len == right:
                    ret.append(left)
        return ret