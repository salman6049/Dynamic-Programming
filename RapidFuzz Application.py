#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from rapidfuzz import fuzz

def repeat_numbers(s, n):
    return re.sub(r'(\d+)', lambda m: m.group(1) * n, s)

def enhanced_similarity(s1, s2, repeat_count=5):
    s1_enhanced = repeat_numbers(s1, repeat_count)
    s2_enhanced = repeat_numbers(s2, repeat_count)

    return fuzz.token_sort_ratio(s1_enhanced, s2_enhanced)

# Usage
s1 = "united states jomo williams"
s2 = "united states jerome williams"
score = enhanced_similarity(s1, s2)

print(score)


# In[2]:


S1 = " john doe subscriber assigned ip address 76126173191 strike 3 holdings"
S2 = " strike 3 holdings lls john doe subscriber assigned ip address 76126173191" 


# In[3]:


enhanced_similarity(S1,S2)


# In[4]:


def get_ngrams(s, n=3):
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def sort_words(s, repeat_count=5):
    words = s.split()
    words = [word * repeat_count if word.isdigit() else word for word in words]
    words.sort()
    return ' '.join(words)


def enhanced_similarity_1(s1, s2, n=3):
    s1_sorted = sort_words(s1)
    s2_sorted = sort_words(s2)

    s1_ngrams = get_ngrams(s1_sorted, n)
    s2_ngrams = get_ngrams(s2_sorted, n)

    return jaccard_similarity(s1_ngrams, s2_ngrams)


# In[5]:


enhanced_similarity_1(S1,S2)


# In[6]:


enhanced_similarity_1(s1,s2)


# In[14]:


S3 = " commissioner of social security administration kelly hilton lang"
S4 = " commissioner of social security administration kelly hilton"


# In[15]:


enhanced_similarity(S3, S4)


# In[16]:


enhanced_similarity_1(S3, S4)


# In[ ]:




