#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def name_checker(ct, do):
    if ' v. ' in ct and ' v. ' in do:
        sub_ct1, sub_ct2 = ct.split(' v. ', maxsplit=1)
        sub_do1, sub_do2 = do.split(' v. ', maxsplit=1)

        sub_ct1 = ' '.join(sub_ct1.split())  # remove extra spaces in sub_ct1
        sub_ct2_words = sub_ct2.split()  # split into words
        sub_do1 = ' '.join(sub_do1.split())  # remove extra spaces in sub_do1
        sub_do2_words = sub_do2.split()  # split into words

        # Check if any word in sub_ct2 is in sub_do2
        if sub_ct1 in sub_do1 and any(word in sub_do2_words for word in sub_ct2_words):
            return ct
    return do

