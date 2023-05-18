#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from rapidfuzz import fuzz
from multiprocessing import Pool

def combined_similarity(string1, string2):
    token_set_ratio = fuzz.token_set_ratio(string1, string2)
    character_ratio = fuzz.ratio(string1, string2)
    return (token_set_ratio + character_ratio) / 2

def get_related_info(similarity_list):
    indexes = [i for i, score in enumerate(similarity_list) if score > threshold]
    scores = [score for score in similarity_list if score > threshold]
    return indexes, scores

def calculate_similarity_for_matching_product_type(x, y):
    if x[1] == y[1]:
        y_new = (name_checker(x[0], y[0]), y[1])  # Create a new tuple instead of modifying y
        return combined_similarity(x[0].lower(), y_new[0].lower())
    else:
        return 0

def process_row(x):
    similarity_scores = [calculate_similarity_for_matching_product_type((x[col1], x[product_type_col]), (y[col2], y[product_type_col])) for _, y in df2.iterrows()]
    related_indexes, related_scores = get_related_info(similarity_scores)
    related_ids = [df2.loc[i, 'ID'] for i in related_indexes]
    return similarity_scores, related_indexes, related_scores, related_ids

def comparison_score(df1, df2, col1, col2, product_type_col, threshold, related_index_col, related_id_col):
    with Pool() as pool:
        results = pool.map(process_row, df1.iterrows())

    df1['similarity_scores'], df1[related_index_col], df1['related_scores'], df1[related_id_col] = zip(*results)

    return df1

