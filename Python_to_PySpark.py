#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load the data
df = spark.read.format('csv').option('header', 'true').load('path/data.csv')

# Select necessary columns
df_ct = df.select('Col1', 'Col2', 'Col3')

# Rename columns
df_ct = df_ct.withColumnRenamed('col1', 'col_1').withColumnRenamed('col2', 'col_2')

# Define UDFs
striped = udf(lambda text: text.strip(" []") if isinstance(text,str) else text, StringType())

@udf(StringType())
def clean_df(text):
    get_string_court_opinion = lambda s: s.split(', No.')[0] if isinstance(s, str) else s
    return get_string_court_opinion(text)

@udf(StringType())
def Abbreviation_Convertor(text):
    for key, value in abbreviations.items():  # Please define the 'abbreviations' dictionary
        if str(key) in text:
            text = text.replace(key, value)
    return text

@udf(StringType())
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

# Apply UDFs
df_ct = df_ct.withColumn('col_1', striped(df_ct['col_1']))
df_ct = df_ct.withColumn('col_1', clean_df(df_ct['col_1']))
df_ct = df_ct.withColumn('col_1', Abbreviation_Convertor(df_ct['col_1']))
df_ct = df_ct.withColumn('col_1', name_checker(df_ct['col_1'], df_ct['col_2']))  # Example usage of name_checker UDF

# Remove null rows
df_ct = df_ct.na.drop(subset=['col_1', 'col_2', 'Col3'])

# Reset index - PySpark doesn't have an equivalent to the pandas reset_index() function.
# PySpark dataframes are distributed, and there is no inherent "index" across the entire dataframe.


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, collect_list
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType
from rapidfuzz import fuzz

spark = SparkSession.builder.getOrCreate()

@udf(returnType=ArrayType(FloatType()))
def get_similarity_scores(col1_list, col2_list, product_type1, product_type2):
    def combined_similarity(string1, string2):
        token_set_ratio = fuzz.token_set_ratio(string1, string2)
        character_ratio = fuzz.ratio(string1, string2)
        return (token_set_ratio + character_ratio) / 2

    def calculate_similarity_for_matching_product_type(x, y):
        if x[1] == y[1]:
            return combined_similarity(x[0].lower(), y[0].lower())
        else:
            return 0

    return [calculate_similarity_for_matching_product_type((x, product_type1), (y, product_type2)) for x in col1_list for y in col2_list]

@udf(returnType=ArrayType(IntegerType()))
def get_related_indexes(similarity_list, threshold=80):
    return [i for i, score in enumerate(similarity_list) if score > threshold]

@udf(returnType=ArrayType(FloatType()))
def get_related_scores(similarity_list, threshold=80):
    return [score for score in similarity_list if score > threshold]

@udf(returnType=ArrayType(StringType()))
def get_related_ids(indexes, id_list):
    return [id_list[i] for i in indexes]

# Convert pandas DataFrame to PySpark DataFrame
df1 = spark.createDataFrame(CT_df)
df2 = spark.createDataFrame(Do_df)

# Collect the columns from df2 that need to be compared with df1 into lists
df2 = df2.withColumn("casename_list", collect_list(df2["casename"]).over(Window.partitionBy("court")))
df2 = df2.withColumn("id_list", collect_list(df2["docket_id"]).over(Window.partitionBy("court")))

# Calculate similarity scores
df1 = df1.join(df2, df1["court"] == df2["court"], "inner")
df1 = df1.withColumn("similarity_scores", get_similarity_scores(df1['casename'], df2['casename_list'], df1['court'], df2['court']))
df1 = df1.withColumn("header_related_index", get_related_indexes(df1["similarity_scores"]))
df1 = df1.withColumn("header_related_scores", get_related_scores(df1["similarity_scores"]))
df1 = df1.withColumn("header_related_id", get_related_ids(df1["header_related_index"], df2["id_list"]))

result = df1

