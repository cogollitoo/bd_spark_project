from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pandas as pd




# spark = SparkSession.builder.appName("Predict Adult Salary").getOrCreate()
#
# schema = StructType([
#     StructField("age", IntegerType(), True),
#     StructField("workclass", StringType(), True),
#     StructField("fnlwgt", IntegerType(), True),
#     StructField("education", StringType(), True),
#     StructField("education-num", IntegerType(), True),
#     StructField("marital-status", StringType(), True),
#     StructField("occupation", StringType(), True),
#     StructField("relationship", StringType(), True),
#     StructField("race", StringType(), True),
#     StructField("sex", StringType(), True),
#     StructField("capital-gain", IntegerType(), True),
#     StructField("capital-loss", IntegerType(), True),
#     StructField("hours-per-week", IntegerType(), True),
#     StructField("native-country", StringType(), True),
#     StructField("salary", StringType(), True)
# ])
#
#
#
# train_df = spark.read.csv('train.csv', header=False, schema=schema)
# test_df = spark.read.csv('test.csv', header=False, schema=schema)
#
# print(train_df.limit(5).toPandas())


