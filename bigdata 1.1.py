# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import PCA
from pyspark.ml.feature import ChiSqSelector

spark = SparkSession.builder.appName("bd_spark.com").master('local[4]').getOrCreate()
# reading data from csv
df_raw = spark.read.option('inferSchema',True).option('header',True).csv('../1987.csv')

################ BEGIN LOADING, FILTERING AND COMBINING ################
# columns to drop
drop_cols0 = ('ArrTime','ActualElapsedTime','TaxiIn','AirTime','Diverted','CarrierDelay','WeatherDelay','NASDelay',
             'SecurityDelay','LateAircraftDelay')
drop_cols1 = ('Cancelled','CancellationCode','FlightNum','TailNum','Year','CRSDepTime', 'Origin', 'Dest', 
              'UniqueCarrier', 'DayofMonth', 'Month')
# dropping forbidden variables
df= df_raw.drop(*drop_cols0)
# dropping variables that we consider they dont give any valuable info
df = df.filter((df.Cancelled != 1)) # dropping rows with cancelled flights
df = df.drop(*drop_cols1)
# replace 'NA' por None values
df = df.replace('NA', None)
# convert strings to integer
df = df.withColumn('ArrDelay',F.col('ArrDelay').cast('integer'))\
    .withColumn('DepDelay',F.col('DepDelay').cast('integer'))\
    .withColumn('DepTime',F.col('DepTime').cast('integer'))\
    .withColumn('CRSElapsedTime',F.col('CRSElapsedTime').cast('integer'))\
    .withColumn('TaxiOut',F.col('TaxiOut').cast('integer'))\
    .withColumn('Distance',F.col('Distance').cast('integer'))
# Function to drop columns containing all null values
# def drop_null_columns(df):
#
#     null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
#     to_drop = [k for k, v in null_counts.items() if v >= df.count()]
#     df = df.drop(*to_drop)
#
#     return df
# df = drop_null_columns(df)
# # drop rows with NA values
# df = df.dropna('any')
# df = df.withColumn('DepTime', F.when(((0 <= df['DepTime']) & (df['DepTime'] <= 2400)),floor(df['DepTime']/100) * 60 + df['DepTime']%100).otherwise(-1))
# df = df.withColumn('CRSArrTime', F.when(((0 <= df['CRSArrTime']) & (df['CRSArrTime'] <= 2400)),floor(df['CRSArrTime']/100) * 60 + df['CRSArrTime']%100).otherwise(-1))
# df = df.filter((df.DepTime != -1) & (df.CRSArrTime != -1) )
# # map strings to numbers with string indexer
# stringIndexer = StringIndexer(inputCols=['DayOfWeek'], outputCols=['DayOfWeekIndx'])
# model = stringIndexer.fit(df)
# df = model.transform(df)
# # perform onehotencoding
# encoder = OneHotEncoder(dropLast=False, inputCols=['DayOfWeekIndx'], outputCols=['DayOfWeekOHE'])
# model= encoder.fit(df)
# df = model.transform(df)
# # dropping encoded columns
# df = df.drop(*['DayOfWeekIndx','DayOfWeek'])
# ################ END LOADING, FILTERING AND COMBINING ################
#
#
# ################ PREPARAR DF ################
#
# #Prepare df to apply the algorithms
# inputcols=df.drop('ArrDelay').columns
# vectorAssembler = VectorAssembler(inputCols = inputcols, outputCol = 'features')
# df = vectorAssembler.transform(df)
# df = df.select(['features', 'ArrDelay'])
# df = df.withColumnRenamed('ArrDelay','label')
#
# ############### ChiSqSelector ################
#
# selector = ChiSqSelector(selectorType = 'fpr', fpr=0.01, featuresCol="features",
#                          outputCol="selectedFeatures", labelCol="label")
#
# df = selector.fit(df).transform(df)
#
#
# # Split df into train and test
# train, test = df.randomSplit([0.8, 0.2])
#
# ################ ALGORITMOS ML ################
#
# ### linear Regression
# lr = LinearRegression(featuresCol = 'selectedFeatures', labelCol='label')
#
# # Paramethers
# paramGridlr = ParamGridBuilder()\
#     .addGrid(lr.regParam, [0.1, 0.3, 0.6, 1.0]) \
#     .addGrid(lr.fitIntercept, [False, True])\
#     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
#     .build()
#
# #NO Cross-Validation
# Model_lr=lr.fit(train)
#
# #Cross-validation (RMSE)
# crossval = CrossValidator(estimator=lr,
#                           estimatorParamMaps=paramGridlr,
#                           evaluator=RegressionEvaluator(metricName='rmse'),
#                           numFolds=3)  # use 3+ folds in practice
# # Run cross-validation, and choose the best set of parameters.
# Model_lr = crossval.fit(train)
#
# #Predict the test (validation with R2, RMSE)
# a=RegressionEvaluator(metricName='r2').evaluate(Model_lr.transform(test))
# a
#
# b=RegressionEvaluator(metricName='rmse').evaluate(Model_lr.transform(test))
# b
#
#
# ############### A PARTIR DE AQUI HAY QUE LIMPIAR CODIGO Y COMPROBAR QUE FUNCIONAN... ETC ################
#
# ###Generalized Linear Regression, Gaussian e identity
# glr = GeneralizedLinearRegression(family ='gaussian', link= 'identity',maxIter=10)
#
# #Parametros setting
# paramGridglr = ParamGridBuilder()\
#     .addGrid(glr.regParam, [0.1,0.5,1.0])\
#     .build()
#
# #Sin Cross-Validation
# #moglr=glr.fit(train)
# #b=RegressionEvaluator(metricName='rmse').evaluate(moglr.transform(test))
#
#
# #Cross-validation (RMSE)
# crossval = CrossValidator(estimator=glr,
#                           estimatorParamMaps=paramGridglr,
#                           evaluator=RegressionEvaluator(metricName='rmse'),
#                           numFolds=2)  # use 3+ folds in practice
#
# # Run cross-validation, and choose the best set of parameters.
# cvModel_glr = crossval.fit(train)
#
# #Predict the test (RMSE)
# b=RegressionEvaluator(metricName='rmse').evaluate(cvModel_glr.transform(test))
#
# ###Decision Tree
# dt=DecisionTreeRegressor(maxDepth=10)
#
# modt=dt.fit(train)
#
# c=RegressionEvaluator(metricName='rmse').evaluate(modt.transform(test))
#
#
# # Fit the model
# #lrModel = lr.fit(training)
#
# # Print the coefficients and intercept for linear regression
# #print("Coefficients: %s" % str(lrModel.coefficients))
# #print("Intercept: %s" % str(lrModel.intercept))
#
# # Summarize the model over the training set and print out some metrics
# #trainingSummary = lrModel.summary
# #print("numIterations: %d" % trainingSummary.totalIterations)
# #print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
# #trainingSummary.residuals.show()
# #print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
# #print("r2: %f" % trainingSummary.r2)
