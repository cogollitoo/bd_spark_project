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
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

spark = SparkSession.builder.appName("bd_spark.com").master('local[4]').getOrCreate()
#file = open('C:/Users/Usuario/Desktop/bd_spark_project-main/analysis.txt', 'w')

################ BEGIN LOADING, FILTERING AND COMBINING ################
# columns to drop
drop_cols0 = ('ArrTime','ActualElapsedTime','TaxiIn','AirTime','Diverted','CarrierDelay','WeatherDelay','NASDelay',
             'SecurityDelay','LateAircraftDelay')
drop_cols1 = ('Cancelled','CancellationCode','FlightNum','TailNum','TaxiOut','Year','CRSDepTime')
drop_cols2 = ('Distance','CRSElapsedTime')
cols_OHE = ['OriginOHE','DestOHE','UniqueCarrierOHE']
# reading data from csv
df_raw = spark.read.option('inferSchema',True).option('header',True).csv('C:/Users/Usuario/Desktop/bd_spark_project-main/1987.csv')
# dropping forbidden variables
df= df_raw.drop(*drop_cols0)
# dropping variables that we consider they dont give any valuable info
df = df.filter((df.Cancelled != 1)) # dropping rows with cancelled flights
df = df.drop(*drop_cols1)
# combining columns to get planned speed of the plane
#df = df.withColumn('CRSPlanSpeed', expr('Distance / CRSElapsedTime'))
# dropping varaibles used to create new columns
#df = df.drop(*drop_cols2)
df = df.withColumn('ArrDelay',F.col('ArrDelay').cast('integer'))\
    .withColumn('DepDelay',F.col('DepDelay').cast('integer'))\
    .withColumn('DepTime',F.col('DepTime').cast('integer'))\
    .withColumn('CRSElapsedTime',F.col('CRSElapsedTime').cast('integer'))\
    .withColumn('Distance',F.col('Distance').cast('integer'))
# drop rows with NA values
df = df.dropna(subset=([c for c in df.columns if c not in {'Origin', 'Dest'}]))
# map strings to numbers with string indexer
stringIndexer = StringIndexer(inputCols=['UniqueCarrier','Origin','Dest'], outputCols=['UniqueCarrierIndx','OriginIndx','DestIndx'])
model = stringIndexer.fit(df)
df = model.transform(df)
# perform onehotencoding
encoder = OneHotEncoder(dropLast=False, inputCols=['UniqueCarrierIndx','OriginIndx','DestIndx'], outputCols=['UniqueCarrierOHE','OriginOHE','DestOHE'])
model= encoder.fit(df)
df = model.transform(df)
# dropping encoded columns
df = df.drop(*['UniqueCarrier','Origin','Dest'])
df = df.drop(*['UniqueCarrierIndx','OriginIndx','DestIndx'])

df = df.withColumn('DepTimeBinarized', F.when( ((700 <= df['DepTime']) & (df['DepTime'] <= 1400)) , 0).otherwise(1))
df = df.withColumn('CRSArrTimeBinarized', F.when( ((700 <= df['CRSArrTime']) & (df['CRSArrTime'] <= 1400)) , 0).otherwise(1))
# map strings to numbers with string indexer
stringIndexer = StringIndexer(inputCols=['DepTimeBinarized','CRSArrTimeBinarized'], outputCols=['DepTimeIndx','CRSArrTimeIndx'])
model = stringIndexer.fit(df)
df = model.transform(df)
# perform onehotencoding
encoder = OneHotEncoder(dropLast=False, inputCols=['DepTimeIndx','CRSArrTimeIndx'], outputCols=['DepTimeOHE','CRSArrTimeOHE'])
model= encoder.fit(df)
df = model.transform(df)
# dropping encoded columns
df = df.drop(*['DepTimeIndx','CRSArrTimeIndx'])

df = df.withColumn('DepTime', F.when(((0 <= df['DepTime']) & (df['DepTime'] <= 2400)),floor(df['DepTime']/100) * 60 + df['DepTime']%100).otherwise(-1))
df = df.withColumn('CRSArrTime', F.when(((0 <= df['CRSArrTime']) & (df['CRSArrTime'] <= 2400)),floor(df['CRSArrTime']/100) * 60 + df['CRSArrTime']%100).otherwise(-1))
df = df.filter((df.DepTime != -1) & (df.CRSArrTime != -1) )

#file.write('Correlation Matrix\n')
#int_cols = ['Month','DayofMonth','DayOfWeek','DepTimeBinarized','CRSArrTimeBinarized','DepTime','CRSArrTime','DepDelay','CRSPlanSpeed','ArrDelay']
#info = correlation_matrix(df,int_cols)
#file.write(info)

# showing final schema
df.printSchema()
df.show()
df = df.drop(*['DepTimeBinarized','CRSArrTimeBinarized','DayofMonth'])

# map strings to numbers with string indexer
stringIndexer = StringIndexer(inputCols=['DayOfWeek'], outputCols=['DayOfWeekIndx'])
model = stringIndexer.fit(df)
df = model.transform(df)
# perform onehotencoding
encoder = OneHotEncoder(dropLast=False, inputCols=['DayOfWeekIndx'], outputCols=['DayOfWeekOHE'])
model= encoder.fit(df)
df = model.transform(df)
# dropping encoded columns
df = df.drop(*['DayOfWeekIndx','DayOfWeek'])
df.show(3)
################ END LOADING, FILTERING AND COMBINING ################


################ PREPARAR DF ################

#Preparo df para poder aplicarlo a los modelos
inputcols1=['DepDelay', 'DayOfWeekOHE', 'UniqueCarrierOHE', 'OriginOHE', 'DestOHE', 'DepTime','CRSArrTime','CRSElapsedTime','Distance']
inputcols2=['DepDelay', 'DayOfWeekOHE', 'UniqueCarrierOHE', 'OriginOHE', 'DestOHE', 'DepTimeOHE','CRSArrTimeOHE','CRSElapsedTime','Distance']
vectorAssembler = VectorAssembler(inputCols = inputcols1, outputCol = 'features')
df = vectorAssembler.transform(df)
df = df.select(['features', 'ArrDelay'])
df = df.withColumnRenamed('ArrDelay','label')
df.show(3)


################ PCA ################

# pca
pca = PCA(k=7, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

pcadf = model.transform(df).select("pcaFeatures","label")
pcadf = pcadf.withColumnRenamed('pcaFeatures','features')
pcadf.show(truncate=False)

################ MinMaxScaler ################

# MinMaxScaler
from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(df)

# rescale each feature to range [min, max].
scaledData = scalerModel.transform(df).select("scaledFeatures","label")
scaledData = scaledData.withColumnRenamed('scaledFeatures','features')
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
scaledData.show(truncate=False)


#####ChiSqSelector
from pyspark.ml.feature import ChiSqSelector
selector = ChiSqSelector(numTopFeatures=25, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="label")

df = selector.fit(df).transform(df)


#Separo df en train/test
# CAMBIAR df POR pcadf O scaledData
train, test = df.randomSplit([0.8, 0.2])

################ ALGORITMOS ML ################

###Defino el linear Regression
lr = LinearRegression(featuresCol = 'features', labelCol='label',maxIter=3,regParam=0.3)

#Parametros a probar
paramGridlr = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.3, 0.6, 1.0]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

#Sin Cross-Validation
#molr=lr.fit(train)
#a=RegressionEvaluator(metricName='rmse').evaluate(molr.transform(test))

#Cross-validation (RMSE)
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGridlr,
                          evaluator=RegressionEvaluator(metricName='rmse'),
                          numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel_lr = crossval.fit(train)

#Predict the test (RMSE)
a=RegressionEvaluator(metricName='rmse').evaluate(cvModel_lr.transform(test))


###Generalized Linear Regression, Gaussian e identity
glr = GeneralizedLinearRegression(family ='gaussian', link= 'identity',maxIter=10)

#Parametros setting
paramGridglr = ParamGridBuilder()\
    .addGrid(glr.regParam, [0.1,0.5,1.0])\
    .build()
    
#Sin Cross-Validation
#moglr=glr.fit(train)
#b=RegressionEvaluator(metricName='rmse').evaluate(moglr.transform(test))


#Cross-validation (RMSE)
crossval = CrossValidator(estimator=glr,
                          estimatorParamMaps=paramGridglr,
                          evaluator=RegressionEvaluator(metricName='rmse'),
                          numFolds=2)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel_glr = crossval.fit(train)

#Predict the test (RMSE)
b=RegressionEvaluator(metricName='rmse').evaluate(cvModel_glr.transform(test))

###Decision Tree
dt=DecisionTreeRegressor(maxDepth=10)

modt=dt.fit(train)

c=RegressionEvaluator(metricName='rmse').evaluate(modt.transform(test))


# Fit the model
#lrModel = lr.fit(training)

# Print the coefficients and intercept for linear regression
#print("Coefficients: %s" % str(lrModel.coefficients))
#print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
#trainingSummary = lrModel.summary
#print("numIterations: %d" % trainingSummary.totalIterations)
#print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
#trainingSummary.residuals.show()
#print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
#print("r2: %f" % trainingSummary.r2)
