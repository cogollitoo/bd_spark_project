from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from tools import kmeans_info, correlation_matrix, one_hot_encoder, apply_LinearRegression, apply_TreeRegressor

import sys
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, LinearRegression, \
    GeneralizedLinearRegression, GBTRegressor

spark = SparkSession.builder.appName("bd_spark.com").master('local[4]').getOrCreate()
file = open('bd_project/output.txt', 'w')

################ BEGIN LOADING, FILTERING AND COMBINING ################
# columns to drop
forbidden_cols = (
'ArrTime', 'ActualElapsedTime', 'TaxiIn', 'AirTime', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
'SecurityDelay', 'LateAircraftDelay')
noinfo_cols = ('Cancelled', 'CancellationCode', 'FlightNum', 'TailNum', 'Year', 'CRSDepTime')
txout = ('TaxiOut')
cols_OHE = ['OriginOHE', 'DestOHE', 'UniqueCarrierOHE']
# reading data from csv
# df_raw = spark.read.option('inferSchema',True).option('header',True).csv('2008.csv')

df_raw = spark.read.option('inferSchema', True).option('header', True).csv(sys.argv[1])

for file in sys.argv[2:]:
    df = spark.read.option('inferSchema', True).option('header', True).csv(file)
    df_raw = df_raw.union(df)
# dropping forbidden variables
df = df_raw.drop(*forbidden_cols)
# dropping variables that we consider they dont give any valuable info
df = df.filter((df.Cancelled != 1))  # dropping rows with cancelled flights
df = df.drop(*noinfo_cols)

# combining columns to get planned speed of the plane

df = df.withColumn('TaxiOut', F.col('TaxiOut').cast('integer'))

df = df.withColumn('TaxiOut', F.when(F.col('TaxiOut').isNull(), 15).otherwise(F.col('TaxiOut')))
df.show(5)
# casting columns to integer
df = df.withColumn('DepDelay', F.col('DepDelay').cast('integer')) \
    .withColumn('ArrDelay', F.col('ArrDelay').cast('integer')) \
    .withColumn('DepDelay', F.col('DepDelay').cast('integer')) \
    .withColumn('Distance', F.col('Distance').cast('integer')) \
    .withColumn('CRSArrTime', F.col('CRSArrTime').cast('integer')) \
    .withColumn('CRSElapsedTime', F.col('CRSElapsedTime').cast('integer')) \
    .withColumn('DayofMonth', F.col('DayofMonth').cast('integer')) \
    .withColumn('DayOfWeek', F.col('DayOfWeek').cast('integer')) \
    .withColumn('Month', F.col('Month').cast('integer')) \
    .withColumn('DepTime', F.col('DepTime').cast('integer'))

df = df.withColumn('CRSPlanSpeed', F.expr('Distance / CRSElapsedTime'))
# drop rows with NA in some column
df = df.na.drop('any')
# map strings to numbers with string indexer
string_cols = ['UniqueCarrier', 'Origin', 'Dest']
df = one_hot_encoder(df=df, inputCols=string_cols, outputCols=cols_OHE)
# correlation matrix
file.write('Correlation Matrix\n')
int_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSArrTime', 'DepDelay', 'CRSPlanSpeed',
            'Distance', 'CRSElapsedTime', 'TaxiOut', 'ArrDelay']
info = correlation_matrix(df, int_cols)
file.write(info)
df_pre = df
# after the correlation analysis dropping variables with correlation less than 0.05
drop_low_corr_cols = ['DayOfWeek', 'DayOfMonth', 'CRSPlanSpeed', 'Distance', 'CRSArrTimeOHE', 'DepTimeOHE']
df = df.drop(*drop_low_corr_cols)
# showing final schema
df.printSchema()
df.show(5)
file.write('\n Final Schema : \n' + df._jdf.schema().treeString())

############### END LOADING, FILTERING AND COMBINING ################

# Trains a k-means model. ( this was for seeing we can make 2 clusters so we could binarize this 2 variables,
# but correlation showed it was better to preserve them in its original codification, binarization code is in tools.py
# inputCols = ['DepTime','ArrDelay']
# file.write('Clustering'+str(inputCols)+'\n')
# for i in range(2,6):
#     info = kmeans_info(i,df,inputCols)
#     file.write(info)
#
# inputCols=['CRSArrTime','ArrDelay']
# file.write('Clustering'+str(inputCols)+'\n')
# # Trains a k-means model.
# for i in range(2,6):
#     info = kmeans_info(i,df,inputCols)
#     file.write(info)

############### PREPARARE DF ################

# Prepare df to apply the algorithms
inputcols = df.drop('ArrDelay').columns
vectorAssembler = VectorAssembler(inputCols=inputcols, outputCol='features')
df_ohe = vectorAssembler.transform(df)
df_ohe = df_ohe.select(['features', 'ArrDelay'])
df_ohe = df_ohe.withColumnRenamed('ArrDelay', 'label')

df = df.drop(*cols_OHE)
inputcols = df.drop('ArrDelay').columns
vectorAssembler = VectorAssembler(inputCols=inputcols, outputCol='features')
df = vectorAssembler.transform(df)
df = df.select(['features', 'ArrDelay'])
df = df.withColumnRenamed('ArrDelay', 'label')

################ ALGORITMOS ML ################
alg = LinearRegression()
file.write('\n---> Linear Regression without OHE variables \n\n')
apply_LinearRegression(df, inputcols, file, alg)
# file.write('\n---> Linear Regression with OHE variables \n\n')
# apply_LinearRegression(df_ohe, inputcols, file, alg)

# alg = GeneralizedLinearRegression()
# file.write('\n---> Generalized Linear Regression without OHE variables \n\n')
# apply_LinearRegression(df, inputcols, file, alg)
# file.write('\n---> Generalized Linear Regression with OHE variables \n\n')
# apply_LinearRegression(df_ohe, inputcols, file, alg)

alg = DecisionTreeRegressor()
file.write('\n---> Decision Tree Regression without OHE variables \n\n')
apply_TreeRegressor(df, file, alg)
# file.write('\n---> Decision Tree Regression with OHE variables \n\n')
# apply_TreeRegressor(df_ohe, file, alg)

alg = RandomForestRegressor()
file.write('\n---> Random Forest Regression without OHE variables \n\n')
apply_TreeRegressor(df, file, alg)
# file.write('\n---> Random Forest Regression with OHE variables \n\n')
# apply_TreeRegressor(df_ohe, file, alg)

alg = GBTRegressor()
file.write('\n---> GBT Regression without OHE variables \n\n')
apply_TreeRegressor(df, file, alg)
# file.write('\n---> GBT Regression Regression with OHE variables \n\n')
# apply_TreeRegressor(df_ohe, file, alg)
