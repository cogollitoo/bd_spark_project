from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from tools import kmeans_info, correlation_matrix, one_hot_encoder, apply_LinearRegression, apply_TreeRegressor

import sys
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, LinearRegression
    
spark = SparkSession.builder.appName("bd_spark.com").master('local[4]').getOrCreate()
file = open('bd_project/output.txt', 'w')

################ BEGIN LOADING, FILTERING AND COMBINING ################
# columns to drop
forbidden_cols = (
'ArrTime', 'ActualElapsedTime', 'TaxiIn', 'AirTime', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
'SecurityDelay', 'LateAircraftDelay')
noinfo_cols = ('Cancelled','CancellationCode','FlightNum','TailNum','CRSDepTime', 'Origin', 'Dest', 'DayofMonth', 'DayOfWeek', 'CRSElapsedTime')

# reading data from csv
df_raw = spark.read.option('inferSchema', True).option('header', True).csv(sys.argv[1])

for file in sys.argv[2:]:
    df = spark.read.option('inferSchema', True).option('header', True).csv(file)
    df_raw = df_raw.union(df)

# dropping forbidden variables
df = df_raw.drop(*forbidden_cols)

# dropping variables that we consider they dont give any valuable info
df = df.filter((df.Cancelled != 1))  # dropping rows with cancelled flights
df = df.drop(*noinfo_cols)

# replace 'NA' por None values
df = df.replace('NA', None)

# convert strings to integer
df = df.withColumn('DepDelay', F.col('DepDelay').cast('integer')) \
    .withColumn('ArrDelay', F.col('ArrDelay').cast('integer')) \
    .withColumn('Distance', F.col('Distance').cast('integer')) \
    .withColumn('CRSArrTime', F.col('CRSArrTime').cast('integer')) \
    .withColumn('TaxiOut',F.col('TaxiOut').cast('integer'))\
    .withColumn('DepTime', F.col('DepTime').cast('integer'))

# Replace Null values in the column 'TaxiOut' with the mean of the column taking onto account all years' datasets
df = df.withColumn('TaxiOut', F.when(F.col('TaxiOut').isNull(), 15).otherwise(F.col('TaxiOut')))

# drop rows with NA values
df = df.dropna('any')

# perform OneHotEncoder for cathegorical features
string_cols = ['Year', 'UniqueCarrier', 'Month']
cols_OHE = ['YearOHE', 'UniqueCarrierOHE', 'MonthOHE']
df = one_hot_encoder(df=df, inputCols=string_cols, outputCols=cols_OHE)

# change format of 'DepTime' and 'CRSArrTime' from hhmm to minutes
df = df.withColumn('DepTime', F.when(((0 <= df['DepTime']) & (df['DepTime'] <= 2400)),F.floor(df['DepTime']/100) * 60 + df['DepTime']%100).otherwise(-1))
df = df.withColumn('CRSArrTime', F.when(((0 <= df['CRSArrTime']) & (df['CRSArrTime'] <= 2400)),F.floor(df['CRSArrTime']/100) * 60 + df['CRSArrTime']%100).otherwise(-1))

############### END LOADING, FILTERING AND COMBINING ################


############### PREPARE DF ################

# Prepare df to apply the algorithms
inputcols = df.drop('ArrDelay').columns
vectorAssembler = VectorAssembler(inputCols=inputcols, outputCol='features')
df = vectorAssembler.transform(df)
df = df.select(['features', 'ArrDelay'])
df = df.withColumnRenamed('ArrDelay','label')

################ ALGORITMOS ML ################
# Linear Regression
alg = LinearRegression()
file.write('\n---> Linear Regression \n\n')
apply_LinearRegression(df, inputcols, file, alg)

#Decision Tree Regressor
alg = DecisionTreeRegressor()
file.write('\n---> Decision Tree Regression \n\n')
apply_TreeRegressor(df, file, alg)

#Ranfom Forest Regressor
alg = RandomForestRegressor()
file.write('\n---> Random Forest Regression \n\n')
apply_TreeRegressor(df, file, alg)

file.close()
