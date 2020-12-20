from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Correlation
import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
import numpy as np


def kmeans_info(i, df, inputCols):
    vecAssembler = VectorAssembler(inputCols=inputCols, outputCol='features')
    df = vecAssembler.transform(df)
    info = '---> With ' + str(i) + ' clusters \n'
    kmeans = KMeans(k=i, seed=1)
    model = kmeans.fit(df.select('features'))
    # Make predictions
    predictions = model.transform(df)
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    info += "Silhouette with squared euclidean distance = " + str(silhouette) + '\n'
    info += "Cluster Centers: " + '\n'
    ctr = []
    centers = model.clusterCenters()
    for center in centers:
        ctr.append(center)
        info += str(center) + '\n'
    info += '--------------------\n'
    return info


def correlation_matrix(df, corr_columns, method='pearson'):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)

    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns).to_string()


def one_hot_encoder(df, inputCols, outputCols):
    tmpCols = [col + 'Indx' for col in inputCols]
    stringIndexer = StringIndexer(inputCols=inputCols, outputCols=tmpCols)
    model = stringIndexer.fit(df)
    df = model.transform(df)
    # perform onehotencoding
    encoder = OneHotEncoder(dropLast=False, inputCols=tmpCols, outputCols=outputCols)
    model = encoder.fit(df)
    df = model.transform(df)
    # dropping encoded columns
    df = df.drop(*inputCols)
    df = df.drop(*tmpCols)
    return df


def apply_LinearRegression(df, inputcols, file, alg):
    ### linear Regression
    # Paramethers
    paramGridlr = ParamGridBuilder() \
        .addGrid(alg.regParam, [0.1, 0.3, 0.6, 1.0]) \
        .addGrid(alg.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    # Cross-validation (RMSE)
    evaluator = RegressionEvaluator(metricName='rmse')
    crossval = CrossValidator(estimator=alg,
                              estimatorParamMaps=paramGridlr,
                              evaluator=evaluator,
                              numFolds=10)  # use 3+ folds in practice
    # Run cross-validation, and choose the best set of parameters.
    cv_model = crossval.fit(df)
    best_model = cv_model.bestModel
    best_reg_param = best_model._java_obj.getRegParam()
    best_elasticnet_param = best_model._java_obj.getElasticNetParam()
    file.write('Minimum RMSE : ' + str(np.min(cv_model.avgMetrics)) + '\n')
    file.write('Elastic net param value : ' + str(best_elasticnet_param) + '\n')
    file.write('Reg param value : ' + str(best_reg_param) + '\n')
    file.write(str(inputcols))
    file.write('\n')
    file.write(str(best_model.coefficients))


def apply_TreeRegressor(df, file, alg):
    ### linear Regression
    # Paramethers
    paramGridlr = ParamGridBuilder() \
        .addGrid(alg.maxDepth, [3, 5, 7, 10]) \
        .build()
    # Cross-validation (RMSE)
    evaluator = RegressionEvaluator(metricName='rmse')
    crossval = CrossValidator(estimator=alg,
                              estimatorParamMaps=paramGridlr,
                              evaluator=evaluator,
                              numFolds=10)  # use 3+ folds in practice
    # Run cross-validation, and choose the best set of parameters.
    cv_model = crossval.fit(df)
    best_model = cv_model.bestModel
    best_maxDepth_param = best_model._java_obj.getMaxDepth()
    file.write('Minimum RMSE : ' + str(np.min(cv_model.avgMetrics)) + '\n')
    file.write('MaxDepth value : ' + str(best_maxDepth_param) + '\n')

# df = df.withColumn('DepTimeBinarized', F.when( ((700 <= df['DepTime']) & (df['DepTime'] <= 1400)) , 0).otherwise(1))
# df = df.withColumn('CRSArrTimeBinarized', F.when( ((700 <= df['CRSArrTime']) & (df['CRSArrTime'] <= 1400)) , 0).otherwise(1))
# # map strings to numbers with string indexer
# stringIndexer = StringIndexer(inputCols=['DepTimeBinarized','CRSArrTimeBinarized'], outputCols=['DepTimeIndx','CRSArrTimeIndx'])
# model = stringIndexer.fit(df)
# df = model.transform(df)
# # perform onehotencoding
# encoder = OneHotEncoder(dropLast=False, inputCols=['DepTimeIndx','CRSArrTimeIndx'], outputCols=['DepTimeOHE','CRSArrTimeOHE'])
# model= encoder.fit(df)
# df = model.transform(df)
# # dropping encoded columns
# df = df.drop(*['DepTimeIndx','CRSArrTimeIndx'])
#
# df = df.withColumn('DepTime', F.when(((0 <= df['DepTime']) & (df['DepTime'] <= 2400)),F.floor(df['DepTime']/100) * 60 + df['DepTime']%100).otherwise(-1))
# df = df.withColumn('CRSArrTime', F.when(((0 <= df['CRSArrTime']) & (df['CRSArrTime'] <= 2400)),F.floor(df['CRSArrTime']/100) * 60 + df['CRSArrTime']%100).otherwise(-1))
# df = df.filter((df.DepTime != -1) & (df.CRSArrTime != -1) )
