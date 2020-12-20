from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator

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
                              numFolds=10)
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
        .addGrid(alg.maxBins, [20, 25, 30]) \
        .build()
    # Cross-validation (RMSE)
    evaluator = RegressionEvaluator(metricName='rmse')
    crossval = CrossValidator(estimator=alg,
                              estimatorParamMaps=paramGridlr,
                              evaluator=evaluator,
                              numFolds=10)
    # Run cross-validation, and choose the best set of parameters.
    cv_model = crossval.fit(df)
    best_model = cv_model.bestModel
    best_maxBins_param = best_model._java_obj.getMaxBins()
    file.write('Minimum RMSE : ' + str(np.min(cv_model.avgMetrics)) + '\n')
    file.write('MaxBins value : ' + str(best_maxBins_param) + '\n')