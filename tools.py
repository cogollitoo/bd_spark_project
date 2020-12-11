from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd

def kmeans_info(i,df,inputCols):
    vecAssembler = VectorAssembler(inputCols=inputCols, outputCol='features')
    df = vecAssembler.transform(df)
    info = '---> With '+str(i)+' clusters \n'
    kmeans = KMeans(k=i, seed=1)
    model = kmeans.fit(df.select('features'))
    # Make predictions
    predictions = model.transform(df)
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    info += "Silhouette with squared euclidean distance = " + str(silhouette)+'\n'
    info += "Cluster Centers: "+'\n'
    ctr = []
    centers = model.clusterCenters()
    for center in centers:
        ctr.append(center)
        info += str(center)+'\n'
    info += '--------------------\n'
    return info


def correlation_matrix(df, corr_columns, method='pearson'):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)

    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns).to_string()