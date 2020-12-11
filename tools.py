from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans

def kmeans_info(i,df,f):
    f.write('---> With '+str(i)+' clusters \n')
    kmeans = KMeans(k=i, seed=1)
    model = kmeans.fit(df.select('features'))
    # Make predictions
    predictions = model.transform(df)
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    f.write("Silhouette with squared euclidean distance = " + str(silhouette)+'\n')
    f.write("Cluster Centers: "+'\n')
    ctr = []
    centers = model.clusterCenters()
    for center in centers:
        ctr.append(center)
        f.write(str(center)+'\n')
    f.write('--------------------\n')