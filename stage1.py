# import findspark
# findspark.init()
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler

from numpy import linalg as la
import numpy as np
from collections import Counter
sc = SparkContext()

spark = SparkSession \
    .builder \
    .appName("Group 9 ") \
    .getOrCreate()


def knn(test):
    test_np = np.array(test[1])
    label_count = []
    test_set = np.tile(test_np, (a, b))
    Distance = la.norm((test_set - train_set), axis=1)
    distance_all = np.argsort(Distance)[:k]
    for i in distance_all:
        label_count.append(label.value[i][0])
    predict = (Counter(label_count).most_common(1))[0][0]
    return (float(predict),float(test[0]))
d = 50
k= 5
train_data = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
train_df = spark.read.csv(train_data, header=False, inferSchema="true")
assembler_train = VectorAssembler(inputCols=train_df.columns[1:784], outputCol="features")
train_vectors = assembler_train.transform(train_df).select(train_df.columns[0],"features")

pca = PCA(k=d, inputCol="features", outputCol="pca")
train_model = pca.fit(train_vectors)
pca_train = train_model.transform(train_vectors).select(train_vectors.columns[0],'pca')

test_data = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
test_df = spark.read.csv(test_data, header=False, inferSchema="true")
assembler_test = VectorAssembler(inputCols=test_df.columns[1:784], outputCol="features")
test_vectors = assembler_test.transform(test_df).select(test_df.columns[0],"features")

pca_test = train_model.transform(test_vectors).select(test_vectors.columns[0],'pca')
p_number = pca_test.rdd.count()




train_np1 = np.array(pca_train.select('pca').collect())
train_label = np.array(pca_train.select('_c0').collect())
a,b,c = train_np1.shape
train_set = train_np1.reshape(a,c)
train_np = sc.broadcast(train_np1)
label = sc.broadcast(train_label)
result = pca_test.rdd.map(knn)


predict_rdd = result.map(lambda x: x[0])
predict_rdd.saveAsTextFile("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/user/czho9311/123")
acc = result.filter(lambda x:x[0]==x[1]).count() / float(p_number)

metrics = MulticlassMetrics(result)
label_metrics = ['0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0']
pre_dict = {}
recall_dict = {}
f_score = {}
for i in label_metrics:
    pre_dict[i] = metrics.precision(i)
    recall_dict[i] = metrics.recall(i)
    f_score[i] = (2*pre_dict[i]*recall_dict[i])/(pre_dict[i]+recall_dict[i])
print("======This is recall=====")
for i in recall_dict.items():
    print(i)
print("======This is pre_dict=====")
for i in pre_dict.items():
    print (i)
print("======This is f_score=====")
for i in f_score.items():
    print(i)
print("======This is confusionMatrix=====")
print(metrics.confusionMatrix().toArray())
print("======This is weightedPrecision=====")
print(metrics.weightedPrecision)
print("======This is weightedRecall=====")
print(metrics.weightedRecall)
print("======This is weightedFMeasure=====")
print(metrics.weightedFMeasure())
print("======This is accuracy=====")
print(acc)
