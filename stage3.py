
# coding: utf-8
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import time
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession \
    .builder \
    .appName("stage3") \
    .getOrCreate()


train_data = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
test_data = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"


# ## Preprocess train data
train_df = spark.read.csv(train_data, header=False, inferSchema="true")
train_df = train_df.withColumnRenamed("_c0","label")
assembler = VectorAssembler(inputCols=train_df.columns[1:784], outputCol="features")
output = assembler.transform(train_df)
train_features = output.select(['features', 'label'])
train_data_labeled=train_features.rdd.map(lambda line: LabeledPoint(line[1],[line[0]]))


# ## Preprocess test data
test_df = spark.read.csv(test_data, header=False, inferSchema="true")
test_df = test_df.withColumnRenamed("_c0","label")

assembler = VectorAssembler(inputCols=test_df.columns[1:784], outputCol="features")
output = assembler.transform(test_df)
test_features = output.select(['features', 'label'])
test_data_labeled=test_features.rdd.map(lambda line: LabeledPoint(line[1],[line[0]]))


### PCA

d1 = 50
d2 = 100

pca_1 = PCA(k=d1, inputCol="features", outputCol="pca")
model_1 = pca_1.fit(train_features)
transformed_train_1 = model_1.transform(train_features)
transformed_test_1 = model_1.transform(test_features)
train_data_labeled_PCA_1=transformed_train_1.rdd.map(lambda line: LabeledPoint(line[1],[line[2]]))
test_data_labeled_PCA_1=transformed_test_1.rdd.map(lambda line: LabeledPoint(line[1],[line[2]]))


pca_2 = PCA(k=d2, inputCol="features", outputCol="pca")
model_2 = pca_2.fit(train_features)
transformed_train_2 = model_2.transform(train_features)
transformed_test_2 = model_2.transform(test_features)
train_data_labeled_PCA_2=transformed_train_2.rdd.map(lambda line: LabeledPoint(line[1],[line[2]]))
test_data_labeled_PCA_2=transformed_test_2.rdd.map(lambda line: LabeledPoint(line[1],[line[2]]))


# Logistic Regression
import pyspark.mllib.regression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

def predictions(train_data_labeled,test_data_labeled):

    time_start=time.time()
    model_lrm = LogisticRegressionWithLBFGS.train(train_data_labeled,
                iterations=100, initialWeights=None, regParam=0.01,
                regType='l2', intercept=False, corrections=10, tolerance=0.0001,
                validateData=True, numClasses=10)


    predictions = model_lrm.predict(test_data_labeled.map(lambda x: x.features))
    predict_label = test_data_labeled.map(lambda x: x.label).repartition(1).saveAsTextFile("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/user/czho9311/stage3")
    labels_and_predictions = test_data_labeled.map(lambda x: x.label).zip(predictions)
    lrAccuracy = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data_labeled.count())

    time_end=time.time()
    time_lrm=(time_end - time_start)
    print("=========================================================================================================")
    print("run time: {},LogisticRegression accuracy: {}".format(time_lrm,lrAccuracy))


predictions(train_data_labeled_PCA_1,test_data_labeled_PCA_1)

predictions(train_data_labeled_PCA_2,test_data_labeled_PCA_2)

predictions(train_data_labeled,test_data_labeled)



from pyspark.mllib.tree import RandomForest


def predictions_RF(train_data_labeled,test_data_labeled,RF_NUM_TREES):

    time_start=time.time()
    model_rf = RandomForest.trainClassifier(train_data_labeled, numClasses=10, categoricalFeaturesInfo={},
            numTrees=RF_NUM_TREES, featureSubsetStrategy="auto", impurity="gini",
            maxDepth=10, maxBins=32, seed=10)


    predictions = model_rf.predict(test_data_labeled.map(lambda x: x.features))
    predict_label = test_data_labeled.map(lambda x: x.label).repartition(1).saveAsTextFile("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/user/czho9311/stage3")
    labels_and_predictions = test_data_labeled.map(lambda x: x.label).zip(predictions)
    rfAccuracy = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data_labeled.count())

    time_end=time.time()
    time_rf=(time_end - time_start)
    print("=========================================================================================================")
    print("run time: {},RandomForest accuracy: {}".format(time_rf,rfAccuracy))


predictions_RF(train_data_labeled,test_data_labeled,RF_NUM_TREES=3)

predictions_RF(train_data_labeled,test_data_labeled,RF_NUM_TREES=5)

predictions_RF(train_data_labeled,test_data_labeled,RF_NUM_TREES=7)
