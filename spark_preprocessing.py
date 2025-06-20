from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name

def preprocess_documents(doc_path):
    spark = SparkSession.builder.appName("KnowledgeMining").getOrCreate()
    df = spark.read.text(doc_path).withColumn("filename", input_file_name())
    return df
