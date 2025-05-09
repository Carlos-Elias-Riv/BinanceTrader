from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DoubleType, LongType
from pyspark.sql.functions import from_json, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel

# 1. Spark Session
spark = SparkSession.builder \
    .appName("KafkaToS3_PrediccionRegresion") \
    .getOrCreate()

# 2. Esquema del JSON
schema = StructType() \
    .add("qty", DoubleType()) \
    .add("quote_qty", DoubleType()) \
    .add("time", LongType())

# 3. Lee del tópico Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "msk-broker1:9092") \
    .option("subscribe", "crypto_topic") \
    .option("startingOffsets", "latest") \
    .load()

# 4. Parseo de datos JSON
parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

# 5. Assembler de features
assembler = VectorAssembler(inputCols=["qty"], outputCol="features")
assembled_df = assembler.transform(parsed_df)

# 6. Carga modelo entrenado
lr_model = LinearRegressionModel.load("s3a://mi-bucket/modelos/regresion_lineal_crypto")

# 7. Predicción
predicciones = lr_model.transform(assembled_df)

# 8. Salida con predicción
resultado = predicciones.select("qty", "quote_qty", "time", "prediction")

# 9. Escritura en S3
query = resultado.writeStream \
    .format("csv") \
    .outputMode("append") \
    .option("header", "true") \
    .option("checkpointLocation", "s3a://mi-bucket/checkpoints/crypto_prediccion/") \
    .option("path", "s3a://mi-bucket/predicciones/") \
    .start()

query.awaitTermination()
