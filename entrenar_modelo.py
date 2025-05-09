from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DoubleType, LongType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Crea la SparkSession
spark = SparkSession.builder.appName("EntrenamientoModelo").getOrCreate()

# # Instalar dependencias necesarias
# spark._sc.install_pypi_package("pandas")
# spark._sc.install_pypi_package("numpy")
# spark._sc.install_pypi_package("matplotlib", "https://pypi.org/simple")

# Define datos de ejemplo o carga históricos de S3
schema = StructType() \
    .add("qty", DoubleType()) \
    .add("quote_qty", DoubleType()) \
    .add("time", LongType())

df = spark.read.json("s3a://mi-bucket/datos_historicos/crypto.json", schema=schema)

# Prepara los datos
assembler = VectorAssembler(inputCols=["qty"], outputCol="features")
assembled_df = assembler.transform(df).select("features", "quote_qty")

# Entrena el modelo
lr = LinearRegression(featuresCol="features", labelCol="quote_qty")
model = lr.fit(assembled_df)

# Guarda el modelo en S3
model.save("s3a://mi-bucket/modelos/regresion_lineal_crypto")
