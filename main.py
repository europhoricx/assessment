from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    row_number,
    desc,
    avg,
    max as spark_max,
    min as spark_min,
    coalesce,
    lit
)
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    LongType,
    DoubleType
)
import os

def main():
    spark = SparkSession.builder \
        .appName("MovieLensAnalysis") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "4") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()

    movies_schema = StructType([
        StructField("MovieID", IntegerType(), False),  
        StructField("Title", StringType(), True),
        StructField("Genres", StringType(), True)     
    ])

    ratings_schema = StructType([
        StructField("UserID", IntegerType(), False),
        StructField("MovieID", IntegerType(), False),
        StructField("Rating", IntegerType(), False),
        StructField("Timestamp", LongType(), True)
    ])

    input_path = "./ml-1m/"
    output_path = "./output/"

    try:

        movies_df = spark.read \
            .option("sep", "::") \
            .option("charset", "UTF-8") \
            .schema(movies_schema) \
            .csv(os.path.join(input_path, "movies.dat"))

        ratings_df = spark.read \
            .option("sep", "::") \
            .option("charset", "UTF-8") \
            .schema(ratings_schema) \
            .csv(os.path.join(input_path, "ratings.dat"))


        assert movies_df.count() > 0, "Movies dataset is empty"
        assert ratings_df.count() > 0, "Ratings dataset is empty"
        print(f"Loaded {movies_df.count()} movies and {ratings_df.count()} ratings")

    except Exception as e:
        print(f"Error: {str(e)}")
        spark.stop()
        return

    movie_stats = ratings_df.groupBy("MovieID") \
        .agg(
            spark_max("Rating").alias("max_rating"),
            spark_min("Rating").alias("min_rating"),
            avg("Rating").cast(DoubleType()).alias("avg_rating")
        )

    movies_with_ratings = movies_df.join(  
        movie_stats,
        on="MovieID",
        how="left"
    ).select(
        "MovieID",
        "Title",
        "Genres",
        coalesce("max_rating", lit(0)).alias("max_rating"),
        coalesce("min_rating", lit(0)).alias("min_rating"),
        coalesce("avg_rating", lit(0.0)).alias("avg_rating")
    )

    user_movie_ratings = ratings_df.groupBy("UserID", "MovieID") \
        .agg(spark_max("Rating").alias("Rating"),
            spark_max("Timestamp").alias("Timestamp")  
)

    window_spec = Window \
        .partitionBy("UserID") \
        .orderBy(
            col("Rating").desc(),
            col("Timestamp").desc(),  
            col("MovieID").asc()      
        )

    top3_movies = user_movie_ratings \
        .withColumn("rank", row_number().over(window_spec)) \
        .filter(col("rank") <= 3) \
        .join(
            movies_df.select("MovieID", "Title", "Genres"),
            on="MovieID",
            how="inner"
        ) \
        .select(
            "UserID",
            "MovieID",
            "Title",
            "Genres",
            "Rating",
            col("rank").alias("rank_order")  
        ) \
        .orderBy("UserID", "rank_order")



    os.makedirs(output_path, exist_ok=True)

    movies_df.write \
        .mode("overwrite") \
        .parquet(os.path.join(output_path, "movies.parquet"))

    ratings_df.write \
        .mode("overwrite") \
        .parquet(os.path.join(output_path, "ratings.parquet"))

    movies_with_ratings.write \
        .mode("overwrite") \
        .parquet(os.path.join(output_path, "movies_with_ratings.parquet"))

    top3_movies.write \
        .mode("overwrite") \
        .parquet(os.path.join(output_path, "top3_movies.parquet"))

    print("Processing got Completed")

    spark.stop()

if __name__ == "__main__":
    main()