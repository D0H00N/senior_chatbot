from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_json

def main():
    # 1) SparkSession 생성
    spark = SparkSession.builder \
        .appName("LoadChunksToMySQL") \
        .config("spark.sql.session.timeZone", "Asia/Seoul") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # 2) 로컬에 저장된  Parquet HDFS 에 저장

    #preproc_path = "hdfs://localhost:9000/user/dhl96/chunks_transcript"
    preproc_path = "file:///C:/project/senior_chatbot/df_text_embed"
    df = spark.read.parquet(preproc_path)

    df.write.mode("overwrite").parquet("hdfs://localhost:9000/user/dhl96/chunks_transcript")

    df_out = (df
    .withColumn("embedding", to_json(col("embedding")))  # StringType으로 변환됨
    .select("video_id","title","upload_date","chunk_id","start_time","end_time","text","embedding","image_path")
)


    # 3) JDBC 설정
    jdbc_url = "jdbc:mysql://localhost:3306/senior_chatbot?serverTimezone=Asia/Seoul"
    connection_properties = {
        "user": "root",
        "password": "1224",
        "driver": "com.mysql.cj.jdbc.Driver"
    }

    # 4) MySQL에 적재
    (
        df_out.write
          .format("jdbc")
          .option("url", jdbc_url)
          .option("dbtable", "chatbot_data")
          .option("user", connection_properties["user"])
          .option("password", connection_properties["password"])
          .option("driver", connection_properties["driver"])
          .option("batchsize", "1000")
          .option("isolationLevel", "NONE")
          .mode("append")
          .save()
    )


    print(" MySQL 적재 완료: senior_chatbot.                  ")
    spark.stop()

if __name__ == "__main__":
    try:
        main()
        print("[hdfs_to_mysql] done")
    except Exception as e:
        import traceback
        print("[hdfs_to_mysql] ERROR:", e)
        traceback.print_exc()
        raise