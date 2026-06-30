STORAGE_ACCOUNT = "dlnycproject"

BRONZE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print(f"Bronze path: {BRONZE}")
print(f"Silver path: {SILVER}")
print(f"Gold path:   {GOLD}")

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ── 311 CLEANING ──
df_311_raw = spark.read.option("multiline", "true").json(f"{BRONZE}/311/")

df_311 = (df_311_raw
    .select(
        "unique_key", "created_date", "closed_date",
        "agency", "complaint_type", "descriptor",
        "borough", "latitude", "longitude",
        "status", "resolution_description"
    )
    .filter(F.col("borough").isNotNull())
    .filter(F.col("created_date").isNotNull())
    .withColumn("created_date", F.col("created_date").cast("timestamp"))
    .withColumn("closed_date",  F.col("closed_date").cast("timestamp"))
    .withColumn("year",  F.year("created_date"))
    .withColumn("month", F.month("created_date"))
    .withColumn("week",  F.weekofyear("created_date"))
    .withColumn("borough",   F.upper(F.trim(F.col("borough"))))
    .withColumn("latitude",  F.col("latitude").cast(DoubleType()))
    .withColumn("longitude", F.col("longitude").cast(DoubleType()))
    .withColumn("source", F.lit("311"))
)

print("=== 311 date sanity check ===")
df_311.select("created_date", "year", "month", "week").limit(5).show()
print(f"311 clean rows: {df_311.count():,}")

# ── CRIME CLEANING ──
df_crime_raw = spark.read.option("multiline", "true").json(f"{BRONZE}/crime/")

df_crime = (df_crime_raw
    .select(
        "cmplnt_num", "cmplnt_fr_dt", "cmplnt_to_dt",
        "ofns_desc", "law_cat_cd", "boro_nm",
        "latitude", "longitude", "susp_sex",
        "susp_age_group", "crm_atpt_cptd_cd"
    )
    .filter(F.col("boro_nm").isNotNull())
    .filter(F.col("cmplnt_fr_dt").isNotNull())
    .withColumn("cmplnt_fr_dt", F.col("cmplnt_fr_dt").cast("timestamp"))
    .withColumn("cmplnt_to_dt", F.col("cmplnt_to_dt").cast("timestamp"))
    .withColumn("year",  F.year("cmplnt_fr_dt"))
    .withColumn("month", F.month("cmplnt_fr_dt"))
    .withColumn("week",  F.weekofyear("cmplnt_fr_dt"))
    .withColumn("borough",   F.upper(F.trim(F.col("boro_nm"))))
    .withColumn("latitude",  F.col("latitude").cast(DoubleType()))
    .withColumn("longitude", F.col("longitude").cast(DoubleType()))
    .withColumn("source", F.lit("crime"))
)

print("=== Crime date sanity check ===")
df_crime.select("cmplnt_fr_dt", "year", "month", "week").limit(5).show()
print(f"Crime clean rows: {df_crime.count():,}")

# ── WRITE TO SILVER ──
df_311.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{SILVER}/311_clean")

df_crime.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{SILVER}/crime_clean")

print("\n✓ Silver layer written successfully")
display(df_311.limit(5))
