STORAGE_ACCOUNT = "dlnycproject"

BRONZE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print(f"Bronze path: {BRONZE}")
print(f"Silver path: {SILVER}")
print(f"Gold path:   {GOLD}")

from pyspark.sql import functions as F

df_311   = spark.read.format("delta").load(f"{SILVER}/311_clean")
df_crime = spark.read.format("delta").load(f"{SILVER}/crime_clean")

# Filter out remaining bad rows
df_311_clean   = df_311.filter(F.col("year").isNotNull() & (F.col("year") > 2000))
df_crime_clean = df_crime.filter(F.col("year").isNotNull() & (F.col("year") > 2000))

print(f"311 valid rows:   {df_311_clean.count():,}")
print(f"Crime valid rows: {df_crime_clean.count():,}")

# ── WEEKLY AGGREGATIONS ──
complaints_weekly = (df_311_clean
    .groupBy("borough", "year", "month", "week")
    .agg(
        F.count("*").alias("total_complaints"),
        F.countDistinct("complaint_type").alias("unique_complaint_types")
    )
)

crimes_weekly = (df_crime_clean
    .groupBy("borough", "year", "month", "week")
    .agg(
        F.count("*").alias("total_crimes"),
        F.sum(F.when(F.col("law_cat_cd") == "FELONY",      1).otherwise(0)).alias("felony_count"),
        F.sum(F.when(F.col("law_cat_cd") == "MISDEMEANOR", 1).otherwise(0)).alias("misdemeanor_count"),
        F.sum(F.when(F.col("law_cat_cd") == "VIOLATION",   1).otherwise(0)).alias("violation_count")
    )
)

# ── MASTER ML FEATURE TABLE ──
gold_features = (complaints_weekly
    .join(crimes_weekly, on=["borough", "year", "month", "week"], how="outer")
    .na.fill(0)
    .withColumn("complaint_crime_ratio",
        F.round(F.col("total_complaints") / (F.col("total_crimes") + 1), 4))
    .withColumn("high_crime_week",
        F.when(F.col("total_crimes") > 500, 1).otherwise(0))
    .orderBy("borough", "year", "week")
)

# ── COMPLAINT + CRIME BREAKDOWNS (for Power BI) ──
complaints_by_type = (df_311_clean
    .groupBy("borough", "year", "week", "complaint_type")
    .agg(F.count("*").alias("count"))
    .orderBy("borough", "year", "week", F.desc("count"))
)

crimes_by_category = (df_crime_clean
    .groupBy("borough", "year", "week", "law_cat_cd", "ofns_desc")
    .agg(F.count("*").alias("count"))
    .orderBy("borough", "year", "week", F.desc("count"))
)

# ── WRITE ALL GOLD TABLES ──
for df, path in [
    (gold_features,       f"{GOLD}/ml_features"),
    (complaints_weekly,   f"{GOLD}/complaints_weekly"),
    (crimes_weekly,       f"{GOLD}/crime_weekly"),
    (complaints_by_type,  f"{GOLD}/complaints_by_type"),
    (crimes_by_category,  f"{GOLD}/crimes_by_category"),
]:
    df.write.format("delta").mode("overwrite") \
      .option("overwriteSchema", "true") \
      .save(path)
    print(f"✓ Written: {path}")

print(f"\nGold ml_features: {gold_features.count():,} rows")
display(gold_features.limit(15))
