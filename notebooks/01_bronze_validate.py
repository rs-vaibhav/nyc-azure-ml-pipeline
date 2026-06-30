STORAGE_ACCOUNT = "dlnycproject"

BRONZE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print(f"Bronze path: {BRONZE}")
print(f"Silver path: {SILVER}")
print(f"Gold path:   {GOLD}")

from pyspark.sql import functions as F

df_311_raw   = spark.read.option("multiline", "true").json(f"{BRONZE}/311/")
df_crime_raw = spark.read.option("multiline", "true").json(f"{BRONZE}/crime/")

print("=== 311 dataset ===")
print(f"Rows: {df_311_raw.count():,}")
print(f"Columns: {len(df_311_raw.columns)}")

print("\n=== Crime dataset ===")
print(f"Rows: {df_crime_raw.count():,}")
print(f"Columns: {len(df_crime_raw.columns)}")

display(df_311_raw.limit(5))
display(df_crime_raw.limit(5))
