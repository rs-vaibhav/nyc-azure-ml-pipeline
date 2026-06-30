"""
Shared configuration for all Databricks pipeline notebooks.
Defines ADLS Gen2 storage paths for the Bronze / Silver / Gold layers.
"""

STORAGE_ACCOUNT = "dlnycproject"

BRONZE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"
