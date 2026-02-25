import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_ANON_KEY"]

client = create_client(url, key)

# Check both tables exist by querying them
for table in ("companies", "stock_prices"):
    result = client.table(table).select("*").limit(1).execute()
    print(f"✓ Table '{table}' is accessible")

print("\nConnection successful!")
