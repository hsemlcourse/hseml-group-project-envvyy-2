import sys
sys.path.insert(0, 'src')
from data_loader import load_twitch_data
from pathlib import Path

data_path = Path("data/raw/twitchdata-update.csv")
if data_path.exists():
    df = load_twitch_data(data_path)
    print(f"Loaded: {len(df)} rows x {len(df.columns)} cols")
    print(df.head())
    print(df.dtypes)
else:
    print("No data found - please download first")