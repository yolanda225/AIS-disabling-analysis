import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN") # stored in .env file

PERIODS = {
    "2017_2019": ("2017-01-01", "2019-12-31"),
    "2023_2025": ("2023-01-01", "2025-12-31"),
}
MIN_GAP_HOURS = 12
MAX_GAP_HOURS = 336
MIN_SHORE_DIST_KM = 92.6   # 50 nm
GRID_RES = 0.25