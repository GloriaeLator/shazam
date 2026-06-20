# config.py
TARGET_FS = 22050
WINDOW_SIZE = 2048
OVERLAP = 1024

PEAK_THRESHOLD = 98
PEAK_SIZE = 10
FAN_OUT = 5

# CSV Database Configuration
DB_FILE = "fingerprints_db.csv"
SONG_DIR = "./songs/"
# Matching Configuration
MIN_VOTE_THRESHOLD = 10  # Minimum votes required to accept a match
