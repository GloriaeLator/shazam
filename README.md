# Audio Fingerprinting System

A Shazam-style audio fingerprinting system that identifies songs from short audio clips using constellation maps and landmark-based hashing.

The project consists of two major stages:

1. **Database Indexing** – Generate fingerprints from a song library and store them in a searchable database.
2. **Audio Matching** – Identify unknown audio clips by comparing their fingerprints against the database.

---

## Project Structure

```text
.
├── config.py
├── indexer.py
├── matcher.py
├── README.md
├── fingerprints_db.csv      # Generated after indexing
├── assets/                  # Generated after indexing
└── songs/
    ├── song1.mp3
    ├── song2.mp3
    └── ...
```

---

## Installation

Install the required dependencies:

```bash
pip install numpy scipy librosa matplotlib streamlit
```

---

## Step 1: Add Songs

Place all reference songs inside the `songs/` directory.

```text
songs/
├── song1.mp3
├── song2.mp3
└── song3.mp3
```

Only `.mp3` files are indexed.

---

## Step 2: Build the Fingerprint Database

Before any matching can be performed, run:

```bash
python indexer.py
```

This script:

* Loads every song from `songs/`
* Generates spectrograms
* Extracts constellation peaks
* Creates landmark hashes
* Stores fingerprints in `fingerprints_db.csv`
* Generates visualization assets inside `assets/`

Example output:

```text
Found 10 song(s) to index.
Indexing: song1.mp3...
Indexing: song2.mp3...
...
Database successfully saved to 'fingerprints_db.csv'
Precomputed assets saved to './assets/'
```

### Generated Files

#### Fingerprint Database

```text
fingerprints_db.csv
```

Contains all generated audio fingerprints.

#### Visualization Assets

```text
assets/
├── song1_data.json
├── song1_constellation.png
├── song2_data.json
├── song2_constellation.png
└── ...
```

These files are used for instant visualization and alignment rendering without recomputing spectrograms.

---

## Step 3: Use the Matcher

The matcher loads the fingerprint database and identifies query clips.

Example:

```python
from matcher import AudioFingerprintMatcher

matcher = AudioFingerprintMatcher()

result = matcher.match("query.wav")

print(result)
```

Example output:

```python
{
    "match_found": True,
    "song_name": "song1.mp3",
    "confidence": 124,
    "offset_seconds": 34.8,
    "offset_frames": 749
}
```

---

## Fingerprinting Pipeline

### 1. Audio Loading

All audio is converted to:

* Mono
* 22050 Hz sampling rate

---

### 2. Spectrogram Generation

A magnitude spectrogram is generated using:

* Hann window
* Window size: 2048
* Overlap: 1024

---

### 3. Constellation Map Extraction

The algorithm:

1. Finds local maxima in the spectrogram.
2. Applies a percentile threshold.
3. Retains only dominant spectral peaks.

These peaks form the constellation map.

---

### 4. Landmark Hashing

For each peak:

* Future neighboring peaks are selected.
* Frequency pairs and time differences are encoded.

Hash format:

```text
(f1, f2, Δt)
```

These hashes are robust against:

* Noise
* Volume changes
* Partial clips
* Compression artifacts

---

### 5. Matching

Query hashes are compared against the database.

For every matching hash:

```text
offset = database_time - query_time
```

The correct song produces a strong cluster of identical offsets.

The most common offset determines:

* Song identity
* Clip alignment
* Match confidence

---

## Configuration

All major parameters are stored in `config.py`.

```python
TARGET_FS
WINDOW_SIZE
OVERLAP

PEAK_THRESHOLD
PEAK_SIZE
FAN_OUT

MIN_VOTE_THRESHOLD
```

These values can be adjusted to trade off:

* Accuracy
* Robustness
* Runtime
* Database size

---

## Running the Streamlit Interface

```bash
streamlit run app.py
```

Features:

* Browse indexed songs
* Visualize spectrograms
* Inspect constellation maps
* Upload query clips
* View matching diagnostics
* Display alignment overlays

---

## Technical Optimizations

Several performance optimizations were implemented to improve indexing and matching speed:

### 1. Packed 64-bit Hash Representation

Instead of storing hashes as Python tuples:

```text
(f1, f2, dt)
```

they are packed into a single `uint64` integer.

Benefits:

* Lower memory usage
* Faster comparisons
* Better cache locality

---

### 2. Binary Search-Based Hash Lookup

Database hashes are sorted once during loading.

Matching uses:

```python
numpy.searchsorted()
```

instead of scanning the entire database.

Benefits:

* O(log N) lookup behavior
* Massive speedup on large databases

---

### 3. NumPy Vectorized Voting

Offset voting is performed using vectorized NumPy operations.

Benefits:

* Eliminates Python loops
* Improves throughput significantly

---

### 4. Binary Database Cache

A compiled cache:

```text
fingerprints_db.csv.cache.npz
```

is automatically generated.

Benefits:

* Avoids repeated CSV parsing
* Faster startup times

---

### 5. Precomputed Visualization Assets

Constellation plots and metadata are generated during indexing.

Benefits:

* Instant UI rendering
* No repeated spectrogram computation

---

### 6. Batch CSV Writes

Hashes for a song are accumulated and written in bulk.

Benefits:

* Reduced I/O overhead
* Faster indexing

---

### 7. Cached Streamlit Resources

Database loading and song analysis are cached using:

```python
st.cache_resource()
st.cache_data()
```

Benefits:

* Faster page refreshes
* Reduced recomputation

---

## Future Improvements

Potential enhancements:

* Real-time microphone identification
* Multi-threaded indexing
* GPU-accelerated spectrogram generation
* Approximate nearest-neighbor search
* Distributed fingerprint database
* Support for additional audio formats
* Incremental database updates

---

## Author

EE200 Course Project – Audio Fingerprinting System

Inspired by the landmark hashing approach used in commercial music identification systems such as Shazam.
