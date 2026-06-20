import csv
import json
import os

import config
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram


def load_audio(path):
    audio, fs = librosa.load(path, sr=config.TARGET_FS, mono=True)
    return fs, audio


def get_spectrogram(audio, fs):
    f, t, Sxx = spectrogram(
        audio,
        fs=fs,
        window="hann",
        nperseg=config.WINDOW_SIZE,
        noverlap=config.OVERLAP,
        mode="magnitude",
    )
    return 20 * np.log10(Sxx + 1e-10)


def get_constellation(S_db):
    local_max = maximum_filter(S_db, size=config.PEAK_SIZE) == S_db
    threshold = np.percentile(S_db, config.PEAK_THRESHOLD)
    peaks = np.argwhere(local_max & (S_db > threshold))

    # Cast to standard int for JSON serialization later
    constellation = [(int(t_idx), int(f_idx)) for f_idx, t_idx in peaks]
    constellation.sort()
    return constellation


def create_hashes(constellation):
    hashes = []
    for i in range(len(constellation)):
        t1, f1 = constellation[i]
        for j in range(1, config.FAN_OUT + 1):
            if i + j >= len(constellation):
                break
            t2, f2 = constellation[i + j]
            dt = t2 - t1
            if dt <= 0:
                continue
            hashes.append(((f1, f2, dt), t1))
    return hashes


def save_precomputed_assets(song_name, S_db, constellation):
    """
    Saves a PNG of the constellation map and a JSON of the raw points
    for instant loading in the Streamlit app.
    """
    os.makedirs("assets", exist_ok=True)
    base_name = os.path.splitext(song_name)[0]

    # 1. Save data for instant overlay rendering
    data_path = os.path.join("assets", f"{base_name}_data.json")
    with open(data_path, "w") as f:
        json.dump({"shape": S_db.shape, "constellation": constellation}, f)

    # 2. Save static PNG
    png_path = os.path.join("assets", f"{base_name}_constellation.png")
    fig, ax = plt.subplots(figsize=(12, 6))

    time_per_frame = (config.WINDOW_SIZE - config.OVERLAP) / config.TARGET_FS

    if constellation:
        t_idx, f_idx = zip(*constellation)
        t_sec = [t * time_per_frame for t in t_idx]
        ax.scatter(t_sec, f_idx, color="lightgrey", s=10, marker="o")

    ax.set_title(f"Constellation Map: {song_name}")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Frequency (Bin Index)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Force strict axis bounds based on spectrogram length
    max_time_sec = S_db.shape[1] * time_per_frame
    ax.set_xlim(0, max_time_sec)
    ax.set_ylim(0, S_db.shape[0])

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)  # Close figure to free memory during bulk indexing


def fingerprint_song(song_path):
    fs, audio = load_audio(song_path)
    S_db = get_spectrogram(audio, fs)
    constellation = get_constellation(S_db)

    # Generate and save instant-load files
    song_name = os.path.basename(song_path)
    save_precomputed_assets(song_name, S_db, constellation)

    return create_hashes(constellation)


def main():
    # Target all mp3s except a potential query file
    songs = [
        f for f in os.listdir("./songs") if f.endswith(".mp3") and f != "query.mp3"
    ]

    if not songs:
        print("No MP3 files found to index.")
        return

    print(f"Found {len(songs)} song(s) to index.")

    with open(config.DB_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header layout
        writer.writerow(["f1", "f2", "dt", "song_name", "anchor_frame"])

        for song in songs:
            print(f"Indexing: {song}...")
            try:
                hashes = fingerprint_song("./songs/" + song)
                # Build all rows for this song up front and flush them in
                # one call instead of one writerow() per hash.
                rows = [[f1, f2, dt, song, anchor] for (f1, f2, dt), anchor in hashes]
                writer.writerows(rows)
            except Exception as e:
                print(f"Error indexing {song}: {e}")

    print(f"\nDatabase successfully saved to '{config.DB_FILE}'")
    print(f"Precomputed assets saved to './assets/'")


if __name__ == "__main__":
    main()
