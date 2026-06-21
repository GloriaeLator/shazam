
import os
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram
import config

DB_PATH = config.DB_FILE.replace('.csv', '.npz')

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

    constellation = [(int(t_idx), int(f_idx)) for f_idx, t_idx in peaks]
    constellation.sort()
    return constellation

def pack_hash(f1, f2, dt):
    return (np.uint64(f1) << 32) | (np.uint64(f2) << 16) | np.uint64(dt)

def fingerprint_song(song_path):
    fs, audio = load_audio(song_path)
    S_db = get_spectrogram(audio, fs)
    constellation = get_constellation(S_db)

    hashes = []
    times = []
    for i in range(len(constellation)):
        t1, f1 = constellation[i]
        for j in range(1, config.FAN_OUT + 1):
            if i + j >= len(constellation):
                break
            t2, f2 = constellation[i + j]
            dt = t2 - t1
            if dt <= 0:
                continue
            hashes.append(pack_hash(f1, f2, dt))
            times.append(t1)
            
    return hashes, times

def main():
    songs = [f for f in os.listdir(config.SONG_DIR) if f.endswith(".mp3") and f != "query.mp3"]

    if not songs:
        print("No MP3 files found to index.")
        return

    print(f"Found {len(songs)} song(s) to index.")

    all_hashes = []
    all_anchors = []
    all_songs = []
    song_names = []

    for s_id, song in enumerate(songs):
        print(f"Indexing: {song}...")
        try:
            song_names.append(song)
            hashes, times = fingerprint_song(os.path.join(config.SONG_DIR, song))
            
            all_hashes.extend(hashes)
            all_anchors.extend(times)
            all_songs.extend([s_id] * len(hashes))
            
        except Exception as e:
            print(f"Error indexing {song}: {e}")

    print("Sorting and compiling binary database...")
    all_hashes_arr = np.array(all_hashes, dtype=np.uint64)
    all_anchors_arr = np.array(all_anchors, dtype=np.int32)
    all_songs_arr = np.array(all_songs, dtype=np.int32)

    # Pre-sort the database so the matcher doesn't have to
    sort_idx = np.argsort(all_hashes_arr, kind="stable")
    
    np.savez(
        DB_PATH,
        db_hashes=all_hashes_arr[sort_idx],
        db_songs=all_songs_arr[sort_idx],
        db_anchors=all_anchors_arr[sort_idx],
        song_names=np.array(song_names, dtype=str),
    )

    print(f"\nSuper-fast binary database successfully saved to '{DB_PATH}'")

if __name__ == "__main__":
    main()
