import os
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram
import config

class AudioFingerprintMatcher:
    def __init__(self, db_file=None):
        self.db_file = db_file or config.DB_FILE.replace('.csv', '.npz')
        
        self.song_id_to_name = {}
        self.name_to_song_id = {}
        self.song_hash_counts = {}
        self.song_peak_counts = {}

        self._load_database()

    @staticmethod
    def pack_hash(f1, f2, dt):
        return (np.uint64(f1) << 32) | (np.uint64(f2) << 16) | np.uint64(dt)

    def _load_database(self):
        """Memory-maps the pre-sorted NumPy binary for instant loading."""
        if not os.path.exists(self.db_file):
            raise FileNotFoundError(f"Binary DB '{self.db_file}' not found. Run indexer.py.")

        # mmap_mode='r' makes initialization O(1) time complexity
        data = np.load(self.db_file, mmap_mode='r')
        
        self.db_hashes = data["db_hashes"]
        self.db_songs = data["db_songs"]
        self.db_anchors = data["db_anchors"]
        song_names = data["song_names"]

        self.name_to_song_id = {str(name): i for i, name in enumerate(song_names)}
        self.song_id_to_name = {i: str(name) for i, name in enumerate(song_names)}

        self._compute_song_stats()

    def _compute_song_stats(self):
        n_songs = len(self.song_id_to_name)
        if n_songs == 0 or len(self.db_songs) == 0:
            return

        hash_counts = np.bincount(self.db_songs, minlength=n_songs)
        self.song_hash_counts = {i: int(c) for i, c in enumerate(hash_counts)}

    @staticmethod
    def load_audio(path):
        audio, fs = librosa.load(path, sr=config.TARGET_FS, mono=True)
        return fs, audio

    @staticmethod
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

    @staticmethod
    def get_constellation(S_db):
        local_max = maximum_filter(S_db, size=config.PEAK_SIZE) == S_db
        threshold = np.percentile(S_db, config.PEAK_THRESHOLD)
        peaks = np.argwhere(local_max & (S_db > threshold))
        constellation = [(int(t_idx), int(f_idx)) for f_idx, t_idx in peaks]
        constellation.sort()
        return constellation

    def create_hashes_numpy(self, constellation):
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
                hashes.append(self.pack_hash(f1, f2, dt))
                times.append(t1)

        return np.array(hashes, dtype=np.uint64), np.array(times, dtype=np.int32)

    def analyze_audio(self, audio_path):
        fs, audio = self.load_audio(audio_path)
        S_db = self.get_spectrogram(audio, fs)
        constellation = self.get_constellation(S_db)
        q_hashes, q_times = self.create_hashes_numpy(constellation)
        return {
            "constellation_len": len(constellation),
            "q_hashes": q_hashes,
            "q_times": q_times,
        }

    def match_from_analysis(self, analysis):
        q_hashes = analysis["q_hashes"]
        q_times = analysis["q_times"]

        if len(q_hashes) == 0:
            return {"match_found": False, "reason": "No query hashes generated"}

        q_sort_idx = np.argsort(q_hashes)
        q_hashes = q_hashes[q_sort_idx]
        q_times = q_times[q_sort_idx]

        left_bounds = np.searchsorted(self.db_hashes, q_hashes, side="left")
        right_bounds = np.searchsorted(self.db_hashes, q_hashes, side="right")

        valid_mask = right_bounds > left_bounds
        if not np.any(valid_mask):
            return {"match_found": False, "reason": "No matched hashes"}

        l_intervals = left_bounds[valid_mask]
        r_intervals = right_bounds[valid_mask]
        matched_q_times_base = q_times[valid_mask]

        segment_lengths = r_intervals - l_intervals
        total_matches = segment_lengths.sum()

        db_indices = np.repeat(l_intervals, segment_lengths) + (
            np.arange(total_matches)
            - np.repeat(np.cumsum(segment_lengths) - segment_lengths, segment_lengths)
        )

        matched_db_songs = self.db_songs[db_indices]
        matched_db_anchors = self.db_anchors[db_indices]
        matched_q_times = np.repeat(matched_q_times_base, segment_lengths)

        offsets = matched_db_anchors - matched_q_times
        combined_votes = (offsets.astype(np.int64) << 32) | matched_db_songs.astype(np.int64)

        unique_keys, counts = np.unique(combined_votes, return_counts=True)
        best_idx = np.argmax(counts)

        score = counts[best_idx]
        best_key = unique_keys[best_idx]

        best_song_id = int(best_key & 0xFFFFFFFF)
        best_offset_frames = int(best_key >> 32)
        song_name = self.song_id_to_name[best_song_id]

        if score < config.MIN_VOTE_THRESHOLD:
            return {
                "match_found": False,
                "reason": f"Top match '{song_name}' had {score} votes (Below threshold)",
            }

        time_per_frame = (config.WINDOW_SIZE - config.OVERLAP) / config.TARGET_FS
        offset_seconds = best_offset_frames * time_per_frame

        return {
            "match_found": True,
            "song_name": song_name,
            "confidence": int(score),
            "offset_seconds": float(offset_seconds),
            "offset_frames": int(best_offset_frames),
        }

    def match(self, query_path):
        analysis = self.analyze_audio(query_path)
        result = self.match_from_analysis(analysis)
        result["analysis"] = analysis
        return result
