import csv
import os

import config
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram


class AudioFingerprintMatcher:
    def __init__(self, db_file=None):
        self.db_file = db_file or config.DB_FILE
        self.cache_file = self.db_file + ".cache.npz"

        # Mapping for song names to integer IDs to keep NumPy arrays numeric
        self.song_id_to_name = {}
        self.name_to_song_id = {}

        # Per-song aggregates, precomputed once at load time
        self.song_hash_counts = {}
        self.song_peak_counts = {}

        # Attempt warm start from cache, fall back to CSV if modified or missing
        if not self._try_load_cache():
            self._load_database_csv()
            self._save_cache()

    @staticmethod
    def pack_hash(f1, f2, dt):
        """Packs a 3-tuple hash into a single 64-bit integer."""
        return (np.uint64(f1) << 32) | (np.uint64(f2) << 16) | np.uint64(dt)

    def _try_load_cache(self):
        """Attempts to load a pre-compiled binary cache of the database."""
        if not os.path.exists(self.cache_file) or not os.path.exists(self.db_file):
            return False

        try:
            # Validate cache viability against source file status
            csv_mtime = os.path.getmtime(self.db_file)
            csv_size = os.path.getsize(self.db_file)

            data = np.load(self.cache_file, allow_pickle=True)
            if data["csv_mtime"] != csv_mtime or data["csv_size"] != csv_size:
                return False  # Cache is stale

            # Load primary structural arrays
            self.db_hashes = data["db_hashes"]
            self.db_songs = data["db_songs"]
            self.db_anchors = data["db_anchors"]

            # Restore structural dictionaries
            song_names = data["song_names"]
            self.name_to_song_id = {name: i for i, name in enumerate(song_names)}
            self.song_id_to_name = {i: name for i, name in enumerate(song_names)}

            # Rehydrate precomputed statistics fields
            self.song_hash_counts = {
                int(i): int(c) for i, c in enumerate(data["song_hash_counts"])
            }
            self.song_peak_counts = {
                int(i): int(c) for i, c in enumerate(data["song_peak_counts"])
            }
            return True
        except Exception:
            return False

    def _save_cache(self):
        """Saves current database structures to an optimized binary sidecar file."""
        try:
            csv_mtime = os.path.getmtime(self.db_file)
            csv_size = os.path.getsize(self.db_file)

            # Sort maps to guarantee index continuity
            sorted_songs = [
                self.song_id_to_name[i] for i in range(len(self.song_id_to_name))
            ]
            hash_counts_arr = np.array(
                [self.song_hash_counts.get(i, 0) for i in range(len(sorted_songs))]
            )
            peak_counts_arr = np.array(
                [self.song_peak_counts.get(i, 0) for i in range(len(sorted_songs))]
            )

            # Prevent np.savez auto-append bugs using structured temp layouts
            tmp_path = self.cache_file[:-4] + ".tmp.npz"
            np.savez(
                tmp_path,
                db_hashes=self.db_hashes,
                db_songs=self.db_songs,
                db_anchors=self.db_anchors,
                song_names=np.array(sorted_songs, dtype=object),
                song_hash_counts=hash_counts_arr,
                song_peak_counts=peak_counts_arr,
                csv_mtime=csv_mtime,
                csv_size=csv_size,
            )
            os.replace(tmp_path, self.cache_file)
        except Exception:
            pass  # Fail gracefully without breaking production executions

    def _load_database_csv(self):
        """Loads CSV database into flat, highly efficient NumPy arrays."""
        try:
            with open(self.db_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Database file '{self.db_file}' not found. Run indexer.py first."
            )

        if not rows:
            self.db_hashes = np.array([], dtype=np.uint64)
            self.db_songs = np.array([], dtype=np.int32)
            self.db_anchors = np.array([], dtype=np.int32)
            return

        f1_col, f2_col, dt_col, song_col, anchor_col = zip(*rows)

        unique_names, song_idx = np.unique(np.array(song_col), return_inverse=True)
        self.name_to_song_id = {name: int(i) for i, name in enumerate(unique_names)}
        self.song_id_to_name = {int(i): name for i, name in enumerate(unique_names)}

        f1_arr = np.array(f1_col, dtype=np.uint64)
        f2_arr = np.array(f2_col, dtype=np.uint64)
        dt_arr = np.array(dt_col, dtype=np.uint64)

        self.db_hashes = self.pack_hash(f1_arr, f2_arr, dt_arr)
        self.db_songs = song_idx.astype(np.int32)
        self.db_anchors = np.array(anchor_col, dtype=np.int32)

        # CRITICAL: Keep database sorted by hashes for binary searching
        sort_idx = np.argsort(self.db_hashes, kind="stable")
        self.db_hashes = self.db_hashes[sort_idx]
        self.db_songs = self.db_songs[sort_idx]
        self.db_anchors = self.db_anchors[sort_idx]

        self._compute_song_stats()

    def _compute_song_stats(self):
        """Precomputes per-song stats using efficient array operations."""
        n_songs = len(self.song_id_to_name)
        if n_songs == 0 or len(self.db_songs) == 0:
            return

        hash_counts = np.bincount(self.db_songs, minlength=n_songs)
        self.song_hash_counts = {i: int(c) for i, c in enumerate(hash_counts)}

        anchor_offset = int(self.db_anchors.max()) + 1 if len(self.db_anchors) else 1
        combined = self.db_songs.astype(
            np.int64
        ) * anchor_offset + self.db_anchors.astype(np.int64)
        unique_combined = np.unique(combined)
        unique_song_ids = (unique_combined // anchor_offset).astype(np.int32)
        peak_counts = np.bincount(unique_song_ids, minlength=n_songs)
        self.song_peak_counts = {i: int(c) for i, c in enumerate(peak_counts)}

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
        """Generates packed hashes and times as parallel NumPy arrays."""
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

    def generate_spectrogram_constellation_plot(self, S_db, constellation):
        fig, ax = plt.subplots(figsize=(12, 6))
        time_per_frame = (config.WINDOW_SIZE - config.OVERLAP) / config.TARGET_FS
        max_time_sec = S_db.shape[1] * time_per_frame

        cax = ax.imshow(
            S_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, max_time_sec, 0, S_db.shape[0]],
        )
        fig.colorbar(cax, ax=ax, format="%+2.0f dB", label="Magnitude")

        if constellation:
            t_idx, f_idx = zip(*constellation)
            t_sec = [t * time_per_frame for t in t_idx]
            ax.scatter(
                t_sec,
                f_idx,
                facecolors="none",
                edgecolors="#00E5FF",  # Cyan halos
                s=25,  # Size adjusted for distinct circles
                marker="o",
                linewidths=1.2,
                label="Constellation Peaks",
            )
            ax.legend(loc="upper right", framealpha=0.85)

        ax.set_title("Spectrogram with Overlaid Constellation Peaks")
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Frequency (Bin Index)")
        fig.tight_layout()
        return fig

    def generate_constellation_plot(self, constellation, S_db_shape=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        time_per_frame = (config.WINDOW_SIZE - config.OVERLAP) / config.TARGET_FS

        if constellation:
            t_idx, f_idx = zip(*constellation)
            t_sec = [t * time_per_frame for t in t_idx]
            ax.scatter(
                t_sec,
                f_idx,
                facecolors="none",
                edgecolors="black",
                s=20,
                marker="o",
                linewidths=1.0,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No constellation points found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title("Audio Constellation Map")
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Frequency (Bin Index)")
        ax.grid(True, linestyle="--", alpha=0.5)

        if S_db_shape:
            max_time_sec = S_db_shape[1] * time_per_frame
            ax.set_xlim(0, max_time_sec)
            ax.set_ylim(0, S_db_shape[0])

        fig.tight_layout()
        return fig

    def analyze_audio(self, audio_path):
        fs, audio = self.load_audio(audio_path)
        S_db = self.get_spectrogram(audio, fs)
        constellation = self.get_constellation(S_db)
        q_hashes, q_times = self.create_hashes_numpy(constellation)
        return {
            "fs": fs,
            "audio": audio,
            "S_db": S_db,
            "constellation": constellation,
            "q_hashes": q_hashes,
            "q_times": q_times,
        }

    def match_from_analysis(self, analysis):
        """Runs voting/alignment step using a super-optimized searchsorted approach."""
        q_hashes = analysis["q_hashes"]
        q_times = analysis["q_times"]

        if len(q_hashes) == 0:
            return {"match_found": False, "reason": "No query hashes generated"}

        # Sort the query parameters to match searchsorted structures
        q_sort_idx = np.argsort(q_hashes)
        q_hashes = q_hashes[q_sort_idx]
        q_times = q_times[q_sort_idx]

        # SUPER-OPTIMIZATION: Replace full-DB np.isin scan with localized bounds discovery
        left_bounds = np.searchsorted(self.db_hashes, q_hashes, side="left")
        right_bounds = np.searchsorted(self.db_hashes, q_hashes, side="right")

        # Keep segments where a matching signature exists in the database
        valid_mask = right_bounds > left_bounds
        if not np.any(valid_mask):
            return {"match_found": False, "reason": "No matched hashes"}

        l_intervals = left_bounds[valid_mask]
        r_intervals = right_bounds[valid_mask]
        matched_q_times_base = q_times[valid_mask]

        # Vectorized expansion handling exact duplicate-hash semantic distributions
        segment_lengths = r_intervals - l_intervals
        total_matches = segment_lengths.sum()

        # Generate target array indices efficiently without Python looping bottlenecks
        db_indices = np.repeat(l_intervals, segment_lengths) + (
            np.arange(total_matches)
            - np.repeat(np.cumsum(segment_lengths) - segment_lengths, segment_lengths)
        )

        matched_db_songs = self.db_songs[db_indices]
        matched_db_anchors = self.db_anchors[db_indices]
        matched_q_times = np.repeat(matched_q_times_base, segment_lengths)

        # Pre-calculate relative temporal distance matrices
        offsets = matched_db_anchors - matched_q_times

        # Pack unique offset-song keys into an integer space for blazingly fast unique mapping
        combined_votes = (offsets.astype(np.int64) << 32) | matched_db_songs.astype(
            np.int64
        )

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

        hop_size = config.WINDOW_SIZE - config.OVERLAP
        time_per_frame = hop_size / config.TARGET_FS
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
