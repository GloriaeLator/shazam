import os
import tempfile
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import config
from matcher import AudioFingerprintMatcher

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="EE200: Audio Fingerprinting",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.15);
        border-radius: 10px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. CACHED RESOURCES ---
@st.cache_resource(show_spinner="Mapping binary database...")
def get_matcher():
    return AudioFingerprintMatcher()

@st.cache_data(show_spinner=False)
def get_precomputed_metadata(_matcher_obj):
    indexed_songs = sorted(list(_matcher_obj.name_to_song_id.keys()))
    song_metadata = {}
    for song in indexed_songs:
        s_id = _matcher_obj.name_to_song_id[song]
        song_metadata[song] = {
            "id": s_id,
            "hashes": _matcher_obj.song_hash_counts.get(s_id, 0),
            "display_title": os.path.splitext(song)[0].replace("_", " ").title(),
        }
    return indexed_songs, song_metadata

try:
    matcher = get_matcher()
    db_load_error = None
    indexed_songs, metadata_lookup = get_precomputed_metadata(matcher)
except FileNotFoundError as e:
    matcher = None
    db_load_error = str(e)

if db_load_error:
    st.error(db_load_error)
    st.stop()

# Shared calculation constants
hop_size = config.WINDOW_SIZE - config.OVERLAP
time_per_frame = hop_size / config.TARGET_FS
freq_per_bin = config.TARGET_FS / config.WINDOW_SIZE


# --- 3. DYNAMIC ROUTING (VIRTUAL TRACK PAGE) ---
query_params = st.query_params
if "track" in query_params:
    track_name = query_params["track"]
    
    if track_name not in metadata_lookup:
        st.error(f"Track '{track_name}' not found in the indexed database.")
        if st.button("⬅️ Back to Main Dashboard"):
            st.query_params.clear()
            st.rerun()
        st.stop()
        
    meta = metadata_lookup[track_name]
    
    # Render Virtual Track Page Header
    st.title(f"🎵 {meta['display_title']}")
    st.caption(f"File Source: {track_name} · Unique Fingerprint Profile")
    
    if st.button("⬅️ Back to Main Dashboard", type="secondary"):
        st.query_params.clear()
        st.rerun()
        
    st.markdown("---")
    
    # Compute constellation map on-the-fly for this track
    target_song_path = os.path.join(config.SONG_DIR, track_name)
    if not os.path.exists(target_song_path):
        st.error(f"Audio asset missing from filesystem. Target missing at: `{target_song_path}`")
    else:
        with st.spinner("Parsing frequency profiles and rendering full track matrix..."):
            try:
                t_fs, t_audio = matcher.load_audio(target_song_path)
                t_S_db = matcher.get_spectrogram(t_audio, t_fs)
                t_constellation = matcher.get_constellation(t_S_db)
                
                # Metrics block
                c1, c2, c3 = st.columns(3)
                c1.metric("Database Index ID", f"#{meta['id']}")
                c2.metric("Total Constellation Landmarks", f"{len(t_constellation):,} peaks")
                c3.metric("Compiled Combinatorial Hashes", f"{meta['hashes']:,} pairs")
                
                # Render Full Screen Constellation Graph
                st.markdown("### 📊 Complete Spectral Landmark Profile")
                if t_constellation:
                    t_times = [pt[0] * time_per_frame for pt in t_constellation]
                    t_freqs = [pt[1] * freq_per_bin for pt in t_constellation]

                    fig3, ax_full = plt.subplots(figsize=(14, 5))
                    ax_full.set_facecolor("#111111")
                    fig3.patch.set_facecolor("#111111")
                    ax_full.tick_params(colors="white")
                    ax_full.xaxis.label.set_color("white")
                    ax_full.yaxis.label.set_color("white")
                    ax_full.title.set_color("white")
                    ax_full.grid(True, linestyle="--", alpha=0.12, color="gray")

                    ax_full.scatter(t_times, t_freqs, c="#4facfe", s=1.2, alpha=0.6, edgecolors="none")
                    ax_full.set_title(f"Landmark Energy Distribution Grid — {meta['display_title']}")
                    ax_full.set_xlabel("Time (Seconds)")
                    ax_full.set_ylabel("Frequency (Hz)")
                    
                    fig3.tight_layout()
                    st.pyplot(fig3, clear_figure=True)
                else:
                    st.warning("No significant high-energy constellation coordinates could be mapped.")
                    
            except Exception as e:
                st.error(f"Failed to process track landmarks: {e}")
    st.stop()


# --- 4. MAIN DASHBOARD APPLICATION LAYOUT ---
st.title("🎧 EE200: Audio Fingerprinting")
st.caption("High-performance Shazam-style landmark identification pipeline.")

with st.sidebar:
    st.subheader("Database Metrics")
    st.metric("Indexed Tracks", f"{len(matcher.name_to_song_id):,}")
    st.metric("Total Hash Records", f"{len(matcher.db_hashes):,}")
    st.markdown("---")
    st.caption(
        f"Window {config.WINDOW_SIZE} · Overlap {config.OVERLAP} · "
        f"Fan-out {config.FAN_OUT} · Vote threshold {config.MIN_VOTE_THRESHOLD}"
    )

tab_identify, tab_library = st.tabs(["🔍 Identify", "📚 Database Library"])

with tab_identify:
    st.subheader("Identify a Clip")
    uploaded_file = st.file_uploader("Upload an audio sample", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Identify Audio", type="primary", use_container_width=True):
            with st.spinner("Analyzing frequencies & matching landmarks..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    match_result = matcher.match(tmp_path)
                    analysis = match_result["analysis"]
                    
                    q_fs, q_audio = matcher.load_audio(tmp_path)
                    q_S_db = matcher.get_spectrogram(q_audio, q_fs)
                    q_constellation = matcher.get_constellation(q_S_db)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            st.markdown("---")
            if not match_result["match_found"]:
                st.error(f"No Match Found ({match_result.get('reason', '0 matched hashes')})")
            else:
                song_name = match_result["song_name"]
                score = match_result["confidence"]
                offset_seconds = match_result["offset_seconds"]
                mins, secs = int(offset_seconds // 60), offset_seconds % 60

                st.success(f"### MATCH FOUND: **{song_name}**")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Confidence Score", f"{score} votes")
                m2.metric("Timestamp Alignment", f"{mins:02d}:{secs:05.2f}")
                m3.metric("Query Peaks", len(q_constellation))
                m4.metric("Hashes Evaluated", len(analysis["q_hashes"]))

                # --- GRAPH 1: SIDE BY SIDE ---
                st.markdown("#### 📊 Constellation Target Analysis")
                target_song_path = os.path.join(config.SONG_DIR, song_name)
                t_constellation = []
                if os.path.exists(target_song_path):
                    try:
                        t_fs, t_audio = matcher.load_audio(target_song_path)
                        t_S_db = matcher.get_spectrogram(t_audio, t_fs)
                        t_constellation = matcher.get_constellation(t_S_db)
                    except Exception:
                        pass

                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                for ax in [ax1, ax2]:
                    ax.set_facecolor("#111111")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    ax.title.set_color("white")
                    ax.grid(True, linestyle="--", alpha=0.15, color="gray")

                if q_constellation:
                    q_times = [pt[0] * time_per_frame for pt in q_constellation]
                    q_freqs = [pt[1] * freq_per_bin for pt in q_constellation]
                    ax1.scatter(q_times, q_freqs, c="#00f2fe", s=3, alpha=0.8, edgecolors="none")
                ax1.set_title("Query Clip Constellation")
                ax1.set_xlabel("Time (Seconds)")
                ax1.set_ylabel("Frequency (Hz)")

                if t_constellation:
                    t_times = [pt[0] * time_per_frame for pt in t_constellation]
                    t_freqs = [pt[1] * freq_per_bin for pt in t_constellation]
                    ax2.scatter(t_times, t_freqs, c="#4facfe", s=1.5, alpha=0.5, edgecolors="none")
                    ax2.axvspan(offset_seconds, offset_seconds + (len(q_audio)/q_fs), color='red', alpha=0.15, label='Matched Window')
                    ax2.legend(facecolor='#111111', edgecolor='none', labelcolor='white')
                ax2.set_title(f"Full Track Constellation ({song_name})")
                ax2.set_xlabel("Time (Seconds)")
                ax2.set_ylabel("Frequency (Hz)")

                fig1.patch.set_facecolor("#111111")
                fig1.tight_layout()
                st.pyplot(fig1, clear_figure=True)

                # --- GRAPH 2: TIME SLIDE HISTOGRAM ---
                st.markdown("#### ⏳ Time Span Overlap Alignment (Vote Convergence)")
                q_hashes = analysis["q_hashes"]
                q_times = analysis["q_times"]
                
                left_bounds = np.searchsorted(matcher.db_hashes, q_hashes, side="left")
                right_bounds = np.searchsorted(matcher.db_hashes, q_hashes, side="right")
                valid_mask = right_bounds > left_bounds
                
                if np.any(valid_mask):
                    l_intervals = left_bounds[valid_mask]
                    r_intervals = right_bounds[valid_mask]
                    matched_q_times_base = q_times[valid_mask]
                    segment_lengths = r_intervals - l_intervals
                    
                    db_indices = np.repeat(l_intervals, segment_lengths) + (
                        np.arange(segment_lengths.sum()) - np.repeat(np.cumsum(segment_lengths) - segment_lengths, segment_lengths)
                    )
                    
                    target_song_id = matcher.name_to_song_id[song_name]
                    song_mask = matcher.db_songs[db_indices] == target_song_id
                    
                    offsets_frames = matcher.db_anchors[db_indices][song_mask] - np.repeat(matched_q_times_base, segment_lengths)[song_mask]
                    offsets_seconds = offsets_frames * time_per_frame

                    fig2, ax3 = plt.subplots(figsize=(14, 3.5))
                    ax3.set_facecolor("#111111")
                    fig2.patch.set_facecolor("#111111")
                    ax3.tick_params(colors="white")
                    ax3.xaxis.label.set_color("white")
                    ax3.yaxis.label.set_color("white")
                    ax3.title.set_color("white")
                    
                    counts, bins, patches = ax3.hist(offsets_seconds, bins=100, color="#ff0844", alpha=0.6, edgecolor='none')
                    max_idx = np.argmax(counts)
                    patches[max_idx].set_facecolor('#ffb199')
                    patches[max_idx].set_alpha(1.0)
                    
                    ax3.axvline(offset_seconds, color="#ffb199", linestyle=":", alpha=0.8, label=f"True Track Start ({mins:02d}:{secs:04.1f}s)")
                    ax3.set_title("Time-Offset Cohort Scatter (Histogram Alignment Peak)")
                    ax3.set_xlabel("Relative Timeline Shift (Seconds Offset)")
                    ax3.set_ylabel("Hash Match Density")
                    ax3.grid(True, linestyle="--", alpha=0.1, color="gray")
                    ax3.legend(facecolor='#111111', edgecolor='none', labelcolor='white')
                    
                    fig2.tight_layout()
                    st.pyplot(fig2, clear_figure=True)

with tab_library:
    if not indexed_songs:
        st.info("The fingerprint database is currently empty. Please run `indexer.py` first.")
    else:
        cols_per_row = 4
        for i in range(0, len(indexed_songs), cols_per_row):
            columns = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j >= len(indexed_songs):
                    continue
                song = indexed_songs[i + j]
                meta = metadata_lookup[song]

                with columns[j]:
                    with st.container(border=True):
                        st.markdown(f"**{meta['display_title']}**")
                        st.caption(f"{meta['hashes']:,} hashes stored")
                        
                        # Use a query parameter rewrite action to cleanly handle standalone scene navigation
                        if st.button("📊 View Profile Mapping", key=f"btn_{meta['id']}", use_container_width=True):
                            st.query_params[ "track" ] = song
                            st.rerun()
