import json
import os
import tempfile
from collections import Counter

import config
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matcher import AudioFingerprintMatcher

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="EE200: Audio Fingerprinting",
    page_icon="🎧",
    layout="wide",
)

# Custom injection for performance skinning and layout pinning
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        padding: 0 18px;
        border-radius: 8px 8px 0 0;
        font-size: 15px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.15);
        border-radius: 10px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: box-shadow 0.15s ease, transform 0.15s ease;
        border-radius: 10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        transform: translateY(-1px);
    }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(120,120,120,0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- 2. CACHED RESOURCES & PIPELINES ---
@st.cache_resource(show_spinner="Loading fingerprint database...")
def get_matcher():
    return AudioFingerprintMatcher()


@st.cache_data(show_spinner=False)
def analyze_song_cached(song_path: str, _mtime: float):
    fs, audio = matcher.load_audio(song_path)
    S_db = matcher.get_spectrogram(audio, fs)
    constellation = matcher.get_constellation(S_db)
    return fs, audio, S_db, constellation


# Cache the metadata payload completely to drop grid rendering cost to absolute zero
@st.cache_data(show_spinner=False)
def get_precomputed_metadata(_matcher_obj):
    """Pre-processes structures to prevent dictionary lookups or list sorting
    overhead inside the rendering loops."""
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
    # Extract structural components out of the loop lifecycle
    indexed_songs, metadata_lookup = get_precomputed_metadata(matcher)
except FileNotFoundError as e:
    matcher = None
    db_load_error = str(e)


# --- 3. STATE CALLBACK MANAGEMENT ---
def _open_song_detail(song_name):
    st.session_state["inspected_song"] = song_name


def _close_song_detail():
    st.session_state["inspected_song"] = None


# Initialize session state explicitly to prevent framework tracking drops
if "inspected_song" not in st.session_state:
    st.session_state["inspected_song"] = None


# --- 4. UI LAYOUT ---
st.title("🎧 EE200: Audio Fingerprinting")
st.caption(
    "Shazam-style landmark fingerprinting — index songs, then identify clips against the database."
)

if db_load_error:
    st.error(db_load_error)
    st.stop()

with st.sidebar:
    st.subheader("Database Overview")
    st.metric("Indexed Tracks", f"{len(matcher.name_to_song_id):,}")
    st.metric("Total Hash Records", f"{len(matcher.db_hashes):,}")
    st.markdown("---")
    st.caption(
        f"Window {config.WINDOW_SIZE} · Overlap {config.OVERLAP} · "
        f"Fan-out {config.FAN_OUT} · Vote threshold {config.MIN_VOTE_THRESHOLD}"
    )

tab_library, tab_identify = st.tabs(["📚 Library", "🔍 Identify"])

with tab_library:
    current_inspect = st.session_state["inspected_song"]

    if not indexed_songs:
        st.info(
            "The fingerprint database is currently empty. Please run `indexer.py` first."
        )
    else:
        if current_inspect is not None:
            # --- SCREEN 2: DETAILED VISUAL INSPECTION ---
            selected_song = current_inspect
            song_id = metadata_lookup[selected_song]["id"]

            b_col1, b_col2 = st.columns([1, 5])
            with b_col1:
                # FIXED: Force layout tree synchronization by matching callback signature
                st.button(
                    "⬅️ Back to Library",
                    key="global_back_to_lib_btn",
                    use_container_width=True,
                    on_click=_close_song_detail,
                )
            with b_col2:
                st.markdown(f"### 🔍 Detailed Visual Inspection: **{selected_song}**")

            st.markdown("---")
            song_dir = getattr(config, "SONG_DIR", ".")
            song_path = os.path.join(song_dir, selected_song)

            if os.path.exists(song_path):
                st.audio(song_path)

                with st.spinner("Processing dual spectral imaging graphs..."):
                    mtime = os.path.getmtime(song_path)
                    fs_song, audio_song, S_db_song, constellation_song = (
                        analyze_song_cached(song_path, mtime)
                    )

                    total_hashes_in_db = matcher.song_hash_counts.get(song_id, 0)
                    total_peaks_in_db = matcher.song_peak_counts.get(song_id, 0)
                    duration_secs = len(audio_song) / fs_song
                    mins, secs = int(duration_secs // 60), duration_secs % 60

                m1, m2, m3 = st.columns(3)
                m1.metric("Database Footprint Size", f"{total_hashes_in_db:,} Hashes")
                m2.metric("Total Constellation Nodes", f"{total_peaks_in_db:,} Peaks")
                m3.metric("Audio Content Runtime", f"{mins:02d}:{secs:04.1f}")

                st.markdown("#### Side-by-Side Analysis Profiles")
                p_col1, p_col2 = st.columns(2)

                with p_col1:
                    st.markdown("**1. Raw Spectrogram Profile (Signal Density)**")
                    time_per_frame = (
                        config.WINDOW_SIZE - config.OVERLAP
                    ) / config.TARGET_FS
                    max_time_sec = S_db_song.shape[1] * time_per_frame

                    fig_raw, ax_raw = plt.subplots(
                        figsize=(8, 4)
                    )  # Reduced sizing slightly for DOM acceleration
                    cax_raw = ax_raw.imshow(
                        S_db_song,
                        aspect="auto",
                        origin="lower",
                        cmap="magma",
                        extent=[0, max_time_sec, 0, S_db_song.shape[0]],
                    )
                    fig_raw.colorbar(
                        cax_raw, ax=ax_raw, format="%+2.0f dB", label="Magnitude"
                    )
                    ax_raw.set_xlabel("Time (Seconds)")
                    ax_raw.set_ylabel("Frequency (Bin Index)")
                    fig_raw.tight_layout()
                    st.pyplot(
                        fig_raw, clear_figure=True
                    )  # Accelerated rendering memory release

                with p_col2:
                    st.markdown(
                        "**2. Structural Map (Spectrogram + Overlaid Landmarks)**"
                    )
                    fig_peaks = matcher.generate_spectrogram_constellation_plot(
                        S_db_song, constellation_song
                    )
                    fig_peaks.set_size_inches(8, 4)
                    st.pyplot(fig_peaks, clear_figure=True)
            else:
                st.warning(
                    f"⚠️ Track binary missing. Audio file `{selected_song}` wasn't found at `{song_path}`."
                )

        else:
            # --- SCREEN 1: THE MAIN LIBRARY CARD GRID ---
            st.markdown(
                f"Database contains **{len(matcher.db_hashes):,}** total hash records across **{len(indexed_songs)}** tracks."
            )
            st.markdown("---")

            cols_per_row = 4
            # Uses precomputed arrays to eliminate loop pipeline execution stalls
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

                            # Explicit callback bindings ensure immediate UI tracking mutations
                            st.button(
                                "Inspect Visuals",
                                key=f"btn_inspect_{meta['id']}",
                                use_container_width=True,
                                on_click=_open_song_detail,
                                args=(song,),
                            )

with tab_identify:
    st.subheader("Identify a Clip")
    uploaded_file = st.file_uploader(
        "Upload an audio sample", type=["wav", "mp3", "m4a"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button(
            "Identify Audio",
            type="primary",
            use_container_width=True,
            key="exec_id_btn",
        ):
            with st.spinner("Analyzing frequencies..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    match_result = matcher.match(tmp_path)
                    analysis = match_result["analysis"]
                    S_db = analysis["S_db"]
                    constellation = analysis["constellation"]
                    query_hashes = analysis["q_hashes"]

                    votes = Counter()
                    if match_result["match_found"]:
                        votes[
                            (match_result["song_name"], match_result["offset_frames"])
                        ] = match_result["confidence"]
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            st.markdown("---")
            if not match_result["match_found"]:
                st.error(
                    f"No Match Found ({match_result.get('reason', '0 matched hashes')})"
                )
            else:
                song_name = match_result["song_name"]
                offset_frames = match_result["offset_frames"]
                score = match_result["confidence"]
                offset_seconds = match_result["offset_seconds"]

                mins, secs = int(offset_seconds // 60), offset_seconds % 60
                st.success(f"### MATCH FOUND: **{song_name}**")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Confidence", f"{score} votes")
                m2.metric("Timestamp", f"{mins:02d}:{secs:05.2f}")
                m3.metric("Peaks Found (Query)", len(constellation))
                m4.metric("Hashes Generated (Query)", len(query_hashes))

                st.write("### Diagnostics Pipeline")
                time_per_frame = (
                    config.WINDOW_SIZE - config.OVERLAP
                ) / config.TARGET_FS
                query_time_seconds = S_db.shape[1] * time_per_frame

                st.markdown("#### 1. Input Clip: Spectrogram & Constellation")
                fig1 = matcher.generate_spectrogram_constellation_plot(
                    S_db, constellation
                )
                st.pyplot(fig1, clear_figure=True)

                base_name = os.path.splitext(song_name)[0]
                asset_data_path = os.path.join("assets", f"{base_name}_data.json")
                asset_png_path = os.path.join(
                    "assets", f"{base_name}_constellation.png"
                )

                if os.path.exists(asset_data_path):
                    st.markdown("#### 2. Matched Song: Constellation Map")
                    st.image(asset_png_path, use_container_width=True)

                    with open(asset_data_path, "r") as json_f:
                        asset_data = json.load(json_f)

                    s_db_shape = asset_data["shape"]
                    constellation_song = asset_data["constellation"]

                    st.markdown("#### 3. Alignment Overlay: Clip vs Song")
                    fig3, ax3 = plt.subplots(figsize=(12, 5))

                    if constellation_song:
                        t_s_idx, f_s = zip(*constellation_song)
                        t_s_sec = [t * time_per_frame for t in t_s_idx]
                        ax3.scatter(
                            t_s_sec,
                            f_s,
                            color="lightgrey",
                            s=10,
                            marker="o",
                            label="Song Constellation (Asset)",
                        )

                    if constellation:
                        t_q_idx, f_q = zip(*constellation)
                        t_q_sec_shifted = [
                            (t * time_per_frame) + offset_seconds for t in t_q_idx
                        ]
                        # CHANGED: marker="o", facecolors="none" (or transparent), edgecolors="red"
                        ax3.scatter(
                            t_q_sec_shifted,
                            f_q,
                            facecolors="none",  # Empty circle center
                            edgecolors="#FF3366",  # Crisp neon ring color
                            s=30,  # Slightly larger area size so the circle is visible
                            marker="o",
                            linewidths=1.5,  # Thicker ring bounds
                            label="Aligned Clip Alignment",
                        )

                    ax3.axvspan(
                        offset_seconds,
                        offset_seconds + query_time_seconds,
                        color="yellow",
                        alpha=0.2,
                        label="Matched Time Window",
                    )
                    ax3.set_title("Constellation Overlay & Time Alignment")
                    ax3.set_xlabel("Time (Seconds)")
                    ax3.set_ylabel("Frequency (Bin Index)")
                    ax3.legend(loc="upper right")
                    ax3.grid(True, linestyle="--", alpha=0.5)

                    max_song_time_sec = s_db_shape[1] * time_per_frame
                    ax3.set_xlim(0, max_song_time_sec)
                    ax3.set_ylim(0, s_db_shape[0])

                    fig3.tight_layout()
                    st.pyplot(fig3, clear_figure=True)
                else:
                    st.warning(
                        f"⚠️ **Precomputed asset files missing.** Expected `{asset_data_path}`. Run `indexer.py` to create it."
                    )

                st.markdown("#### 4. Time Alignment Histogram")
                fig4, ax4 = plt.subplots(figsize=(10, 3))
                offsets_seconds_list = [
                    (off * time_per_frame)
                    for (_, off), count in votes.items()
                    for _ in range(count)
                ]
                ax4.hist(offsets_seconds_list, bins=50, color="crimson", alpha=0.7)
                ax4.set_title("Time Alignment Offsets Histogram")
                ax4.set_xlabel("Time Offset (Seconds)")
                ax4.set_ylabel("Match Count")
                fig4.tight_layout()
                st.pyplot(fig4, clear_figure=True)
