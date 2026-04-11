import streamlit as st
import cv2
import os
import sys
import pickle
import tempfile
import subprocess
import numpy as np
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tennis Analysis System",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Header */
  .app-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 12px;
    padding: 20px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .app-header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: 1px;
  }
  .app-header p {
    color: #a8c8d8;
    margin: 4px 0 0 0;
    font-size: 0.85rem;
  }

  /* Metric cards */
    .metric-card {
        background: #F8F8FF;
        border: 1px solid #F8F8FF;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
  .metric-card .label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #7b8ab0;
    margin-bottom: 6px;
  }
  .metric-card .value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #4fc3f7;
    line-height: 1;
  }
  .metric-card .unit {
    font-size: 0.75rem;
    color: #7b8ab0;
    margin-top: 2px;
  }
  .metric-card.p2 .value { color: #f06292; }

  /* Slot card (placeholder) */
  .slot-card {
    background: #F8F8FF;
    border: 1px solid #F8F8FF;
    border-radius: 10px;
    padding: 24px 16px;
    text-align: center;
    color: #3a4560;
    font-size: 0.8rem;
  }
  .slot-card .slot-icon { font-size: 1.8rem; margin-bottom: 8px; }

  /* Section title */
  .section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a3a6b;
    letter-spacing: 0.5px;
    border-left: 3px solid #4fc3f7;
    padding-left: 10px;
    margin: 20px 0 12px 0;
  }

  /* Log area */
  .log-box {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    font-family: monospace;
    color: #8b9cc8;
    max-height: 200px;
    overflow-y: auto;
  }
  .log-box .ok  { color: #56d364; }
  .log-box .run { color: #f0c040; }
  .log-box .err { color: #f85149; }

  /* Streamlit overrides */
  div[data-testid="stVideo"] { border-radius: 8px; overflow: hidden; }
  .stButton > button {
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 10px 20px;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* Heatmap container */
  .heatmap-container {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 16px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>🎾</div>
  <div>
    <h1>Tennis Analysis System</h1>
    <p>Player tracking · Court detection · Speed estimation · Heatmap</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def get_video_id(filename: str) -> str:
    import re
    stem = Path(filename).stem
    match = re.search(r'(\d+)$', stem)
    return match.group(1) if match else stem


def get_stub_paths(video_stem: str, stub_dir: str = "tracker_stubs"):
    return {
        "player": os.path.join(stub_dir, f"player_detections_{video_stem}.pkl"),
        "ball":   os.path.join(stub_dir, f"ball_detections_{video_stem}.pkl"),
    }

def stub_exists(video_stem: str, stub_dir: str = "tracker_stubs") -> dict:
    paths = get_stub_paths(video_stem, stub_dir)
    return {k: os.path.exists(v) for k, v in paths.items()}

def convert_avi_to_mp4(avi_path: str, mp4_path: str) -> bool:
    try:
        import imageio
        reader = imageio.get_reader(avi_path)
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264',
                                    pixelformat='yuv420p', macro_block_size=1)
        for frame in reader:
            writer.append_data(frame)
        reader.close()
        writer.close()
        return os.path.getsize(mp4_path) > 0
    except Exception as e:
        return False

def calculate_speed_stats(speeds: list) -> dict:
    """Tính avg/max speed cho từng player từ list speeds."""
    stats = {1: {"all": []}, 2: {"all": []}}
    for frame_speeds in speeds:
        for pid, spd in frame_speeds.items():
            if pid in stats:
                stats[pid]["all"].append(spd)
    result = {}
    for pid in [1, 2]:
        vals = stats[pid]["all"]
        if vals:
            result[pid] = {
                "avg": round(float(np.mean(vals)), 1),
                "max": round(float(np.max(vals)), 1),
            }
        else:
            result[pid] = {"avg": 0.0, "max": 0.0}
    return result

def run_pipeline(input_video_path: str, video_stem: str,
                 conf: float, use_polygon: bool,
                 log_placeholder, output_dir: str = "output_videos"):
    """
    Chạy toàn bộ pipeline và trả về:
      (output_mp4_path, heatmap_path, speed_stats)
    """
    os.makedirs(output_dir, exist_ok=True)

    logs = []
    def log(msg: str, kind: str = "ok"):
        logs.append(f'<span class="{kind}">{msg}</span>')
        log_placeholder.markdown(
            '<div class="log-box">' + "<br>".join(logs) + "</div>",
            unsafe_allow_html=True
        )

    # ── Import project modules (same directory) ───────────────────────────
    try:
        from utils import read_video, save_video
        from trackers import PlayerTracker, BallTracker
        from court_line_detector import CourtLineDetector
        from mini_court import MiniCourt
        from speed_estimator import SpeedEstimator
        from heatmap import HeatmapGenerator
    except ImportError as e:
        log(f"❌ Import error: {e}", "err")
        return None, None, None

    # ── Load video ────────────────────────────────────────────────────────
    log("⏳ Loading video frames...", "run")
    video_frames = read_video(input_video_path)
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    log(f"✓ Loaded {len(video_frames)} frames  |  FPS: {fps:.1f}")

    vid_id = get_video_id(video_stem)
    stub_paths = get_stub_paths(vid_id)
    stubs = stub_exists(vid_id)

    # ── Player detection ──────────────────────────────────────────────────
    if stubs["player"]:
        log(f"✓ Player stub found → loading pkl", "ok")
    else:
        log("⏳ Running player detection (YOLO)...", "run")

    player_tracker = PlayerTracker(model_path="models/yolo26x.pt", use_polygon=use_polygon)
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=stubs["player"],
        stub_path=stub_paths["player"],
    )
    log("✓ Player detections ready")

    # ── Ball detection ────────────────────────────────────────────────────
    if stubs["ball"]:
        log("✓ Ball stub found → loading pkl", "ok")
    else:
        log("⏳ Running ball detection (YOLO)...", "run")

    ball_tracker = BallTracker(model_path="models/yolo26m_best_100e.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=stubs["ball"],
        stub_path=stub_paths["ball"],
    )
    log("✓ Ball detections ready")

    # ── Court keypoints ───────────────────────────────────────────────────
    log("⏳ Detecting court keypoints (ResNet50)...", "run")
    court_detector = CourtLineDetector(model_path="models/keypoints_model_04.pth")
    court_keypoints = court_detector.predict(video_frames[0])
    log("✓ Court keypoints detected")

    # ── Filter players ────────────────────────────────────────────────────
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    log("✓ Players filtered")

    # ── Mini court + homography ───────────────────────────────────────────
    mini_court = MiniCourt(video_frames[0])
    mini_court.set_homography(court_keypoints)
    player_mini, ball_mini = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints
    )
    log("✓ Mini court coordinates computed")

    # ── Speed estimation ──────────────────────────────────────────────────
    speed_estimator = SpeedEstimator(fps=fps)
    speeds = speed_estimator.calculate_speed(player_mini, mini_court.get_width_of_mini_court())
    speed_stats = calculate_speed_stats(speeds)
    log("✓ Speed estimation done")
    log("⏳ Counting shots per player...", "run")
    frame_nums_with_ball_hits = ball_tracker.get_ball_shot_frames(ball_detections)
    shot_count = ball_tracker.get_shot_count_per_player(
        frame_nums_with_ball_hits, player_detections, ball_detections
    )
    log(f"✓ Shot count — P1: {shot_count[1]}, P2: {shot_count[2]}")

    # ── Draw output frames ────────────────────────────────────────────────
    log("⏳ Rendering output video...", "run")
    out_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    out_frames = ball_tracker.draw_bboxes(out_frames, ball_detections)
    out_frames = court_detector.draw_keypoints_on_video(out_frames, court_keypoints)
    out_frames = mini_court.draw_mini_court(out_frames)
    out_frames = mini_court.draw_points_on_mini_court(out_frames, player_mini)
    out_frames = mini_court.draw_points_on_mini_court(out_frames, ball_mini, color=(0, 255, 255))
    out_frames = speed_estimator.draw_speed_on_frames(out_frames, speeds, player_detections)
    for i, frame in enumerate(out_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    avi_path = os.path.join(output_dir, f"output_{video_stem}.avi")
    mp4_path = os.path.join(output_dir, f"output_{video_stem}.mp4")
    save_video(out_frames, avi_path)
    log("✓ AVI saved — converting to MP4...", "run")

    ok = convert_avi_to_mp4(avi_path, mp4_path)
    if ok:
        log("✓ MP4 ready")
    else:
        log("⚠️ ffmpeg not found — using AVI path directly", "run")
        mp4_path = avi_path

    # ── Heatmap ───────────────────────────────────────────────────────────
    log("⏳ Generating heatmap...", "run")
    heatmap_gen = HeatmapGenerator(
        court_width=mini_court.court_end_x - mini_court.court_start_x,
        court_height=mini_court.court_end_y - mini_court.court_start_y,
    )
    heatmap_frame = heatmap_gen.draw_heatmap_on_last_frame(out_frames, player_mini, mini_court)
    heatmap_path = os.path.join(output_dir, f"heatmap_{video_stem}.png")
    cv2.imwrite(heatmap_path, heatmap_frame)
    log("✓ Heatmap saved")
    log("🎾 Analysis complete!", "ok")

    return mp4_path, heatmap_path, speed_stats, shot_count


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Input Video")
    uploaded = st.file_uploader("Upload video (.mp4)", type=["mp4", "avi"])

    video_stem = None
    if uploaded:
        video_stem = Path(uploaded.name).stem
        stubs = stub_exists(get_video_id(video_stem))
        stub_ok  = stubs["player"] and stubs["ball"]
        stub_partial = stubs["player"] or stubs["ball"]

        if stub_ok:
            st.success("✅ Stub cache found — will skip YOLO inference")
        elif stub_partial:
            st.warning("⚠️ Partial stub found")
        else:
            st.info("ℹ️ No cache — will run full pipeline")

        st.caption(f"File: `{uploaded.name}`")

    st.divider()
    st.markdown("### ⚙️ Config")
    conf       = st.slider("Ball detection confidence", 0.1, 0.9, 0.3, 0.05)
    use_polygon = st.checkbox("Use polygon filtering", value=True)

    st.divider()
    run_btn = st.button("▶ Run Analysis", disabled=(uploaded is None))

    st.divider()
    st.markdown("### 📋 Processing Log")
    log_area = st.empty()
    log_area.markdown('<div class="log-box">Waiting for input...</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main area — run pipeline on button click
# ══════════════════════════════════════════════════════════════════════════════
if run_btn and uploaded and video_stem:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Running pipeline..."):
        mp4_path, heatmap_path, speed_stats, shot_count  = run_pipeline(
            input_video_path=tmp_path,
            video_stem=video_stem,
            conf=conf,
            use_polygon=use_polygon,
            log_placeholder=log_area,
        )

    if mp4_path:
        st.session_state["mp4_path"]    = mp4_path
        st.session_state["heatmap_path"] = heatmap_path
        st.session_state["speed_stats"]  = speed_stats
        st.session_state["video_stem"]   = video_stem
        st.session_state["shot_count"] = shot_count

# ══════════════════════════════════════════════════════════════════════════════
# Results section (persistent via session_state)
# ══════════════════════════════════════════════════════════════════════════════
if "mp4_path" in st.session_state:
    mp4_path     = st.session_state["mp4_path"]
    heatmap_path = st.session_state["heatmap_path"]
    speed_stats  = st.session_state["speed_stats"]

    # ── Tabs: Output Video | Player Heatmap ───────────────────────────────
    tab_video, tab_heatmap = st.tabs(["📹 Output Video", "🗺️ Player Heatmap"])

    # ── Tab 1: Video + Speed Stats + Advanced Stats ───────────────────────
    with tab_video:
        # Video player
        if os.path.exists(mp4_path):
            with open(mp4_path, "rb") as f:
                st.video(f.read())
        else:
            st.error("Video file not found.")

        # Speed stats
        st.markdown('<div class="section-title">⚡ Speed Statistics</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        p1 = speed_stats.get(1, {"avg": 0, "max": 0})
        p2 = speed_stats.get(2, {"avg": 0, "max": 0})

        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">🔵 Player 1 — Avg Speed</div>
              <div class="value">{p1['avg']}</div>
              <div class="unit">km/h</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">🔵 Player 1 — Max Speed</div>
              <div class="value">{p1['max']}</div>
              <div class="unit">km/h</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card p2">
              <div class="label">🔴 Player 2 — Avg Speed</div>
              <div class="value">{p2['avg']}</div>
              <div class="unit">km/h</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card p2">
              <div class="label">🔴 Player 2 — Max Speed</div>
              <div class="value">{p2['max']}</div>
              <div class="unit">km/h</div>
            </div>""", unsafe_allow_html=True)



        shot_count = st.session_state.get("shot_count", {1: 0, 2: 0})

        sc1, sc2 = st.columns(2, gap="small")
        with sc1:
            st.markdown(f"""
            <div class="metric-card">
            <div class="label">🔵 Player 1 — Shot Count</div>
            <div class="value">{shot_count.get(1, 0)}</div>
            <div class="unit">shots</div>
            </div>""", unsafe_allow_html=True)

        with sc2:
            st.markdown(f"""
            <div class="metric-card p2">
            <div class="label">🔴 Player 2 — Shot Count</div>
            <div class="value">{shot_count.get(2, 0)}</div>
            <div class="unit">shots</div>
            </div>""", unsafe_allow_html=True)

        # Advanced stats slots
        st.markdown('<div class="section-title">📊 Advanced Stats</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4, gap="small")
        slots = [
            ("🎯", "Shot Count", "Player 1 vs Player 2"),
            ("🔄", "Rally Length", "Avg / Max rallies"),
            ("📐", "Court Coverage", "% court covered"),
            ("➕", "Custom Stat", "Add your own metric"),
        ]
        for col, (icon, title, desc) in zip([s1, s2, s3, s4], slots):
            with col:
                st.markdown(f"""
                <div class="slot-card">
                  <div class="slot-icon">{icon}</div>
                  <strong style="color:#4a5580">{title}</strong><br>
                  <span style="font-size:0.72rem">{desc}</span>
                </div>""", unsafe_allow_html=True)

    # ── Tab 2: Player Heatmap ─────────────────────────────────────────────
    with tab_heatmap:
        if heatmap_path and os.path.exists(heatmap_path):
            st.markdown('<div class="section-title">🗺️ Player Movement Heatmap</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "<p style='color:#7b8ab0; font-size:0.85rem; margin-bottom:16px;'>"
                "Heatmap showing movement density for both players across the match. "
                "Warmer colors indicate higher presence in that zone.</p>",
                unsafe_allow_html=True
            )
            # Center the heatmap image
            col_l, col_c, col_r = st.columns([1, 3, 1])
            with col_c:
                st.image(heatmap_path, use_container_width=True,
                         caption="Movement heatmap — Player 1 & Player 2")
        else:
            st.warning("Heatmap not generated yet. Please run the analysis first.")

else:
    # Placeholder khi chưa có kết quả
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #3a4560;">
      <div style="font-size: 4rem; margin-bottom: 16px;">🎾</div>
      <div style="font-family: 'Rajdhani', sans-serif; font-size: 1.4rem; color: #5a6a90;">
        Upload a video and click <b>Run Analysis</b> to get started
      </div>
      <div style="font-size: 0.85rem; margin-top: 8px; color: #2e3a55;">
        Player tracking · Ball tracking · Court keypoints · Speed · Heatmap
      </div>
    </div>
    """, unsafe_allow_html=True)