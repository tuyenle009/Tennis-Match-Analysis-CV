import streamlit as st
import cv2
import os
import sys
import pickle
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from utils import smooth_player_trajectory
from trajectory import TrajectoryMapGenerator
from rally_analysis import RallyAnalyzer, generate_insights
from report import RallyReportPDF

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
            

/* ── Welcome screen (empty state) ───────────────────────────── */
  .welcome-screen {
    padding: 40px 20px;
    max-width: 1100px;
    margin: 0 auto;
  }
  .welcome-hero {
    text-align: center;
    margin-bottom: 50px;
  }
  .welcome-icon {
    font-size: 4rem;
    margin-bottom: 16px;
  }
  .welcome-hero h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    color: #1a3a6b;
    margin: 0 0 8px 0;
    letter-spacing: 1px;
  }
  .welcome-hero .tagline {
    font-size: 1.1rem;
    color: #5a6a90;
    margin: 0 0 8px 0;
  }
  .welcome-hero .subtle {
    font-size: 0.9rem;
    color: #8898b5;
    margin: 0;
  }
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin-bottom: 50px;
  }
  .feature-card {
    background: #f8faff;
    border: 1px solid #e4eafb;
    border-radius: 12px;
    padding: 24px 18px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  }
  .feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(26, 58, 107, 0.1);
    border-color: #4fc3f7;
  }
  .feature-icon {
    font-size: 2.2rem;
    margin-bottom: 12px;
  }
  .feature-card h3 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.15rem;
    color: #1a3a6b;
    margin: 0 0 8px 0;
    font-weight: 600;
  }
  .feature-card p {
    font-size: 0.82rem;
    color: #5a6a90;
    line-height: 1.4;
    margin: 0;
  }
  .quick-start {
    background: linear-gradient(135deg, #f0f4fa, #e8f1fa);
    border-left: 4px solid #4fc3f7;
    border-radius: 8px;
    padding: 20px 28px;
    max-width: 600px;
    margin: 0 auto;
  }
  .quick-start h4 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    color: #1a3a6b;
    margin: 0 0 12px 0;
  }
  .quick-start ol {
    margin: 0;
    padding-left: 22px;
    color: #3a4560;
    font-size: 0.9rem;
  }
  .quick-start li {
    margin-bottom: 6px;
  }
  @media (max-width: 768px) {
    .feature-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }


</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>🎾</div>
  <div>
    <h1>Tennis Rally Analytics</h1>
    <p>Per-rally movement, speed & trajectory analysis for coaching</p>
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
                 log_placeholder, output_dir: str = "output_videos",
                 progress_callback=None):
    """
    Chạy toàn bộ pipeline và trả về 9 giá trị.
    Mỗi video có 1 folder riêng: output_videos/<video_stem>/
    """
    # Tạo folder riêng cho video này (output_videos/inp_vid5/)
    output_dir = os.path.join(output_dir, video_stem)
    os.makedirs(output_dir, exist_ok=True)

    logs = []
    def log(msg: str, kind: str = "ok"):
        logs.append(f'<span class="{kind}">{msg}</span>')
        log_placeholder.markdown(
            '<div class="log-box">' + "<br>".join(logs) + "</div>",
            unsafe_allow_html=True
        )
   # ── Progress helper ────────────────────────────────────────────────────
    TOTAL_STEPS = 9
    def update_progress(step_num, label):
        if progress_callback:
            pct = int(step_num / TOTAL_STEPS * 100)
            progress_callback(pct, f"Step {step_num}/{TOTAL_STEPS}: {label}")
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
        return None, None, None, None, None, None, None, None, None

    # ── Load video ────────────────────────────────────────────────────────
    update_progress(1, "Loading video frames")
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
    update_progress(2, "Detecting players")
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
    update_progress(3, "Detecting ball")
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
    update_progress(4, "Detecting court keypoints")
    log("⏳ Detecting court keypoints (ResNet50)...", "run")
    court_detector = CourtLineDetector(model_path="models/keypoints_model_04.pth")
    court_keypoints = court_detector.predict(video_frames[0])
    log("✓ Court keypoints detected")

    # ── Filter players ────────────────────────────────────────────────────
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    log("✓ Players filtered")

    # ── Mini court + homography ───────────────────────────────────────────
    update_progress(5, "Computing court coordinates")
    mini_court = MiniCourt(video_frames[0])
    mini_court.set_homography(court_keypoints)
    
    player_mini, ball_mini = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints
    )
    log("✓ Mini court coordinates computed")

    player_mini = smooth_player_trajectory(player_mini, window=5)
    log("✓ Trajectory smoothed (window=5)")

    # ── Speed estimation ──────────────────────────────────────────────────
    update_progress(6, "Analyzing rally & computing speeds")
    speed_estimator = SpeedEstimator(fps=fps)
    speeds = speed_estimator.calculate_speed(player_mini, mini_court.get_width_of_mini_court())
    total_distances = speed_estimator.calculate_total_distance(
    player_mini,
    mini_court.get_width_of_mini_court()
    )
    log(f"✓ Total distance — P1: {total_distances[1]}m, P2: {total_distances[2]}m")

    speed_stats = calculate_speed_stats(speeds)
    log("✓ Speed estimation done")

    # ── Rally analysis ────────────────────────────────────────────────────
    log("⏳ Analyzing rally...", "run")
    analyzer = RallyAnalyzer(fps=fps, mini_court=mini_court)
    rally_stats = analyzer.analyze(player_mini, ball_mini, speeds, total_distances)
    insights = generate_insights(rally_stats)
    log(f"✓ Rally analysis done — {len(insights)} insights")


    


    log("⏳ Counting shots per player...", "run")
    frame_nums_with_ball_hits = ball_tracker.get_ball_shot_frames(ball_detections)
    shot_count = ball_tracker.get_shot_count_per_player(
        frame_nums_with_ball_hits, player_detections, ball_detections
    )
    log(f"✓ Shot count — P1: {shot_count[1]}, P2: {shot_count[2]}")

    # ── Draw output frames ────────────────────────────────────────────────
    update_progress(7, "Rendering output video")
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
    update_progress(8, "Generating visualizations")
    log("⏳ Generating heatmap...", "run")
    heatmap_gen = HeatmapGenerator(
        court_width=mini_court.court_end_x - mini_court.court_start_x,
        court_height=mini_court.court_end_y - mini_court.court_start_y,
    )
    heatmap_frame = heatmap_gen.draw_heatmap_on_last_frame(out_frames, player_mini, mini_court)
    heatmap_path = os.path.join(output_dir, f"heatmap_{video_stem}.png")
    cv2.imwrite(heatmap_path, heatmap_frame)
    log("✓ Heatmap saved")

    # ── Trajectory Map ────────────────────────────────────────────────────
    log("⏳ Generating trajectory map...", "run")
    trajectory_gen = TrajectoryMapGenerator()
    trajectory_img = trajectory_gen.generate(
        player_mini, ball_mini, mini_court, video_frames[0].shape
    )
    trajectory_path = os.path.join(output_dir, f"trajectory_{video_stem}.png")
    cv2.imwrite(trajectory_path, trajectory_img)
    log("✓ Trajectory map saved")

    # ── PDF Report ────────────────────────────────────────────────────────
    update_progress(9, "Creating PDF report")
    log("⏳ Generating PDF report...", "run")
    report_gen = RallyReportPDF()
    pdf_path = os.path.join(output_dir, f"rally_report_{video_stem}.pdf")
    try:
        report_gen.generate(
            rally_stats=rally_stats,
            insights=insights,
            trajectory_path=trajectory_path,
            output_path=pdf_path,
            video_name=video_stem,
            fps=fps,
        )
        log("✓ PDF report saved")
    except Exception as e:
        log(f"⚠️ PDF generation failed: {e}", "err")
        pdf_path = None




    log("🎾 Analysis complete!", "ok")

    return mp4_path, heatmap_path, trajectory_path, speed_stats, shot_count, total_distances, rally_stats, insights, pdf_path


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
    conf       = st.slider("Ball detection confidence", 0.1, 0.9, 0.25, 0.05)
    # use_polygon = st.checkbox("Use polygon filtering", value=True)
    use_polygon= True

    st.divider()
    run_btn = st.button("▶ Run Analysis", disabled=(uploaded is None))

    st.divider()
    # Download PDF button (chỉ hiện khi đã có PDF)
    if st.session_state.get("pdf_path") and os.path.exists(st.session_state["pdf_path"]):
        with open(st.session_state["pdf_path"], "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(st.session_state["pdf_path"]),
                mime="application/pdf",
                use_container_width=True,
            )

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

    # Progress bar container (tách riêng để clear sau khi xong)
    progress_container = st.empty()
    with progress_container.container():
        st.markdown(
            '<div class="section-title">⏳ Analyzing Rally...</div>',
            unsafe_allow_html=True
        )
        progress_bar = st.progress(0, text="Initializing pipeline...")

    def progress_callback(pct, step_label):
        progress_bar.progress(pct / 100, text=step_label)

    mp4_path, heatmap_path, trajectory_path, speed_stats, shot_count, total_distances, rally_stats, insights, pdf_path = run_pipeline(
        input_video_path=tmp_path,
        video_stem=video_stem,
        conf=conf,
        use_polygon=use_polygon,
        log_placeholder=log_area,
        progress_callback=progress_callback,
    )

    # Clear progress bar khi xong
    progress_container.empty()

    if mp4_path:
        st.session_state["mp4_path"]    = mp4_path
        st.session_state["heatmap_path"] = heatmap_path
        st.session_state["trajectory_path"] = trajectory_path
        st.session_state["speed_stats"]  = speed_stats
        st.session_state["video_stem"]   = video_stem
        st.session_state["shot_count"] = shot_count
        st.session_state["total_distances"] = total_distances
        st.session_state["rally_stats"]     = rally_stats    # ← THÊM
        st.session_state["insights"]        = insights       # ← THÊM
        st.session_state["pdf_path"]        = pdf_path       # ← THÊM
        st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# Results section (persistent via session_state)
# ══════════════════════════════════════════════════════════════════════════════
if "mp4_path" in st.session_state:
    mp4_path     = st.session_state["mp4_path"]
    heatmap_path = st.session_state["heatmap_path"]
    trajectory_path = st.session_state["trajectory_path"]
    speed_stats  = st.session_state["speed_stats"]
    rally_stats     = st.session_state["rally_stats"]
    insights        = st.session_state["insights"]

    # ─── Mapping VN → EN cho hiển thị UI ─────────────────────
    # (Giữ giá trị VN gốc trong rally_stats để dùng cho PDF + insights cards)
    style_map = {
        "tấn công":         "Attacking",
        "phòng thủ":        "Defensive",
        "trung lập":        "Neutral",
        "không xác định":   "Unknown",
    }
    zone_map = {
        "phía P1 - trái":   "P1 side - left",
        "phía P1 - phải":   "P1 side - right",
        "giữa sân - trái":  "Mid court - left",
        "giữa sân - phải":  "Mid court - right",
        "phía P2 - trái":   "P2 side - left",
        "phía P2 - phải":   "P2 side - right",
    }
    p1_style = style_map.get(rally_stats['position_styles'].get(1),
                              rally_stats['position_styles'].get(1, '-'))
    p2_style = style_map.get(rally_stats['position_styles'].get(2),
                              rally_stats['position_styles'].get(2, '-'))
    zone_display = zone_map.get(rally_stats.get('ball_end_zone'),
                                 rally_stats.get('ball_end_zone') or 'N/A')



    # ── Tabs: Output Video | Player Heatmap ───────────────────────────────
    tab_video, tab_trajectory, tab_heatmap, tab_insights = st.tabs([
    "📹 Output Video",
    "🗺️ Trajectory Map",
    "🔥 Heatmap",
    "📊 Rally Insights"
])

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


        st.markdown(
            '<div class="section-title">🚧 Shot Count '
            '<span style="font-size:0.7rem; color:#f0c040; '
            'background:#3a2f1a; padding:2px 8px; border-radius:4px; '
            'margin-left:8px;">EXPERIMENTAL — IN DEVELOPMENT</span></div>',
            unsafe_allow_html=True
        )
        st.caption(
            "⚠️ Hit detection is currently experimental. "
            "The algorithm is based on the sign change of the ball's delta_y and still has many limitations "
            "(perspective camera, ball detection gaps). The data is for reference only."
        )

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



        st.markdown('<div class="section-title">🏃 Total Distance</div>', unsafe_allow_html=True)

        total_distances = st.session_state.get("total_distances", {1: 0.0, 2: 0.0})
        d1, d2 = st.columns(2, gap="small")

        with d1:
            st.markdown(f"""
            <div class="metric-card">
            <div class="label">🔵 Player 1 — Total Distance</div>
            <div class="value">{total_distances.get(1, 0.0)}</div>
            <div class="unit">meters</div>
            </div>""", unsafe_allow_html=True)

        with d2:
            st.markdown(f"""
            <div class="metric-card p2">
            <div class="label">🔴 Player 2 — Total Distance</div>
            <div class="value">{total_distances.get(2, 0.0)}</div>
            <div class="unit">meters</div>
            </div>""", unsafe_allow_html=True)


        # Advanced stats slots
        st.markdown('<div class="section-title">📊 Advanced Stats</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4, gap="small")
        
        slots_data = [
            ("📐", "Court Coverage",
            f"P1: {rally_stats['coverage_areas'][1]:.0f}m² | P2: {rally_stats['coverage_areas'][2]:.0f}m²"),
            ("⚖️", "Movement Ratio",
            f"{rally_stats['movement_ratio']:.2f}x (P{rally_stats['high_runner']} runs more)"),
            ("⚡", "Sprint Count",
            f"P1: {rally_stats['sprint_counts'][1]} | P2: {rally_stats['sprint_counts'][2]}"),
            ("🎯", "Ball End Zone", zone_display),
        ]
        for col, (icon, title, value) in zip([s1, s2, s3, s4], slots_data):
            with col:
                st.markdown(f"""
                <div class="slot-card">
                <div class="slot-icon">{icon}</div>
                <strong style="color:#4a5580">{title}</strong><br>
                <span style="font-size:0.78rem; color:#2c3a50; font-weight:600">{value}</span>
                </div>""", unsafe_allow_html=True)


    # ── Tab 2: Trajectory Map ─────────────────────────────────────────────
    with tab_trajectory:
        if trajectory_path and os.path.exists(trajectory_path):
            st.markdown('<div class="section-title">🗺️ Rally Trajectory Map</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "Movement paths of both players during this rally. "
                "Color brightens over time (start = dim, end = bright). "
                "○ = start point, ● = end point, "
                "★ = serve location, ✕ = ball-out location.",
                unsafe_allow_html=True
            )
            col_l, col_c, col_r = st.columns([1, 3, 1])
            with col_c:
                st.image(trajectory_path, use_container_width=True,
                        caption="Trajectory map — Player 1 (blue), Player 2 (red)")
        else:
            st.warning("Trajectory map not generated yet. Please run the analysis first.")

    # ── Tab 3: Player Heatmap ─────────────────────────────────────────────
    with tab_heatmap:
        if heatmap_path and os.path.exists(heatmap_path):
            st.markdown('<div class="section-title">🗺️ Player Movement Heatmap</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "<p style='color:#7b8ab0; font-size:0.85rem; margin-bottom:16px;'>"
                "Heatmap showing movement density for both players in this rally. "
                "Warmer colors indicate zones where the player spent more time.",
                unsafe_allow_html=True
            )
            # Center the heatmap image
            col_l, col_c, col_r = st.columns([1, 3, 1])
            with col_c:
                st.image(heatmap_path, use_container_width=True,
                         caption="Movement heatmap — Player 1 & Player 2")
        else:
            st.warning("Heatmap not generated yet. Please run the analysis first.")

    # ── Tab 4: Rally Insights ─────────────────────────────────────────────
    with tab_insights:
        st.markdown('<div class="section-title">📊 Rally Performance Summary</div>',
                    unsafe_allow_html=True)

        # Bảng so sánh full
        rs = rally_stats

        table_html = f"""
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;
                    background:#F8F8FF; border-radius:8px; overflow:hidden;">
        <thead>
            <tr style="background:#1a3a6b; color:white;">
            <th style="padding:10px; text-align:left;">Metric</th>
            <th style="padding:10px; text-align:center;">🔵 Player 1</th>
            <th style="padding:10px; text-align:center;">🔴 Player 2</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #e0e6f0;">
            <td style="padding:8px 10px;">Distance</td>
            <td style="text-align:center;">{rs['distances'][1]:.1f} m</td>
            <td style="text-align:center;">{rs['distances'][2]:.1f} m</td>
            </tr>
            <tr style="border-bottom:1px solid #e0e6f0;">
            <td style="padding:8px 10px;">Peak Speed</td>
            <td style="text-align:center;">{rs['peak_speeds'][1]} km/h</td>
            <td style="text-align:center;">{rs['peak_speeds'][2]} km/h</td>
            </tr>
            <tr style="border-bottom:1px solid #e0e6f0;">
            <td style="padding:8px 10px;">Avg Speed</td>
            <td style="text-align:center;">{rs['avg_speeds'][1]} km/h</td>
            <td style="text-align:center;">{rs['avg_speeds'][2]} km/h</td>
            </tr>
            <tr style="border-bottom:1px solid #e0e6f0;">
            <td style="padding:8px 10px;">Sprint Count</td>
            <td style="text-align:center;">{rs['sprint_counts'][1]}</td>
            <td style="text-align:center;">{rs['sprint_counts'][2]}</td>
            </tr>
            <tr style="border-bottom:1px solid #e0e6f0;">
            <td style="padding:8px 10px;">Court Coverage</td>
            <td style="text-align:center;">{rs['coverage_areas'][1]:.0f} m²</td>
            <td style="text-align:center;">{rs['coverage_areas'][2]:.0f} m²</td>
            </tr>
            <tr>
            <td style="padding:8px 10px;">Position Style</td>
            <td style="text-align:center;">{p1_style}</td>
            <td style="text-align:center;">{p2_style}</td>
            </tr>
        </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        # Meta info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**⏱️ Duration**: {rs['duration_seconds']:.1f} seconds")
        with col2:
            st.markdown(f"**🎯 Ball End Zone**: {zone_display}")

        # Insights
        st.markdown('<div class="section-title">🔍 Auto-Generated Insights</div>',
                    unsafe_allow_html=True)

        if insights:
            for ins in insights:
                st.markdown(f"""
                <div style="background:#f0f4fa; border-left:4px solid #4fc3f7;
                            padding:10px 14px; margin-bottom:8px; border-radius:6px;
                            color:#1a3a6b; font-size:0.92rem;">
                • {ins}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No insights matched for this rally.")


else:
    # Welcome screen — empty state khi chưa upload video
    st.markdown("""
    <div class="welcome-screen">
      <div class="welcome-hero">
        <div class="welcome-icon">🎾</div>
        <h1>Tennis Rally Analytics</h1>
        <p class="tagline">Per-rally analytics for tennis coaches and players</p>
        <p class="subtle">Upload a rally video in the sidebar to get started</p>
      </div>

      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon">🗺️</div>
          <h3>Trajectory Map</h3>
          <p>Movement paths of both players with time-gradient visualization</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">📊</div>
          <h3>Performance Metrics</h3>
          <p>Speed, distance, sprint count, court coverage, position style</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🔍</div>
          <h3>Auto Insights</h3>
          <p>Rule-based tactical analysis in natural language for coaches</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">📄</div>
          <h3>PDF Report</h3>
          <p>One-page coaching report — downloadable, printable, shareable</p>
        </div>
      </div>

      <div class="quick-start">
        <h4>Quick Start</h4>
        <ol>
          <li>Upload rally video in the sidebar</li>
          <li>Configure detection settings (optional)</li>
          <li>Click <strong>▶ Run Analysis</strong></li>
        </ol>
      </div>
    </div>
    """, unsafe_allow_html=True)