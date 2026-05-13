# utils/smoothing.py
import pandas as pd

def smooth_trajectory(player_mini_court_detections, window=5):
    """
    Làm mượt trajectory player bằng moving average — giảm jitter do bbox rung.
    Quan trọng để tính distance chính xác (không bị thổi phồng).
    
    Args:
        player_mini_court_detections: list[dict] — [{player_id: (x, y)}, ...]
        window: kích thước cửa sổ moving average (frame)
    
    Returns:
        list[dict] cùng format nhưng đã smooth
    """
    if not player_mini_court_detections:
        return player_mini_court_detections

    # Thu thập tất cả player_id xuất hiện trong video
    all_player_ids = set()
    for frame in player_mini_court_detections:
        all_player_ids.update(frame.keys())

    # Build series x, y riêng cho từng player
    smoothed_series = {}
    for pid in all_player_ids:
        xs, ys = [], []
        for frame in player_mini_court_detections:
            if pid in frame:
                xs.append(frame[pid][0])
                ys.append(frame[pid][1])
            else:
                xs.append(None)
                ys.append(None)

        s_x = pd.Series(xs).rolling(window=window, min_periods=1, center=True).mean()
        s_y = pd.Series(ys).rolling(window=window, min_periods=1, center=True).mean()
        smoothed_series[pid] = (s_x.tolist(), s_y.tolist())

    # Reconstruct list[dict], giữ nguyên frame nào không có player
    result = []
    for i, frame in enumerate(player_mini_court_detections):
        new_frame = {}
        for pid in frame:
            x = smoothed_series[pid][0][i]
            y = smoothed_series[pid][1][i]
            if x is not None and y is not None:
                new_frame[pid] = (float(x), float(y))
        result.append(new_frame)

    return result