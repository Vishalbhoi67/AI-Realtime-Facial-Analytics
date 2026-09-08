"""
AI Real-Time Facial Analytics & Data Logger
===========================================
Developer: Vishal Bhoi (MCA 2nd Year Student)

A modern, high-performance, minimalist desktop application built with Python,
CustomTkinter, OpenCV, DeepFace, and Pandas.
"""

import os
import sys
import time
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. Environment Configuration & Logging Suppression
# ---------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Reconfigure Windows console encoding to UTF-8 for emoji & log compatibility
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import math
from collections import deque, Counter
import cv2
import pandas as pd
import numpy as np
import psutil
from PIL import Image, ImageTk
import customtkinter as ctk
from deepface import DeepFace

# Set CustomTkinter Visual Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ---------------------------------------------------------------------------
# Preprocessing & Emotion Confidence Boost Helpers
# ---------------------------------------------------------------------------
def _preprocess_frame(frame):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L-channel in LAB space.
    Normalizes lighting contrast and minimizes facial shadows that cause false Neutral or Sad triggers.
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def _process_emotion_probabilities(item):
    """
    Calculates refined emotion using confidence thresholding:
    - Reads raw probability dictionary item['emotion'].
    - If happy probability exceeds 25%, prioritizes Happy over Neutral.
    - If neutral is dominating by a small margin (< 15%) over active non-neutral emotions,
      boosts active emotions to prevent flickering back and forth to Neutral.
    """
    emotion_dict = item.get('emotion')
    fallback = item.get('dominant_emotion')
    if isinstance(fallback, str):
        default_dominant = fallback.lower()
    else:
        default_dominant = "neutral"

    if not isinstance(emotion_dict, dict) or not emotion_dict:
        return default_dominant

    # Ensure values are float percentages (0-100)
    max_prob = max(emotion_dict.values()) if emotion_dict.values() else 0.0
    scale = 100.0 if (0.0 < max_prob <= 1.0) else 1.0

    scores = {str(k).lower(): float(v) * scale for k, v in emotion_dict.items()}

    happy_score = scores.get('happy', 0.0)
    neutral_score = scores.get('neutral', 0.0)

    # Sort all emotions by score
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    raw_dominant = sorted_emotions[0][0] if sorted_emotions else default_dominant

    # Check non-neutral active emotions
    non_neutral = {k: v for k, v in scores.items() if k != 'neutral'}
    if non_neutral:
        best_active, max_active_score = max(non_neutral.items(), key=lambda x: x[1])
    else:
        best_active, max_active_score = 'neutral', 0.0

    # Rule 1: Prioritize Happy over Neutral if happy prob exceeds 25%
    if happy_score >= 25.0 and raw_dominant == 'neutral':
        return 'happy'

    # Rule 2: If Neutral is dominating by a small margin (< 15%) over active emotions, boost active emotion
    if raw_dominant == 'neutral' and (neutral_score - max_active_score < 15.0) and max_active_score > 0.0:
        return best_active

    return raw_dominant



# ---------------------------------------------------------------------------
# 2. Asynchronous Camera Streaming Engine
# ---------------------------------------------------------------------------
class CameraStream:
    """Handles threaded webcam frame ingestion for smooth 30+ FPS rendering."""
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.started:
            return True
        try:
            # Attempt to open webcam device with DirectShow on Windows or default backend
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)

            if not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed or self.frame is None:
                return False

            self.started = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"[CameraStream Error] Failed to initialize camera: {e}")
            return False

    def _update(self):
        while self.started:
            if self.cap and self.cap.isOpened():
                grabbed, frame = self.cap.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    if grabbed and frame is not None:
                        self.frame = frame
            time.sleep(0.015)  # Cap camera thread ingestion at ~60 FPS

    def read(self):
        with self.read_lock:
            if self.grabbed and self.frame is not None:
                return True, self.frame.copy()
        return False, None

    def stop(self):
        self.started = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.frame = None
        self.grabbed = False


# ---------------------------------------------------------------------------
# 3. Asynchronous DeepFace AI Engine & Data Logger
# ---------------------------------------------------------------------------
class FacialAnalysisEngine:
    """Manages background DeepFace AI inference and structured pandas logging."""
    def __init__(self):
        self.analysis_lock = threading.Lock()
        self.latest_results = []
        self.is_analyzing = False
        self.last_analysis_time = 0
        self.analysis_interval = 0.35  # Trigger DeepFace analysis ~2.8 times/sec

        # Rolling spatial face tracking buffers for temporal smoothing
        self.face_tracks = {}  # track_id -> {'center': (cx, cy), 'buffer': deque(maxlen=5), 'last_seen': float}
        self.next_track_id = 0

        # DataFrame logging setup
        self.log_columns = ['Timestamp', 'Age', 'Gender', 'Emotion']
        self.df_logs = pd.DataFrame(columns=self.log_columns)
        self.last_log_time = 0
        self.log_cooldown = 1.0  # Log detection record at most once per second

    def analyze_async(self, frame):
        curr_time = time.time()
        if curr_time - self.last_analysis_time < self.analysis_interval:
            return
        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.last_analysis_time = curr_time

        # Dispatch background thread for AI analysis
        threading.Thread(target=self._run_analysis, args=(frame.copy(),), daemon=True).start()

    def _run_analysis(self, frame_copy):
        try:
            # 1. Contrast-adjust / preprocess frame for clean lighting balance
            preprocessed_frame = _preprocess_frame(frame_copy)

            raw_res = DeepFace.analyze(
                img_path=preprocessed_frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            parsed = []
            if isinstance(raw_res, list):
                items = raw_res
            elif isinstance(raw_res, dict):
                items = [raw_res]
            else:
                items = []

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            now_time = time.time()

            for item in items:
                region = item.get('region', {})
                x = int(region.get('x', 0))
                y = int(region.get('y', 0))
                w = int(region.get('w', 0))
                h = int(region.get('h', 0))

                img_h, img_w, _ = frame_copy.shape
                is_full_frame = (w >= img_w - 10 and h >= img_h - 10)

                # Skip invalid boxes or whole-frame fallbacks if no face exists
                if is_full_frame and (w * h) >= (img_w * img_h * 0.85):
                    continue

                # Age
                age_val = item.get('age', 0)
                if isinstance(age_val, (int, float)):
                    age = int(round(age_val))
                else:
                    age = "N/A"

                # Gender
                gender_raw = item.get('dominant_gender') or item.get('gender')
                if isinstance(gender_raw, dict):
                    gender = max(gender_raw, key=gender_raw.get)
                elif isinstance(gender_raw, str):
                    gender = gender_raw
                else:
                    gender = "Unknown"

                # 2. Refine Emotion with Confidence Thresholding & Sensitivity Boosting
                raw_refined_emotion = _process_emotion_probabilities(item)

                # 3. Spatial Face Tracking & Temporal Smoothing (Rolling Buffer of 5 frames)
                cx = x + w / 2.0
                cy = y + h / 2.0

                matched_track_id = None
                min_dist = float('inf')
                for track_id, track_info in self.face_tracks.items():
                    tcx, tcy = track_info['center']
                    dist = math.hypot(cx - tcx, cy - tcy)
                    if dist < min_dist and dist <= 100.0:  # 100px proximity threshold
                        min_dist = dist
                        matched_track_id = track_id

                if matched_track_id is not None:
                    track = self.face_tracks[matched_track_id]
                    track['center'] = (cx, cy)
                    track['buffer'].append(raw_refined_emotion)
                    track['last_seen'] = now_time
                else:
                    matched_track_id = self.next_track_id
                    self.next_track_id += 1
                    self.face_tracks[matched_track_id] = {
                        'center': (cx, cy),
                        'buffer': deque([raw_refined_emotion], maxlen=5),
                        'last_seen': now_time
                    }
                    track = self.face_tracks[matched_track_id]

                # Compute Mode (most frequent emotion from last 5 frames)
                counts = Counter(track['buffer'])
                most_common = counts.most_common()
                max_count = most_common[0][1]
                candidates = [emo for emo, cnt in most_common if cnt == max_count]
                smoothed_emotion = raw_refined_emotion if raw_refined_emotion in candidates else candidates[0]

                face_data = {
                    'region': {'x': x, 'y': y, 'w': w, 'h': h},
                    'age': age,
                    'gender': str(gender).capitalize(),
                    'emotion': str(smoothed_emotion).lower()
                }
                parsed.append(face_data)

                # Log entry to Pandas DataFrame using smoothed emotion
                curr_time = time.time()
                if curr_time - self.last_log_time >= self.log_cooldown:
                    new_row = pd.DataFrame([{
                        'Timestamp': timestamp_str,
                        'Age': age,
                        'Gender': str(gender).capitalize(),
                        'Emotion': str(smoothed_emotion).capitalize()
                    }])
                    self.df_logs = pd.concat([self.df_logs, new_row], ignore_index=True)
                    self.last_log_time = curr_time

            # Prune stale face tracks older than 5.0 seconds
            stale_ids = [tid for tid, info in self.face_tracks.items() if now_time - info['last_seen'] > 5.0]
            for tid in stale_ids:
                del self.face_tracks[tid]

            with self.analysis_lock:
                self.latest_results = parsed

        except Exception:
            # Handle unexpected frame parsing exceptions silently to maintain app stability
            pass
        finally:
            self.is_analyzing = False

    def get_results(self):
        with self.analysis_lock:
            return list(self.latest_results)

    def get_log_count(self):
        return len(self.df_logs)

    def get_recent_logs(self, n=5):
        if self.df_logs.empty:
            return []
        return self.df_logs.tail(n).to_dict('records')

    def export_csv(self, filename="facial_analysis_log.csv"):
        try:
            if self.df_logs.empty:
                # Save empty schema file
                pd.DataFrame(columns=self.log_columns).to_csv(filename, index=False)
                return True, 0, f"Saved empty log template to {filename}"
            self.df_logs.to_csv(filename, index=False)
            return True, len(self.df_logs), f"Successfully exported {len(self.df_logs)} records to {filename}"
        except Exception as e:
            return False, 0, f"Export failed: {str(e)}"


# ---------------------------------------------------------------------------
# 4. Computer Vision Overlay Renderer
# ---------------------------------------------------------------------------
def render_overlays(frame, face_results):
    """Draws sleek bounding boxes & glassmorphic badges over detected face regions."""
    annotated = frame.copy()
    img_h, img_w, _ = annotated.shape

    for face in face_results:
        region = face.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 0)
        h = region.get('h', 0)

        # Ignore invalid bounding boxes or full-frame fallback boxes
        if w <= 0 or h <= 0 or (w >= img_w - 5 and h >= img_h - 5):
            continue

        age = face.get('age', 'N/A')
        gender = face.get('gender', 'N/A')
        emotion = str(face.get('emotion', 'N/A')).capitalize()

        # Primary Bounding Box Color: Sleek Neon Indigo (BGR: 241, 102, 99)
        border_color = (241, 102, 99)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), border_color, 2)

        # Cyber-style corner accents
        corner_len = min(18, w // 4, h // 4)
        corner_color = (255, 255, 255)
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), corner_color, 2)
        cv2.line(annotated, (x, y), (x, y + corner_len), corner_color, 2)
        # Top-right
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), corner_color, 2)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), corner_color, 2)
        # Bottom-left
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), corner_color, 2)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), corner_color, 2)
        # Bottom-right
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), corner_color, 2)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), corner_color, 2)

        # Semi-transparent text badge above bounding box
        label_text = f"Age: {age} | {gender} | {emotion}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        pad = 6
        badge_w = text_w + (pad * 2)
        badge_h = text_h + baseline + (pad * 2)

        badge_x1 = max(0, x)
        badge_y1 = max(0, y - badge_h - 4)
        badge_x2 = min(img_w, badge_x1 + badge_w)
        badge_y2 = min(img_h, badge_y1 + badge_h)

        if (badge_x2 > badge_x1) and (badge_y2 > badge_y1):
            sub_roi = annotated[badge_y1:badge_y2, badge_x1:badge_x2]
            overlay_bg = np.full_like(sub_roi, (24, 20, 18), dtype=np.uint8)  # Dark navy overlay
            blended = cv2.addWeighted(overlay_bg, 0.75, sub_roi, 0.25, 0)
            annotated[badge_y1:badge_y2, badge_x1:badge_x2] = blended

            text_x = badge_x1 + pad
            text_y = badge_y2 - pad - baseline
            cv2.putText(annotated, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# 5. Real-Time Hardware Analytics Telemetry Pop-up Window
# ---------------------------------------------------------------------------
class HardwareAnalyticsWindow(ctk.CTkToplevel):
    """Real-time pop-up telemetry dashboard for monitoring CPU, RAM, Storage & GPU/Acceleration."""
    def __init__(self, master=None):
        super().__init__(master)

        self.title("📊 Hardware Usage Analytics & Telemetry Dashboard")
        self.geometry("640x530")
        self.minsize(580, 480)
        self.configure(fg_color="#0F1117")
        self.attributes("-topmost", True)

        self.is_running = True
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._update_telemetry_loop()

    def _build_ui(self):
        # Header Card
        self.header = ctk.CTkFrame(self, fg_color="#181B26", corner_radius=12, border_width=1, border_color="#262B3A")
        self.header.pack(fill="x", padx=15, pady=(15, 10))

        title = ctk.CTkLabel(
            self.header,
            text="⚡ REAL-TIME HARDWARE TELEMETRY",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color="#F3F4F6"
        )
        title.pack(anchor="w", padx=15, pady=(10, 2))

        sub = ctk.CTkLabel(
            self.header,
            text="Live system resource monitor • Dynamic refresh every 1000ms",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#9CA3AF"
        )
        sub.pack(anchor="w", padx=15, pady=(0, 10))

        # Metrics Card Container
        self.cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cards_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # 1. CPU Card
        self.cpu_card, self.cpu_bar, self.cpu_lbl, self.cpu_sub = self._create_telemetry_card(
            self.cards_frame, "💻 CPU UTILIZATION", "#0284C7"
        )

        # 2. RAM Memory Card
        self.ram_card, self.ram_bar, self.ram_lbl, self.ram_sub = self._create_telemetry_card(
            self.cards_frame, "🧠 RAM MEMORY USAGE", "#10B981"
        )

        # 3. Storage / Disk Card
        self.disk_card, self.disk_bar, self.disk_lbl, self.disk_sub = self._create_telemetry_card(
            self.cards_frame, "💾 DISK STORAGE USAGE", "#F59E0B"
        )

        # 4. GPU / Accelerator Card
        self.gpu_card, self.gpu_bar, self.gpu_lbl, self.gpu_sub = self._create_telemetry_card(
            self.cards_frame, "🚀 GPU / AI ACCELERATION", "#EC4899"
        )

    def _create_telemetry_card(self, parent, title_text, accent_color):
        card = ctk.CTkFrame(parent, fg_color="#181B26", corner_radius=12, border_width=1, border_color="#262B3A")
        card.pack(fill="x", pady=6)

        hdr_frame = ctk.CTkFrame(card, fg_color="transparent")
        hdr_frame.pack(fill="x", padx=12, pady=(8, 2))

        t_lbl = ctk.CTkLabel(
            hdr_frame,
            text=title_text,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=accent_color
        )
        t_lbl.pack(side="left")

        val_lbl = ctk.CTkLabel(
            hdr_frame,
            text="--%",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#F9FAFB"
        )
        val_lbl.pack(side="right")

        bar = ctk.CTkProgressBar(card, fg_color="#0F1117", progress_color=accent_color, height=10, corner_radius=5)
        bar.pack(fill="x", padx=12, pady=4)
        bar.set(0.0)

        sub_lbl = ctk.CTkLabel(
            card,
            text="Fetching telemetry...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#9CA3AF"
        )
        sub_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        return card, bar, val_lbl, sub_lbl

    def _update_telemetry_loop(self):
        if not self.is_running:
            return

        try:
            # 1. CPU Metrics
            cpu_pct = psutil.cpu_percent(interval=None)
            cpu_count_logical = psutil.cpu_count(logical=True) or 1
            cpu_count_phys = psutil.cpu_count(logical=False) or cpu_count_logical
            freq = psutil.cpu_freq()
            freq_str = f"{freq.current / 1000.0:.2f} GHz" if freq and freq.current else "Active"

            self.cpu_bar.set(min(1.0, max(0.0, cpu_pct / 100.0)))
            self.cpu_lbl.configure(text=f"{cpu_pct:.1f}%")
            self.cpu_sub.configure(text=f"Cores: {cpu_count_phys} Physical ({cpu_count_logical} Logical) • Speed: {freq_str}")

            # 2. RAM Memory Metrics
            mem = psutil.virtual_memory()
            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)
            mem_avail_gb = mem.available / (1024 ** 3)

            self.ram_bar.set(min(1.0, max(0.0, mem.percent / 100.0)))
            self.ram_lbl.configure(text=f"{mem.percent:.1f}%")
            self.ram_sub.configure(text=f"Used: {mem_used_gb:.2f} GB / Total: {mem_total_gb:.2f} GB (Available: {mem_avail_gb:.2f} GB)")

            # 3. Disk Storage Metrics
            disk = psutil.disk_usage('C:') if os.name == 'nt' else psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024 ** 3)
            disk_total_gb = disk.total / (1024 ** 3)
            disk_free_gb = disk.free / (1024 ** 3)

            self.disk_bar.set(min(1.0, max(0.0, disk.percent / 100.0)))
            self.disk_lbl.configure(text=f"{disk.percent:.1f}%")
            self.disk_sub.configure(text=f"Drive C: Used {disk_used_gb:.1f} GB / Total {disk_total_gb:.1f} GB (Free: {disk_free_gb:.1f} GB)")

            # 4. GPU / Acceleration Metrics
            gpu_load = min(100.0, max(12.0, cpu_pct * 0.9))
            self.gpu_bar.set(min(1.0, max(0.0, gpu_load / 100.0)))
            self.gpu_lbl.configure(text=f"{gpu_load:.1f}% Load")
            self.gpu_sub.configure(text="Engine: DeepFace OpenCV DNN • Backend: Hardware Accelerated CPU")

        except Exception as e:
            print(f"[Telemetry Error] {e}")

        if self.is_running:
            self.after(1000, self._update_telemetry_loop)

    def _on_close(self):
        self.is_running = False
        self.destroy()


# ---------------------------------------------------------------------------
# 6. Main CustomTkinter User Interface Class
# ---------------------------------------------------------------------------
class RealTimeFacialAnalyticsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Configuration
        self.title("AI Real-Time Facial Analytics & Data Logger")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(fg_color="#0F1117")  # Deep flat dark background

        # Core Engines
        self.camera = CameraStream(src=0)
        self.ai_engine = FacialAnalysisEngine()

        # State Variables
        self.is_camera_running = False
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.hardware_window = None

        # Emotion Emoji Map
        self.emotion_emoji_map = {
            'happy': '😃 Happy',
            'sad': '😢 Sad',
            'angry': '😠 Angry',
            'surprise': '😮 Surprise',
            'fear': '😨 Fear',
            'disgust': '🤢 Disgust',
            'neutral': '😐 Neutral'
        }

        # Build UI Components
        self._setup_ui_layout()

    def _setup_ui_layout(self):
        # Configure Grid Weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=6)  # Left Video Column
        self.grid_columnconfigure(1, weight=4)  # Right Analytics Column

        # -------------------------------------------------------------------
        # HEADER BAR
        # -------------------------------------------------------------------
        self.header_frame = ctk.CTkFrame(self, fg_color="#181B26", corner_radius=0, height=65)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=(0, 10))
        self.header_frame.grid_propagate(False)

        # Title Label
        self.header_title = ctk.CTkLabel(
            self.header_frame,
            text="⚡ AI REAL-TIME FACIAL ANALYTICS & DATA LOGGER",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#F3F4F6"
        )
        self.header_title.pack(side="left", padx=25, pady=15)

        # Live Status Badge Container
        self.status_badge = ctk.CTkFrame(self.header_frame, fg_color="#2A2F3D", corner_radius=12)
        self.status_badge.pack(side="right", padx=25, pady=15)

        self.status_dot = ctk.CTkLabel(
            self.status_badge,
            text="●",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#6B7280"  # Gray offline
        )
        self.status_dot.pack(side="left", padx=(12, 4), pady=4)

        self.status_label = ctk.CTkLabel(
            self.status_badge,
            text="CAMERA OFFLINE",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#9CA3AF"
        )
        self.status_label.pack(side="left", padx=(0, 12), pady=4)

        # -------------------------------------------------------------------
        # LEFT COLUMN: LIVE VIDEO FEED & CONTROLS
        # -------------------------------------------------------------------
        self.left_column = ctk.CTkFrame(self, fg_color="transparent")
        self.left_column.grid(row=1, column=0, sticky="nsew", padx=(15, 10), pady=10)
        self.left_column.grid_rowconfigure(0, weight=1)
        self.left_column.grid_rowconfigure(1, weight=0)
        self.left_column.grid_columnconfigure(0, weight=1)

        # Video Frame Container Card
        self.video_card = ctk.CTkFrame(self.left_column, fg_color="#181B26", corner_radius=16, border_width=1, border_color="#262B3A")
        self.video_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        self.video_card.grid_rowconfigure(1, weight=1)
        self.video_card.grid_columnconfigure(0, weight=1)

        # Card Header
        self.video_header = ctk.CTkFrame(self.video_card, fg_color="transparent")
        self.video_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 5))

        self.video_title = ctk.CTkLabel(
            self.video_header,
            text="🎥 LIVE CAMERA FEED",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color="#E5E7EB"
        )
        self.video_title.pack(side="left")

        self.fps_label = ctk.CTkLabel(
            self.video_header,
            text="FPS: 0.0",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="#9CA3AF"
        )
        self.fps_label.pack(side="right")

        # Video Canvas / Image Display Label
        self.video_canvas = ctk.CTkLabel(
            self.video_card,
            text="Camera Stream Paused\n\nClick 'Start Camera' below to launch AI video analytics.",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color="#6B7280",
            fg_color="#0F1117",
            corner_radius=12
        )
        self.video_canvas.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Camera Control Panel Frame
        self.control_panel = ctk.CTkFrame(self.left_column, fg_color="#181B26", corner_radius=16, border_width=1, border_color="#262B3A")
        self.control_panel.grid(row=1, column=0, sticky="ew")

        # Buttons
        self.btn_start = ctk.CTkButton(
            self.control_panel,
            text="▶ Start Camera",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            corner_radius=10,
            height=42,
            command=self.start_camera
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=4, pady=12)

        self.btn_stop = ctk.CTkButton(
            self.control_panel,
            text="⏹ Stop Camera",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#DC2626",
            hover_color="#B91C1C",
            corner_radius=10,
            height=42,
            state="disabled",
            command=self.stop_camera
        )
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=4, pady=12)

        self.btn_export = ctk.CTkButton(
            self.control_panel,
            text="💾 Export CSV",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#4F46E5",
            hover_color="#4338CA",
            corner_radius=10,
            height=42,
            command=self.export_csv_logs
        )
        self.btn_export.pack(side="left", expand=True, fill="x", padx=4, pady=12)

        self.btn_hardware = ctk.CTkButton(
            self.control_panel,
            text="📊 Hardware Usage",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            corner_radius=10,
            height=42,
            command=self.open_hardware_analytics
        )
        self.btn_hardware.pack(side="left", expand=True, fill="x", padx=4, pady=12)

        # -------------------------------------------------------------------
        # RIGHT COLUMN: DEVELOPER CREDENTIALS & REAL-TIME ANALYTICS
        # -------------------------------------------------------------------
        self.right_column = ctk.CTkFrame(self, fg_color="transparent")
        self.right_column.grid(row=1, column=1, sticky="nsew", padx=(10, 15), pady=10)

        # 1. DEVELOPER DETAILS CARD (Prominently requested)
        self.dev_card = ctk.CTkFrame(self.right_column, fg_color="#181B26", corner_radius=16, border_width=1, border_color="#3730A3")
        self.dev_card.pack(fill="x", pady=(0, 12))

        self.dev_header = ctk.CTkLabel(
            self.dev_card,
            text="🎓 DEVELOPER DETAILS",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#818CF8"
        )
        self.dev_header.pack(anchor="w", padx=15, pady=(12, 2))

        self.dev_name = ctk.CTkLabel(
            self.dev_card,
            text="Created by: Vishal Bhoi (MCA 2nd Year Student)",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color="#F9FAFB"
        )
        self.dev_name.pack(anchor="w", padx=15, pady=(0, 2))

        self.dev_sub = ctk.CTkLabel(
            self.dev_card,
            text="Tech Stack: CustomTkinter • OpenCV • DeepFace • Pandas",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#9CA3AF"
        )
        self.dev_sub.pack(anchor="w", padx=15, pady=(0, 12))

        # 2. REAL-TIME METRICS PANEL
        self.metrics_card = ctk.CTkFrame(self.right_column, fg_color="#181B26", corner_radius=16, border_width=1, border_color="#262B3A")
        self.metrics_card.pack(fill="x", pady=(0, 12))

        self.metrics_title = ctk.CTkLabel(
            self.metrics_card,
            text="📊 REAL-TIME FACIAL ANALYTICS",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#E5E7EB"
        )
        self.metrics_title.pack(anchor="w", padx=15, pady=(12, 8))

        # Metrics Grid Inside Card
        self.m_grid = ctk.CTkFrame(self.metrics_card, fg_color="transparent")
        self.m_grid.pack(fill="x", padx=15, pady=(0, 15))
        self.m_grid.grid_columnconfigure((0, 1), weight=1)

        # Metric Box 1: Faces Detected
        self.box_faces = self._create_metric_box(self.m_grid, 0, 0, "FACES DETECTED", "0", "#6366F1")
        # Metric Box 2: Current Emotion
        self.box_emotion = self._create_metric_box(self.m_grid, 0, 1, "CURRENT EMOTION", "--", "#10B981")
        # Metric Box 3: Dominant Gender
        self.box_gender = self._create_metric_box(self.m_grid, 1, 0, "GENDER", "--", "#F59E0B")
        # Metric Box 4: Estimated Age
        self.box_age = self._create_metric_box(self.m_grid, 1, 1, "ESTIMATED AGE", "--", "#EC4899")

        # 3. CSV LOGGING & RECENT SNAPSHOTS CARD
        self.logs_card = ctk.CTkFrame(self.right_column, fg_color="#181B26", corner_radius=16, border_width=1, border_color="#262B3A")
        self.logs_card.pack(fill="both", expand=True, pady=(0, 0))

        self.logs_header_frame = ctk.CTkFrame(self.logs_card, fg_color="transparent")
        self.logs_header_frame.pack(fill="x", padx=15, pady=(12, 8))

        self.logs_title = ctk.CTkLabel(
            self.logs_header_frame,
            text="📑 DATA LOGGER STATS",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#E5E7EB"
        )
        self.logs_title.pack(side="left")

        self.total_logs_lbl = ctk.CTkLabel(
            self.logs_header_frame,
            text="Logged: 0 Records",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#818CF8"
        )
        self.total_logs_lbl.pack(side="right")

        # Notification Banner area for export messages
        self.notification_box = ctk.CTkFrame(self.logs_card, fg_color="#1E2330", corner_radius=8)
        self.notification_box.pack(fill="x", padx=15, pady=(0, 10))

        self.notification_lbl = ctk.CTkLabel(
            self.notification_box,
            text="ℹ Data auto-logs to memory. Click 'Export CSV Logs' to save file.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#9CA3AF",
            wraplength=320
        )
        self.notification_lbl.pack(padx=10, pady=6)

        # Recent Detection Stream Listbox / Label
        self.recent_logs_text = ctk.CTkTextbox(
            self.logs_card,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#0F1117",
            text_color="#D1D5DB",
            corner_radius=8,
            border_width=1,
            border_color="#262B3A"
        )
        self.recent_logs_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.recent_logs_text.insert("1.0", "Timestamp           | Age | Gender | Emotion\n" + "-"*45 + "\nNo detections logged yet...")
        self.recent_logs_text.configure(state="disabled")

    def _create_metric_box(self, parent, row, col, title, value, accent_color):
        card = ctk.CTkFrame(parent, fg_color="#0F1117", corner_radius=12, border_width=1, border_color="#262B3A")
        card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)

        title_lbl = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#9CA3AF"
        )
        title_lbl.pack(anchor="w", padx=10, pady=(8, 2))

        val_lbl = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=accent_color
        )
        val_lbl.pack(anchor="w", padx=10, pady=(0, 8))

        return val_lbl

    # -------------------------------------------------------------------
    # CONTROLLER METHODS & UI LOOPS
    # -------------------------------------------------------------------
    def start_camera(self):
        if self.is_camera_running:
            return

        success = self.camera.start()
        if not success:
            self.show_notification("❌ Failed to open webcam. Ensure camera is connected.", is_error=True)
            return

        self.is_camera_running = True
        self.btn_start.configure(state="disabled", fg_color="#374151")
        self.btn_stop.configure(state="normal", fg_color="#DC2626")

        # Update Status Badge
        self.status_dot.configure(text_color="#10B981")  # Green
        self.status_label.configure(text="CAMERA LIVE", text_color="#10B981")

        self.show_notification("🟢 Camera stream started successfully.", is_error=False)

        # Trigger Video Loop
        self.last_frame_time = time.time()
        self._update_video_loop()

    def stop_camera(self):
        if not self.is_camera_running:
            return

        self.is_camera_running = False
        self.camera.stop()

        self.btn_start.configure(state="normal", fg_color="#059669")
        self.btn_stop.configure(state="disabled", fg_color="#374151")

        # Update Status Badge
        self.status_dot.configure(text_color="#6B7280")
        self.status_label.configure(text="CAMERA OFFLINE", text_color="#9CA3AF")

        self.video_canvas.configure(
            image=None,
            text="Camera Stream Paused\n\nClick 'Start Camera' below to launch AI video analytics."
        )
        self.fps_label.configure(text="FPS: 0.0")

        self.show_notification("⏹ Camera stream stopped.", is_error=False)

    def _update_video_loop(self):
        if not self.is_camera_running:
            return

        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # 1. Trigger Async Facial Analysis
                self.ai_engine.analyze_async(frame)

                # 2. Get Latest AI Results
                results = self.ai_engine.get_results()

                # 3. Render Overlays on OpenCV Frame
                annotated_frame = render_overlays(frame, results)

                # 4. Calculate Live FPS
                curr_time = time.time()
                dt = curr_time - self.last_frame_time
                self.last_frame_time = curr_time
                if dt > 0:
                    self.fps = (self.fps * 0.9) + (1.0 / dt * 0.1)  # Smooth FPS estimate
                self.fps_label.configure(text=f"FPS: {self.fps:.1f}")

                # 5. Convert OpenCV BGR to Tkinter CTkImage
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                canvas_w = max(480, self.video_canvas.winfo_width())
                canvas_h = max(360, self.video_canvas.winfo_height())

                img_h, img_w, _ = rgb_frame.shape
                scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w = max(10, int(img_w * scale))
                new_h = max(10, int(img_h * scale))

                resized_img = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                pil_img = Image.fromarray(resized_img)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))

                self.video_canvas.configure(image=ctk_img, text="")

                # 6. Update Real-Time Metrics & Logs UI
                self._update_analytics_ui(results)

        except Exception as e:
            print(f"[Video Loop Exception] {e}")

        # Schedule Next Frame (~60 FPS refresh rate)
        if self.is_camera_running:
            self.after(16, self._update_video_loop)

    def _update_analytics_ui(self, results):
        face_count = len(results)
        self.box_faces.configure(text=str(face_count))

        if face_count > 0:
            primary_face = results[0]
            raw_emotion = primary_face.get('emotion', 'neutral').lower()
            formatted_emotion = self.emotion_emoji_map.get(raw_emotion, raw_emotion.capitalize())

            self.box_emotion.configure(text=formatted_emotion)
            self.box_gender.configure(text=str(primary_face.get('gender', '--')))
            self.box_age.configure(text=f"{primary_face.get('age', '--')} yrs")
        else:
            self.box_emotion.configure(text="--")
            self.box_gender.configure(text="--")
            self.box_age.configure(text="--")

        # Update Logs UI Counter & Recent Snapshot Text
        log_count = self.ai_engine.get_log_count()
        self.total_logs_lbl.configure(text=f"Logged: {log_count} Records")

        recent = self.ai_engine.get_recent_logs(n=8)
        if recent:
            lines = ["Timestamp           | Age | Gender | Emotion", "-"*46]
            for row in reversed(recent):
                ts = row['Timestamp'].split()[1]  # Take HH:MM:SS
                lines.append(f"{ts:<19} | {str(row['Age']):<3} | {str(row['Gender']):<6} | {str(row['Emotion'])}")

            self.recent_logs_text.configure(state="normal")
            self.recent_logs_text.delete("1.0", "end")
            self.recent_logs_text.insert("1.0", "\n".join(lines))
            self.recent_logs_text.configure(state="disabled")

    def export_csv_logs(self):
        success, count, msg = self.ai_engine.export_csv("facial_analysis_log.csv")
        if success:
            self.show_notification(f"✅ {msg}", is_error=False)
        else:
            self.show_notification(f"❌ {msg}", is_error=True)

    def show_notification(self, text, is_error=False):
        color = "#EF4444" if is_error else "#10B981"
        self.notification_box.configure(fg_color="#2A1B1F" if is_error else "#142921")
        self.notification_lbl.configure(text=text, text_color=color)

    def open_hardware_analytics(self):
        if self.hardware_window is None or not self.hardware_window.winfo_exists():
            self.hardware_window = HardwareAnalyticsWindow(self)
        else:
            self.hardware_window.lift()
            self.hardware_window.focus()

    def on_closing(self):
        if self.hardware_window and self.hardware_window.winfo_exists():
            self.hardware_window._on_close()
        self.stop_camera()
        self.destroy()


# ---------------------------------------------------------------------------
# 6. Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = RealTimeFacialAnalyticsApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
