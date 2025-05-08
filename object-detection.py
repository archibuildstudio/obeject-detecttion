import logging
import os
import sys
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List
from tqdm import tqdm
from collections import defaultdict

# Suppress Ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from ultralytics import YOLO
import yolox
print("yolox.__version__:", yolox.__version__)

# Add ByteTrack path
sys.path.append(f"{os.getcwd()}/ByteTrack")

from ultralytics.trackers import BYTETracker
from yolox.tracker.byte_tracker import BYTETracker as YoloXBYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

import supervision
print("supervision.__version__:", supervision.__version__)

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator

# Setup paths
HOME = os.getcwd()
SOURCE_VIDEO_PATH = os.path.join(HOME, "moving-counts.mp4")
TARGET_VIDEO_PATH = os.path.join(HOME, "moving-counts-result.mp4")

print(SOURCE_VIDEO_PATH)
print("Video exists?", os.path.isfile(SOURCE_VIDEO_PATH))

# Tracker config
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# Load and configure model
MODEL = "yolov8x.pt"
model = YOLO(MODEL)
model.verbose = False
model.fuse()

# Initialize BYTETracker
byte_tracker = YoloXBYTETracker(BYTETrackerArgs())

# Prepare video reader, info, annotators
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=.5)

# Convert detections to required format for BYTETracker
def detections2boxes(detections):
    return np.hstack((
        detections.xyxy,
        detections.confidence.reshape(-1, 1)
    ))

# Convert STrack list to bounding boxes
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks])

# Match tracker outputs to YOLO detections (if needed)
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> List[int]:
    if len(detections) == 0 or len(tracks) == 0:
        return [None] * len(detections)

    detection_boxes = detections.xyxy
    tracks_boxes = tracks2boxes(tracks)

    if tracks_boxes.shape[0] == 0 or detection_boxes.shape[0] == 0:
        return [None] * len(detections)

    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

# Maps class_id -> set of unique tracker_ids
unique_ids_per_class = defaultdict(set)

# Process video and write result
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(generator, total=video_info.total_frames):
        results = model(frame)[0]

        detections = Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )

        tracks = byte_tracker.update(
            detections2boxes(detections=detections),
            frame.shape,
            frame.shape
        )

        tracker_ids = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_ids)

        # Count unique IDs
        for tid, class_id in zip(tracker_ids, detections.class_id):
            if tid is not None:
                unique_ids_per_class[class_id].add(tid)

        # Label detections
        labels = [
            f"{model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate frame
        frame = box_annotator.annotate(
            frame=frame,
            detections=detections,
            labels=labels
        )

        # Draw count
        y_offset = 50
        for class_id, ids in unique_ids_per_class.items():
            class_name = model.names[class_id]
            count_label = f"{class_name}: {len(ids)}"
            cv2.putText(
                frame,
                count_label,
                org=(50, y_offset),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=(0, 255, 0),
                thickness=3
            )
            y_offset += 50  # Move down for the next label

        sink.write_frame(frame)

# Show frames using matplotlib
def show_image(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

cap = cv2.VideoCapture(TARGET_VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open the result video.")
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # Show every 30th frame
            show_image(frame)

cap.release()

# Open final result in system media player
webbrowser.open(f'file:///{TARGET_VIDEO_PATH}')
