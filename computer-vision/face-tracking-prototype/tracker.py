"""Temporary geometric tracking, with no appearance features or recognition."""

from dataclasses import dataclass
from math import hypot

Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    """Intersection over union for (left, top, right, bottom) boxes."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Box
    last_seen: float


class FaceTracker:
    """Greedily match boxes by overlap and distance relative to face size.

    Returns only tracks matched in the current frame. Lost tracks remain in
    memory briefly for reassociation but are never drawn as current detections.
    """

    def __init__(self, max_lost_seconds: float = 1.0, max_distance: float = 1.0):
        if max_lost_seconds < 0 or max_distance <= 0:
            raise ValueError("Lost timeout must be >= 0 and distance must be > 0")
        self.max_lost_seconds = max_lost_seconds
        self.max_distance = max_distance
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.last_timestamp = -1.0

    def update(self, boxes: list[Box], timestamp: float) -> list[Track]:
        """Update once per sampled frame; timestamp is seconds in source time."""
        if timestamp < self.last_timestamp:
            raise ValueError("Tracker timestamps must not go backwards")
        self.last_timestamp = timestamp
        self.tracks = {
            key: track for key, track in self.tracks.items()
            if timestamp - track.last_seen <= self.max_lost_seconds
        }
        candidates = []
        for key, track in self.tracks.items():
            a = track.box
            for index, b in enumerate(boxes):
                size = max(hypot(a[2] - a[0], a[3] - a[1]),
                           hypot(b[2] - b[0], b[3] - b[1]), 1.0)
                distance = hypot((a[0] + a[2] - b[0] - b[2]) / 2,
                                 (a[1] + a[3] - b[1] - b[3]) / 2) / size
                overlap = iou(a, b)
                if distance <= self.max_distance:
                    candidates.append((distance + 1 - overlap, key, index))
        matched_tracks, matched_boxes = set(), set()
        visible = {}
        for _, key, index in sorted(candidates):
            if key in matched_tracks or index in matched_boxes:
                continue
            track = self.tracks[key]
            track.box, track.last_seen = boxes[index], timestamp
            matched_tracks.add(key)
            matched_boxes.add(index)
            visible[index] = track
        for index, box in enumerate(boxes):
            if index not in matched_boxes:
                track = Track(self.next_id, box, timestamp)
                self.tracks[self.next_id] = track
                self.next_id += 1
                visible[index] = track
        return [visible[index] for index in range(len(boxes))]
