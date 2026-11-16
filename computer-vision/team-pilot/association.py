"""Conservative geometric proposals, not verified student ownership."""
from threading import Lock


def associate(box, faces):
    x1, y1, x2, y2 = box
    # SCB boxes describe student regions, not hands. Limit matching to upper 60%.
    candidates = [f['track_id'] for f in faces
                  if x1 <= (f['box'][0] + f['box'][2]) / 2 <= x2
                  and y1 <= (f['box'][1] + f['box'][3]) / 2 <= y1 + .6 * (y2-y1)]
    return {'track_id': candidates[0] if len(candidates) == 1 else None,
            'association_status': 'geometric_candidate' if len(candidates) == 1
            else 'ambiguous' if candidates else 'unassigned',
            'candidate_track_ids': candidates}


def result_is_fresh(captured_at, now, ttl):
    return 0 <= now - captured_at <= ttl


class LatestSlot:
    """One pending frame. In-flight inference is never queued or cancelled."""
    def __init__(self):
        self.lock = Lock()
        self.pending = None
        self.overwritten = 0

    def put(self, item):
        with self.lock:
            if self.pending is not None:
                self.overwritten += 1
            self.pending = item

    def take(self):
        with self.lock:
            item, self.pending = self.pending, None
            return item
