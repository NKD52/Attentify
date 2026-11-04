import unittest

from tracker import FaceTracker


class TrackingTests(unittest.TestCase):
    def test_five_faces_keep_ids_when_detection_order_changes(self):
        tracker = FaceTracker()
        boxes = [(i * 100, 10, i * 100 + 40, 60) for i in range(5)]
        first = tracker.update(boxes, 0)
        self.assertEqual([t.track_id for t in first], [1, 2, 3, 4, 5])
        moved = [(x1 + 3, y1, x2 + 3, y2) for x1, y1, x2, y2 in reversed(boxes)]
        self.assertEqual([t.track_id for t in tracker.update(moved, 0.1)], [5, 4, 3, 2, 1])

    def test_gap_recovery_and_expiry(self):
        tracker = FaceTracker(max_lost_seconds=1)
        box = (10, 10, 50, 60)
        self.assertEqual(tracker.update([box], 0)[0].track_id, 1)
        self.assertEqual(tracker.update([], 0.2), [])
        self.assertEqual(tracker.update([box], 0.5)[0].track_id, 1)
        self.assertEqual(tracker.update([], 1.6), [])
        self.assertEqual(tracker.tracks, {})
        self.assertEqual(tracker.update([box], 1.7)[0].track_id, 2)

    def test_distant_face_does_not_take_old_id(self):
        tracker = FaceTracker()
        tracker.update([(0, 0, 40, 50)], 0)
        result = tracker.update([(500, 300, 540, 350)], 0.1)
        self.assertEqual(result[0].track_id, 2)

    def test_one_old_track_cannot_match_two_faces(self):
        tracker = FaceTracker()
        tracker.update([(10, 10, 50, 60)], 0)
        result = tracker.update([(11, 10, 51, 60), (15, 12, 55, 62)], 0.1)
        self.assertEqual(len({t.track_id for t in result}), 2)

    def test_backwards_time_is_rejected(self):
        tracker = FaceTracker()
        tracker.update([], 1)
        with self.assertRaises(ValueError):
            tracker.update([], 0)


if __name__ == "__main__":
    unittest.main()
