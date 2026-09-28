import unittest

from app.analytics import MotionAnalyzer
from app.control.safety import DesiredCommand, SafetyLayer
from app.schemas import FaceDetection, Track
from app.tracking.iou_tracker import IoUTracker, intersection_over_union
from app.tracking.target_manager import TargetManager


class TrackerTests(unittest.TestCase):
    def test_iou_and_stable_track_id(self):
        tracker = IoUTracker(max_age=2)
        first = tracker.update([FaceDetection((10, 10, 50, 50), 0.9)], now=1.0)
        second = tracker.update([FaceDetection((12, 11, 52, 51), 0.9)], now=1.1)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertGreater(intersection_over_union((10, 10, 50, 50), (12, 11, 52, 51)), 0.8)

    def test_track_expires_after_max_age(self):
        tracker = IoUTracker(max_age=1)
        tracker.update([FaceDetection((0, 0, 20, 20), 0.9)], now=1.0)
        tracker.update([], now=1.1)
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([], now=1.2)
        self.assertEqual(len(tracker.tracks), 0)


class TargetTests(unittest.TestCase):
    def test_lock_does_not_switch_to_another_track(self):
        target = TargetManager(lost_timeout=0.5)
        selected = Track(4, (10, 10, 40, 40), 0.9, 1.0, 1.0)
        other = Track(7, (100, 10, 140, 50), 0.95, 1.0, 1.0)
        ok, _ = target.select(4, [selected, other])
        self.assertTrue(ok)
        target.update([other], now=2.0)
        target.update([other], now=2.6)
        self.assertEqual(target.target_id, 4)
        self.assertEqual(target.state, "TARGET_LOST")

    def test_reacquires_same_selected_track(self):
        target = TargetManager(lost_timeout=2.0)
        track = Track(2, (5, 5, 35, 35), 0.9, 1.0, 1.0)
        target.select(2, [track])
        target.update([], now=1.1)
        events = target.update([track], now=1.3)
        self.assertEqual(target.state, "TARGET_REACQUIRED")
        self.assertEqual(events[0].type, "TARGET_REACQUIRED")
        self.assertEqual(target.target_id, 2)


class AnalyticsTests(unittest.TestCase):
    def test_motion_center_error_and_monocular_estimate(self):
        analyzer = MotionAnalyzer()
        target = Track(1, (40, 40, 80, 80), 0.9, 1.0, 1.0)
        sample = analyzer.update(target, 200, 200, 1.0, True, 0.16, 900)
        self.assertAlmostEqual(sample["error_x"], -0.2)
        self.assertAlmostEqual(sample["error_y"], -0.2)
        self.assertAlmostEqual(sample["estimated_distance"], 3.6)
        moved = Track(1, (50, 40, 90, 80), 0.9, 1.0, 1.0)
        next_sample = analyzer.update(moved, 200, 200, 1.1, True, 0.16, 900)
        self.assertGreater(next_sample["speed"], 0)
        self.assertEqual(next_sample["direction"], "RIGHT")


class SafetyTests(unittest.TestCase):
    def test_commands_are_clamped_and_disabled_without_valid_target(self):
        safety = SafetyLayer(max_axis=0.2)
        safety.target_valid = True
        safe = safety.apply(DesiredCommand(yaw_rate=5, pitch=-2))
        self.assertEqual(safe.yaw_rate, 0.2)
        self.assertEqual(safe.pitch, -0.2)
        safety.emergency_stop()
        self.assertEqual(safety.apply(DesiredCommand(yaw_rate=0.1)).yaw_rate, 0.0)

    def test_invalid_target_suppresses_all_outputs(self):
        safety = SafetyLayer()
        self.assertEqual(safety.apply(DesiredCommand(roll=0.1)).roll, 0.0)


if __name__ == "__main__":
    unittest.main()
