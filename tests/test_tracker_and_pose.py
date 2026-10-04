import numpy as np
import pytest

from conftest import HANDS, make_hand
from gestures import Pose, classify_pose
from tracker import HandObservation, normalize_landmarks, to_pixel_space


def test_wrist_becomes_origin_and_palm_length_is_one():
    pts = HANDS["OPEN_PALM"] * 73.0 + np.array([400.0, 150.0, 12.0], dtype=np.float32)
    scaled, scale = normalize_landmarks(pts)
    assert np.allclose(scaled[0], 0.0)
    assert np.isclose(np.linalg.norm(scaled[9]), 1.0)
    assert np.isclose(scale, 73.0, rtol=1e-4)


def test_normalisation_is_distance_and_position_invariant():
    near = HANDS["PEACE"] * 140.0 + np.array([100.0, 300.0, 0.0], dtype=np.float32)
    far = HANDS["PEACE"] * 35.0 + np.array([500.0, 60.0, 0.0], dtype=np.float32)
    a, _ = normalize_landmarks(near)
    b, _ = normalize_landmarks(far)
    assert np.allclose(a, b, atol=1e-5)


def test_pixel_space_conversion_respects_aspect_ratio():
    norm = np.zeros((21, 3), dtype=np.float32)
    norm[9] = (0.5, 0.5, 0.1)
    px = to_pixel_space(norm, 640, 360)
    assert np.allclose(px[9], (320.0, 180.0, 64.0))


def test_degenerate_hand_rejected():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((21, 3)))
    obs = HandObservation.from_raw(1.0, np.zeros((21, 3)), 640, 360)
    assert not obs.present


@pytest.mark.parametrize("name", list(HANDS))
def test_every_synthetic_pose_is_recognised(cfg, name):
    lm, _ = normalize_landmarks(HANDS[name] * 90.0)
    assert classify_pose(lm, cfg.pose) == Pose(name)


@pytest.mark.parametrize("deg", [-35, -15, 15, 35])
def test_typing_poses_tolerate_hand_roll(cfg, deg):
    for name, fingers in [("POINT", (True, False, False, False)),
                          ("PEACE", (True, True, False, False)),
                          ("FIST", (False,) * 4)]:
        lm, _ = normalize_landmarks(make_hand(fingers, degrees=deg) * 90.0)
        assert classify_pose(lm, cfg.pose) == Pose(name)


def test_no_hand_has_no_pose(cfg):
    assert classify_pose(None, cfg.pose) is None
