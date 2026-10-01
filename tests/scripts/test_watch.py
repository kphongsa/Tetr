from scripts.watch import frames_to_show


def test_frames_to_show():
    frames = [{"cleared_rows": r} for r in ([], [], [21], [], [20, 21], [])]
    assert frames_to_show(frames, 0, None, False, 1) == [0, 1, 2, 3, 4, 5]
    assert frames_to_show(frames, 2, 4, False, 1) == [2, 3, 4]
    assert frames_to_show(frames, 0, None, True, 1) == [2, 4, 5]  # clears + last frame
    assert frames_to_show(frames, 0, None, True, 2) == [4, 5]
