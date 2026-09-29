import numpy as np
import pytest

from core.replay_buffer import ReplayBuffer

SPEC = {"x": ((2,), np.float32), "done": ((), np.bool_)}


def make(capacity=3, seed=0):
    return ReplayBuffer(capacity, SPEC, np.random.default_rng(seed))


def test_add_and_sample_shapes():
    buf = make()
    buf.add(x=[1, 2], done=False)
    batch = buf.sample(5)
    assert batch["x"].shape == (5, 2) and batch["done"].shape == (5,)
    assert (batch["x"] == [1, 2]).all()  # only one item to pick


def test_ring_overwrites_oldest():
    buf = make(capacity=3)
    for i in range(5):
        buf.add(x=[i, i], done=False)
    assert len(buf) == 3
    assert sorted(buf.data["x"][:, 0]) == [2, 3, 4]  # 0 and 1 were overwritten


def test_sampling_is_reproducible():
    a, b = make(capacity=10, seed=7), make(capacity=10, seed=7)
    for buf in (a, b):
        for i in range(10):
            buf.add(x=[i, 0], done=i % 2 == 0)
    assert np.array_equal(a.sample(8)["x"], b.sample(8)["x"])


def test_rejects_wrong_fields_and_empty_sample():
    buf = make()
    with pytest.raises(KeyError):
        buf.add(x=[0, 0])
    with pytest.raises(ValueError):
        buf.sample(1)
