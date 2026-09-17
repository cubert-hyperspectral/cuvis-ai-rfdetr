"""targets_from_mask multi-class path: per-class connected components + label = class_id - offset."""

import numpy as np
import torch

from cuvis_ai_rfdetr.functional import targets_from_mask


def _mask():
    # 1 shell blob (class 1), 1 fo blob (class 2), 2 fake blobs (class 3)
    m = np.zeros((1, 40, 40), np.int32)
    m[0, 2:10, 2:10] = 1  # shell -> label 0
    m[0, 2:10, 20:28] = 2  # fo    -> label 1
    m[0, 20:28, 2:10] = 3  # fake  -> label 2
    m[0, 20:28, 20:28] = 3  # fake  -> label 2 (second instance)
    return torch.from_numpy(m)


def test_single_class_default_all_zero():
    t = targets_from_mask(_mask())[0]
    assert t["labels"].tolist() == [0, 0, 0, 0]  # 4 components, all one class
    assert t["boxes"].shape == (4, 4)


def test_multiclass_labels_offset():
    t = targets_from_mask(_mask(), with_masks=True, multiclass=True)[0]
    assert sorted(t["labels"].tolist()) == [0, 1, 2, 2]  # shell0, fo1, fake2 x2
    assert t["boxes"].shape == (4, 4)
    assert t["masks"].shape == (4, 40, 40) and t["masks"].dtype == torch.bool


def test_multiclass_label_offset_zero():
    t = targets_from_mask(_mask(), multiclass=True, label_offset=0)[0]
    assert sorted(t["labels"].tolist()) == [1, 2, 3, 3]  # ids used directly

