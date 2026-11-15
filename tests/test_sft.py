import torch

from scripts.sft import make_collate


def test_collate_pads_inputs_and_masks_padding_labels():
    collate = make_collate(pad_id=0)
    input_ids, labels = collate([
        ([1, 2, 3], [-100, 2, 3]),
        ([1, 4], [-100, 4]),
    ])
    assert input_ids.tolist() == [[1, 2, 3], [1, 4, 0]]
    assert labels.tolist() == [[-100, 2, 3], [-100, 4, -100]]
