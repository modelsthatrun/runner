"""mtr: measure one checkpoint on your GPU by one protocol and write a run record (docs/protocol.md),
render its outputs on test images (outputs.py), and send records to ModelsThatRun (submit.py).

Adapters may import `weights_dir` from here to keep downloaded weights in the shared cache, `device` for the device
the call runs on ("cuda", or "cpu" for a CPU row), and the standard prediction types for to_prediction().
"""

from .adapter import device, weights_dir
from .predictions import Alpha, Boxes, Depth, Keypoints, Masks, Text

__all__ = ["weights_dir", "device", "Boxes", "Masks", "Alpha", "Depth", "Keypoints", "Text"]
