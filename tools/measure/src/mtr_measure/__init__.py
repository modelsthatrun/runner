"""mtr: measure one checkpoint on your GPU by one protocol and write a run record (docs/protocol.md),
render its outputs on test images (outputs.py), and send records to ModelsThatRun (submit.py).

Adapters may import `weights_dir` from here to keep downloaded weights in the shared cache, and the
standard prediction types for to_prediction().
"""

from .adapter import weights_dir
from .predictions import Alpha, Boxes, Depth, Keypoints, Masks, Text

__all__ = ["weights_dir", "Boxes", "Masks", "Alpha", "Depth", "Keypoints", "Text"]
