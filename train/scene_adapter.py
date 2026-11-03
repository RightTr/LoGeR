"""Dataset boundary for the Stage 1 experiment.

Implement iter_chunks after a scene sample is available. Yield dictionaries with:
images [B,N,3,H,W] in [0,1], depth [B,N,H,W] in camera Z units,
intrinsics [B,N,3,3] at image resolution, poses [B,N,4,4] camera-to-world,
world_to_dop [B,3,3] mapping world XY to original DOP pixels.
The yielded chunks must follow sequence order and use the configured chunk length.
"""


def iter_chunks(scene, length):
    raise NotImplementedError("Provide the scene depth, camera and DOP data layout in train/scene_adapter.py")
