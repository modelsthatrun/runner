# Timing images

Every image model of a task is timed on the same photo, so their speeds compare. `timing.toml` names each
task's photo and its sha256. `mtr run` uses it when you don't pass `--input`, and refuses a file whose
hash doesn't match.

| Photo | Tasks | Author | License |
|---|---|---|---|
| `dog-walk-square.jpg` (1600×1067) | object detection, instance segmentation, image matting, depth estimation, pose estimation | Retired electrician, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Moscow,_Gagarina_Square,_walking_the_dog_June_2025_01.jpg) | CC0 1.0 |

The photo is downscaled from the original to 1600 px on its long side, with its metadata removed. It
shows one person, a dog and parked cars in daylight: an ordinary scene, not a hard case.
