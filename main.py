from os import path as opath

from mypylibs.calibration import calibrate
from mypylibs.pose import estimate_scene_poses

ROOT_IN = 'in'
ROOT_OUT = 'out'


def process_scenes(camera: str, K):
    """Run step 5 (inter-frame pose) on every scene video for this camera."""
    scene = 0
    while True:
        scene += 1
        video = opath.join(ROOT_IN, f'{camera}_{scene:02}.mp4')
        if not opath.exists(video):
            break
        estimate_scene_poses(video, K)


def main(camera: int):

    cam = f'{camera:02}'
    video = opath.join(ROOT_IN, f'{cam}_calibration.mp4')
    if not opath.exists(video):
        raise FileNotFoundError(video)

    K, D, error = calibrate(video, ROOT_OUT, cam)
    process_scenes(cam, K)


if __name__ == '__main__':

    camera = 0
    while(True):
        camera += 1
        try:
            main(camera)
        except FileNotFoundError:
            break
