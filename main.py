from os import path as opath

from mypylibs.calibration import calibrate

ROOT_IN = 'in'
ROOT_OUT = 'out'


def main(camera: int):

    cam = f'{camera:02}'
    video = opath.join(ROOT_IN, f'{cam}_calibration.mp4')
    if not opath.exists(video):
        raise FileNotFoundError(video)

    K, D, error = calibrate(video, ROOT_OUT, cam)


if __name__ == '__main__':

    camera = 0
    while(True):
        camera += 1
        try:
            main(camera)
        except FileNotFoundError:
            break
