import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from mypylibs.frames import read_frames, to_grayscale

# recoverPose only gives the direction of t, so the real baseline has to come
# from outside: the distance the camera travelled over the whole clip, in cm.
# Override per scene with `travel_cm: <value>` in in/XX_YY.yaml.
DEFAULT_TRAVEL_CM = 30.0

# Fixed colour scale for every frame so the video does not flicker
DEPTH_MIN_CM = 0.0
DEPTH_MAX_CM = 250.0

NUM_DISPARITIES = 64  # must be a multiple of 16
BLOCK_SIZE = 5


def load_travel_cm(video):
    config = os.path.splitext(video)[0] + '.yaml'
    if os.path.exists(config):
        with open(config) as f:
            params = yaml.safe_load(f) or {}
        if 'travel_cm' in params:
            return float(params['travel_cm'])
    return DEFAULT_TRAVEL_CM


def make_matcher():
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE ** 2,
        P2=32 * BLOCK_SIZE ** 2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def rectify_pair(img1, img2, K, D, R, t):
    """Warp both frames so that epipolar lines become horizontal scanlines."""
    h, w = img1.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, D, K, D, (w, h), R, t, alpha=0)
    map1 = cv2.initUndistortRectifyMap(K, D, R1, P1, (w, h), cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(K, D, R2, P2, (w, h), cv2.CV_32FC1)
    rect1 = cv2.remap(img1, *map1, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, *map2, cv2.INTER_LINEAR)
    return rect1, rect2, P2


def depth_map(img1, img2, K, D, R, t, baseline_cm, matcher):
    """Depth in cm, Z = f * B / d, in the rectified view of img1."""
    gray1, gray2 = to_grayscale(img1), to_grayscale(img2)
    rect1, rect2, P2 = rectify_pair(gray1, gray2, K, D, R, t)

    # P2[0, 3] = -f * Tx: positive means the second camera sits to the left,
    # so it has to be passed as the left image for disparities to be positive
    if P2[0, 3] > 0:
        disp = matcher.compute(rect2, rect1)
    else:
        disp = matcher.compute(rect1, rect2)
    disp = disp.astype(np.float32) / 16.0  # SGBM returns fixed-point disparities

    f = P2[0, 0]
    depth = np.full(disp.shape, np.nan, np.float32)
    valid = disp > 0
    depth[valid] = f * baseline_cm / disp[valid]
    return depth


class DepthVideoWriter:
    """Renders depth maps as matplotlib heatmaps with a cm colour bar."""

    def __init__(self, path, fps, shape):
        h, w = shape
        self.fig, self.ax = plt.subplots(figsize=(w / 80 + 1.5, h / 80), dpi=80)
        cmap = plt.get_cmap('turbo_r').copy()
        cmap.set_bad('black')  # pixels with no valid disparity
        self.im = self.ax.imshow(np.zeros(shape), cmap=cmap, vmin=DEPTH_MIN_CM, vmax=DEPTH_MAX_CM)
        self.ax.axis('off')
        self.fig.colorbar(self.im, ax=self.ax, label='Depth (cm)')
        self.fig.tight_layout()
        self.fig.canvas.draw()
        size = self.fig.canvas.get_width_height()
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    def write(self, depth, title):
        self.im.set_data(np.ma.masked_invalid(depth))
        self.ax.set_title(title)
        self.fig.canvas.draw()
        rgb = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()
        plt.close(self.fig)


def scene_depth_video(video, poses, K, D, out_path):
    """Step 6: one depth map per frame pair from step 5, written as a video."""
    frames = read_frames(video)
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
    cm_per_frame = load_travel_cm(video) / (len(frames) - 1)  # assumes constant speed
    matcher = make_matcher()

    writer = DepthVideoWriter(out_path, fps, frames[0].shape[:2])
    for pose in poses:
        i, j = pose['frame_i'], pose['frame_j']
        baseline_cm = cm_per_frame * (j - i)
        depth = depth_map(frames[i], frames[j], K, D, pose['R'], pose['t'], baseline_cm, matcher)
        writer.write(depth, f'Frames {i}-{j}')
    writer.close()

    print(f'{video}: wrote {out_path} (baseline {cm_per_frame:.3f} cm per frame)')
