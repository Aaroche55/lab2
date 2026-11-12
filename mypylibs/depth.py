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
    return rect1, rect2, P1, P2


def depth_map(img1, img2, K, D, R, t, baseline_cm, matcher):
    """Depth in cm, Z = f * B / d, in the rectified view of img1."""
    rect1, rect2, P1, P2 = rectify_pair(img1, img2, K, D, R, t)
    gray1, gray2 = to_grayscale(rect1), to_grayscale(rect2)

    # P2[0, 3] = -f * Tx: positive means the second camera sits to the left,
    # so it has to be passed as the left image for disparities to be positive
    if P2[0, 3] > 0:
        disp = matcher.compute(rect2, rect1)
        left_color, projection = rect2, P2
    else:
        disp = matcher.compute(rect1, rect2)
        left_color, projection = rect1, P1
    disp = disp.astype(np.float32) / 16.0  # SGBM returns fixed-point disparities

    f = P2[0, 0]
    depth = np.full(disp.shape, np.nan, np.float32)
    valid = disp > 0
    depth[valid] = f * baseline_cm / disp[valid]
    return depth, left_color, projection


def write_point_cloud(path, depth, image, projection, step=1):
    """Save one depth image as a coloured ASCII PLY point cloud (coordinates in cm)."""
    fx, fy = projection[0, 0], projection[1, 1]
    cx, cy = projection[0, 2], projection[1, 2]
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:step, 0:width:step]
    z = depth[::step, ::step]
    valid = np.isfinite(z) & (z > 0) & (z <= DEPTH_MAX_CM)
    xs, ys, z = xs[valid].astype(np.float32), ys[valid].astype(np.float32), z[valid]
    colors = image[::step, ::step][valid][:, ::-1]  # PLY stores RGB, OpenCV uses BGR
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.column_stack((x, y, z))

    with open(path, 'w', encoding='ascii') as ply:
        ply.write('ply\nformat ascii 1.0\n')
        ply.write(f'element vertex {len(points)}\n')
        ply.write('property float x\nproperty float y\nproperty float z\n')
        ply.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for (px, py, pz), (red, green, blue) in zip(points, colors):
            ply.write(f'{px:.3f} {py:.3f} {pz:.3f} {int(red)} {int(green)} {int(blue)}\n')
    print(f'{path}: saved {len(points)} coloured points')


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
    """Generate a depth video and export the best-supported cloud frame."""
    frames = read_frames(video)
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)

    cm_per_frame = load_travel_cm(video) / (len(frames) - 1)
    matcher = make_matcher()

    writer = DepthVideoWriter(out_path, fps, frames[0].shape[:2])
    cloud_path = os.path.splitext(out_path)[0] + '.ply'

    best_count = 0
    best_cloud = None
    best_pair = None

    for pose in poses:
        i, j = pose['frame_i'], pose['frame_j']
        baseline_cm = cm_per_frame * (j - i)

        depth, cloud_image, projection = depth_map(
            frames[i],
            frames[j],
            K,
            D,
            pose['R'],
            pose['t'],
            baseline_cm,
            matcher,
        )

        # Use the same validity filter as the point-cloud exporter.
        valid = (
            np.isfinite(depth)
            & (depth > 0)
            & (depth <= DEPTH_MAX_CM)
        )
        count = int(np.count_nonzero(valid))

        if count > best_count:
            best_count = count
            best_cloud = (
                depth.copy(),
                cloud_image.copy(),
                projection.copy(),
            )
            best_pair = (i, j)

        writer.write(depth, f'Frames {i}-{j}')

    writer.close()

    if best_cloud is not None:
        write_point_cloud(cloud_path, *best_cloud)
        print(
            f'{video}: selected frames '
            f'{best_pair[0]}-{best_pair[1]} '
            f'with {best_count} valid cloud points'
        )
    else:
        # Remove any empty or outdated cloud from an earlier run.
        if os.path.exists(cloud_path):
            os.remove(cloud_path)

        print(
            f'WARNING: {video}: no valid cloud points within '
            f'0-{DEPTH_MAX_CM} cm in any processed pair. '
            'No point cloud exported.'
        )

    print(
        f'{video}: wrote {out_path} '
        f'(baseline {cm_per_frame:.3f} cm per frame)'
    )
