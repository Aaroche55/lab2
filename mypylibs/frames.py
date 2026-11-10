import os

import cv2
import numpy as np

# Inner corners of the printed board (9x6 squares -> 8x5 inner corners)
PATTERN = (8, 5)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sharpness(gray):
    # Variance of the Laplacian: blurry images have few strong edges -> low variance
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_corners(gray, pattern=PATTERN):
    """Find and refine checkerboard inner corners in one grayscale frame."""
    found, corners = cv2.findChessboardCorners(gray, pattern)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def extract_calibration_frames(path, n=20, pattern=PATTERN, out_dir=None):
    """Return (indices, frames) of n sharp frames showing the checkerboard.

    Only frames where the board is detected are candidates. The candidates are
    split into n consecutive groups and the sharpest of each group is kept, so
    the selection spans the whole video and therefore different board poses.
    If out_dir is given, the chosen frames are also saved there as PNGs.
    """
    frames = read_frames(path)

    candidates = []
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(gray, pattern)
        if found:
            candidates.append((i, sharpness(gray)))

    if len(candidates) < n:
        raise ValueError(f'{path}: board found in only {len(candidates)} frames, need {n}')

    indices = []
    for group in np.array_split(np.array(candidates), n):
        indices.append(int(group[np.argmax(group[:, 1]), 0]))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for i in indices:
            cv2.imwrite(os.path.join(out_dir, f'frame_{i:04}.png'), to_grayscale(frames[i]))

    return indices, [frames[i] for i in indices]


if __name__ == '__main__':
    video = os.path.join('in', '01_calibration.mp4')
    out_dir = os.path.join('out', '01_calibration')
    indices, frames = extract_calibration_frames(video, out_dir=out_dir)
    detected = 0
    for index, frame in zip(indices, frames):
        gray = to_grayscale(frame)
        corners = find_corners(gray)
        if corners is None:
            continue

        detected += 1
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(preview, PATTERN, corners, True)
        cv2.imwrite(os.path.join(out_dir, f'corners_{index:04}.png'), preview)

    print(f'Saved {len(frames)} grayscale frames and detected corners in {detected} to {out_dir}: {indices}')

