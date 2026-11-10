import os

import cv2
import numpy as np

from mypylibs.frames import PATTERN, extract_calibration_frames, to_grayscale

# Side of one checkerboard square in cm: sets the unit of everything downstream
SQUARE_SIZE = 2.5

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points(pattern=PATTERN, s=SQUARE_SIZE):
    """3D corner positions (i*s, j*s, 0) on the Z=0 board plane, in the order
    findChessboardCorners returns them (row by row)."""
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * s
    return pts


def find_corners(frames, pattern=PATTERN):
    """Sub-pixel corner locations for each frame (step 2)."""
    corners_all = []
    for frame in frames:
        gray = to_grayscale(frame)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)
        corners_all.append(corners)
    return corners_all


def mean_reprojection_error(obj_pts, img_pts, K, D, rvecs, tvecs):
    """Mean distance in pixels between detected corners and the 3D board
    corners projected back through the calibrated camera."""
    errors = []
    for obj, img, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        errors.append(np.linalg.norm(img.reshape(-1, 2) - projected.reshape(-1, 2), axis=1))
    return float(np.concatenate(errors).mean())


def calibrate(video, out_dir, camera):
    """Calibrate one camera from its checkerboard video and save K and D."""
    _, frames = extract_calibration_frames(video)
    img_pts = find_corners(frames)
    obj_pts = [board_points()] * len(img_pts)

    h, w = frames[0].shape[:2]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    error = mean_reprojection_error(obj_pts, img_pts, K, D, rvecs, tvecs)

    np.savetxt(os.path.join(out_dir, f'{camera}_K.csv'), K, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(out_dir, f'{camera}_D.csv'), D.reshape(1, -1), delimiter=',',
               fmt='%.6f', header='k1,k2,p1,p2,k3', comments='')

    print(f'Camera {camera}: {len(img_pts)} frames')
    print(f'  fx={K[0, 0]:.2f}  fy={K[1, 1]:.2f}  cx={K[0, 2]:.2f}  cy={K[1, 2]:.2f}  skew={K[0, 1]:.4f}')
    k1, k2, p1, p2, k3 = D.ravel()[:5]
    print(f'  k1={k1:.4f}  k2={k2:.4f}  k3={k3:.4f}  p1={p1:.5f}  p2={p2:.5f}')
    print(f'  mean reprojection error={error:.4f} px  (OpenCV RMS={rms:.4f} px)')

    return K, D, error
