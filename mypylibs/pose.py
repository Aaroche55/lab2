import cv2
import numpy as np

from mypylibs.frames import read_frames, to_grayscale

# Lowe ratio test: keep a match only if it is clearly better than the 2nd-best
RATIO = 0.75
MIN_MATCHES = 8
# Frames between the two images of a pair: consecutive frames (GAP=1) barely
# move at 30fps, which makes the Essential matrix and the disparities unreliable
GAP = 20


def match_frames(gray1, gray2):
    """SIFT keypoints + descriptors, then ratio-tested matches between two frames."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None, None

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def estimate_pose(pts1, pts2, K):
    """Essential matrix from matched points, then relative R and unit t."""
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, 0

    # recoverPose picks the physically valid (R, t) among the 4 decompositions of E
    n_inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, int(n_inliers)


def estimate_scene_poses(video, K):
    """Relative (R, t) for every consecutive frame pair in a scene video.

    Returns a list of dicts: {frame_i, frame_j, R, t, inliers}.
    Pairs with too few matches are skipped.
    """
    frames = read_frames(video)
    if len(frames) < 2:
        raise ValueError(f'{video}: need at least 2 frames')

    poses = []
    for i in range(len(frames) - GAP):
        gray1 = to_grayscale(frames[i])
        gray2 = to_grayscale(frames[i + GAP])
        pts1, pts2 = match_frames(gray1, gray2)
        if pts1 is None:
            continue

        R, t, n_inliers = estimate_pose(pts1, pts2, K)
        if R is None:
            continue

        poses.append({
            'frame_i': i,
            'frame_j': i + GAP,
            'R': R,
            't': t,
            'inliers': n_inliers,
        })

    print(f'{video}: recovered pose for {len(poses)}/{len(frames) - GAP} frame pairs')
    return poses
