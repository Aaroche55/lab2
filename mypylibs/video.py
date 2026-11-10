import cv2


def shrink_video(src, dst, width=640, height=360, target_fps=30.0):
    """Resize a video and drop frames down to target_fps, writing an .mp4.

    Every video (calibration and scenes) must go through this with the same
    arguments, otherwise the intrinsics no longer match the scene frames.
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(src)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(src_fps / target_fps))  # 60fps -> keep every 2nd frame
    out_fps = src_fps / step

    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            # INTER_AREA is the recommended interpolation when downscaling
            writer.write(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
        i += 1

    cap.release()
    writer.release()


if __name__ == '__main__':
    import sys
    shrink_video(sys.argv[1], sys.argv[2])
    print(f'Wrote {sys.argv[2]}')
