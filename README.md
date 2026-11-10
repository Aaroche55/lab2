# CS6524: Computer Vision — Lab 2: Camera Calibration and Stereo Vision

**Immersive Software Engineering (ISE) — University of Limerick**  

**Instructor:** Juan Albarracín [✉️](mailto:Juan.Albarracin@ul.ie)\
**Teacher Assistant:** Thendral Balakrishnan [✉️](mailto:balakrishnan.thendral@ul.ie)\
**Module Lead:** James Patten [✉️](mailto:James.Patten@ul.ie)

**Deadline:** Sunday, November 15th at Midnight (Autumn Semester Week 3)

---

## 🎯 Objective

Learn how to calibrate cameras, how to find their parameters, and use them to estimate the depth of the objects in a video using stereo vision.

---

## 📌 Context

Camera intrinsics lie in a $3 \times 3$ matrix that dictates how a specific camera projects 3D points from the real world into the 2D image plane. Using the operations learned in today's lecture, it is possible to know the trajectory of a camera with respect to an arbitrary starting position, just by checking the frames it captures, and using this information simulate an inter-frame stereo setup to perform dense depth estimation.

This document will guide you through the following steps:
- Record a calibration checkerboard video with your phone.
- Use OpenCV functions to:
    - Extract the corners of the checkerboard.
    - Calculate the intrinsics of your camera.
- Record a new video with a scene depicting different depths.
- Using the intrinsics, calculate the relative position of the camera between every two consecutive frames.
- Use the information above to calculate the depth of every pixel using stereo vision.

---

## 🔄 Step-by-Step Guide

### 1. Record a Checkerboard Video & Extract 20 Calibration Frames

- Mount the provided A4 checkerboard securely onto a rigid, flat surface (such as a clipboard or foam board) to prevent warping.
- Record a 5 to 10-second video with your smartphone camera moving smoothly around the target from various distances, tilt angles, and perspective orientations. Ensure that autofocus and auto-exposure are locked prior to recording.
    - The camera must remain still: the only moving object should be the checkerboard.
- Check the video file size: ensure that it does not exceed 5MB.
    - If it does, either configure your camera to take low resolution videos (up to $640 \times 360$ at 30fps), or use `cv2.resize` and video i/o OpenCV functions to create a new video.
    - ⚠️ Make sure the post-calibration videos that you record undergo **exactly the same** pre-processing.
- Extract at least 20 sharp, non-blurry frames spanning diverse perspective views of the checkerboard.

### 2. Extract the corners of the checkerboard of the frame

- For each frame, convert the image to grayscale and detect the inner corners using `cv2.findChessboardCorners`.
- [Optional] Refine corner locations to sub-pixel precision using cv2.cornerSubPix().

### 3. Find the Intrinsics of Your Camera

- Construct the real-world 3D coordinates $P_i = (X, Y, Z)^T$ for the $9 \times 6$ inner grid on the $Z=0$ plane: $(i \cdot s, j \cdot s, 0)$ where $s = 2.5\text{ cm}$.
    - ⚠️ Be aware of the value of $s$, as it can be anything (meters, milimeters, etc.) and OpenCV will not warn you about the ascale you use. Defining $s = 2.5\text{ cm}$ is important because you will need to estimate the depth of the objects in centimeters.
- Pass the array of 3D object points and corresponding 2D sub-pixel image coordinates from all 20 frames to cv2.calibrateCamera().
- Extract and log the following parameters in your report:
    - Intrinsic Matrix ($K$): Focal lengths $(f_x, f_y)$, principal point $(c_x, c_y)$, and skew.
    - Distortion Coefficients ($D$): Radial parameters $(k_1, k_2, k_3)$ and tangential parameters $(p_1, p_2)$.
    - Reprojection Error: Compute the mean pixel reprojection error (target $\le 0.5\text{ pixels}$).
- Save your calibrated $K$ matrix and distortion vector $D$ into a text file using `numpy.savetxt`.

### 4. Record a Scene Video with Depth Variation

- Select a static physical scene containing distinct, well-defined depth layers (e.g., foreground objects at 0.5m, midground elements at 1.5m, and background elements at 3m+). Avoid scenes dominated by textureless, uniform surfaces (like blank walls).
- Record a 3 to 5-second video moving the camera along a smooth, sideways horizontal translation path to create a physical baseline.
    - Recording Constraints:
        - Keep focus and exposure locked.
        - Avoid rapid camera rotations or sudden panning movements, as pure rotation degrades inter-frame translation estimation.
    - ⚠️ Double-check that you performed on this video the same pre-processing you applied to the calibration video to reduce the file size.

### 5. Get Inter-Frame Translation & Pose Matrices
For every two consecutive frame pairs $(I_t, I_{t+1})$:
- Use `cv2.SIFT_create()` to:
    - Detect detect keypoints (i.e., corners) on the image.
    - Compute feature vectors of each keypoint.
    - Match the keypoints between the two frames.
- Once you know how the keypoints moved from one frame to the other, you can estimate the *Fundamental Matrix* $F$ that transforms one frame into the other. Use `cv2.findFundamentalMat` for that, by providing the list of matched point between frames.
- Compute the *Essential Matrix* $E$ using your calibrated camera intrinsics: $E = K^T F K$ (or directly via `cv2.findEssentialMat`).
- Decompose $E$ into the relative rotation matrix $R_{t, t+1}$ and translation unit vector $t_{t, t+1}$ using `cv2.recoverPose`.

### 6. Dense Stereo Vision & Depth Map Calculation
- Using the estimated relative pose $(R, t)$ and intrinsic matrix $K$, rectify consecutive frame pairs using cv2.stereoRectify() and cv2.initUndistortRectifyMap() / cv2.remap() so that epipolar lines align horizontally across scanlines.
- Compute dense disparity maps $d(x, y) = x_{\text{left}} - x_{\text{right}}$ across rectified pairs using OpenCV's StereoBM or StereoSGBM block matching algorithms.
- Convert disparity values into continuous physical depth (in centimeters): $$Z(x, y) = \frac{f \cdot B}{d(x, y)}$$ where $f$ is the focal length (in pixels) and $B = \|t\|$ is the inter-frame baseline distance.
- Plot the depth values as a heatmap and provide a color scale in centimenters (use `matplotlib` for this).

### 🎁 Bonus (+0.5 Grade Point)
Include a [point cloud](https://en.wikipedia.org/wiki/Point_cloud) of the scene in `.ply` format, just one from one of the depth frames per scene. Make sure that the clouds can be smoothly visualized from [CloudCompare](https://www.cloudcompare.org/) and, importantly, that the scene is decently depicted, for which you might need to choose the points you want to send to the cloud smartly. 

---

## 📋 Submission Guidelines and Rules

### 1. Repository Setup and Student Identification
- **Forking Requirement:** You must fork the official starter repository into your own GitHub account or team space.
- **Repository Naming Convention:** Rename your forked repository to: `cs6524-lab2-group<XX>`
- **Identification (`TEAM.md`):** Add a `TEAM.md` file in the repository root with the full names, student IDs, and GitHub usernames of all team members before making your final push.

### 2. Code Structure and Execution
- **Main Entry Point:** All execution must be triggered by running `python main.py` in the repository root without any command-line arguments.
- **Custom Modules:** Any additional helper scripts or custom functions must be placed inside the `mypylibs/` directory.
- **Allowed and Prohibited Libraries:** All the libraries present in the docker image are allowed, without any function restriction. The image for this lab has OpenCV.

### 3. Input Specifications (`in/`)
All the members must use their own cameras so, for each member's camera, provide **one** calibration video, **one** set of intrinsics, and at **least two** test scenes to stimate depth with everythin must go in the `in/` directory. Details below:

- **Naming Convention:** `01_calibration.mp4`, `01_01.mp4`, `01_02.png`, meaning respectively, calibration of camera $1$, test scene $1$ of camera $1$, and test scene $2$ of camera $1$. For camre $2$: `02_calibration.mp4`, `02_01.mp4`, `02_02.png`, and so on for each member of the group.
- **Optional Configuration:** Not specific per-scene/camera configuration is foreseen, but if you consider including custom paramenters, you can add them to `in/` (e.g., `01_01.yaml`, or `01_calibration.yaml`, as it bst suits to you).
- **Video Rules and Resolution Limits:**
    - Select a static physical scenes containing distinct, well-defined depth layers (e.g., foreground objects at 0.5m, midground elements at 1.5m, and background elements at 3m+). Avoid scenes dominated by textureless, uniform surfaces (like blank walls).
    - Check the video file size: ensure that it does not exceed 5MB.
        - If it does, either configure your camera to take low resolution videos (up to $640 \times 360$ at 30fps), or use `cv2.resize` and video i/o OpenCV functions to create a new video (with less frames if necessary).
        - ⚠️ Make sure the post-calibration videos that you record undergo **exactly the same** pre-processing.
    - You can loosen the rules above a bit, as long as all the required inputs run on Actions smoothly.

### 4. Output Specifications (`out/`)
1. `XX_K.csv`: Intrinsic Matrix from camera `XX` in human-readable format (use `numpy.savetxt`)
2. `XX_D.csv`: Distortion Coeficients from camera `XX`, same as $K$.
3. `XX-YY.mp4`: Inter-frame depth scene from camera `XX`'s scene `YY`. The video must contain a colormap with clear scale in centimeters.
4. [Just for the bonus] `XX-YY.ply`: Point cloud of **one** of the frames from `XX-YY.mp4`, depicting the scene.

### 5. Written Report and Assets (`report.tex` and `assets/latex/`)
- *File Location and Template:* Write your report in `report.tex` in the repository root, using the provided IEEE Transactions LaTeX template.
- **Report Assets (`assets/latex/`):** Place all figures, sub-diagrams, and external files needed for your LaTeX report inside the `assets/latex/` directory to keep the root directory tidy.
- **Length Limit:** Maximum 3 pages (including figures and references).
- **Required Content:**
    - **Methodology:** Concise description of the whole pipeline with particular considerations you find necessary to add.
    - **Discussion and Failure Analysis:** Critical analysis of the results, whether the pipeline succeeded or failed, along with concrete steps for improvement if given more time.
- **Writing Quality and Authenticity:** Be brief, rigorous, and specific. Your analysis must convince the reader that you personally conducted the experiments, understood the mathematical trade-offs, and derived the insights yourself. Avoid pasting generic, boilerplate text from LLMs—reports that read like unverified AI summaries will be flagged and incur heavy grade deductions.

### 6. CI/CD Pipeline, Local Testing and Bonus Point
- **Automated Workflow:** GitHub Actions will execute `main.py` and compile `report.tex` into `report.pdf` upon pushing to `main`.
- **Local Docker Testing:** It is strongly recommended to test your code locally using the provided Docker environment before pushing to ensure pipeline compatibility.
