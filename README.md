# video2mesh

`video2mesh` turns a video file into:

- a 3D mesh of the scene (`.ply` and `.obj`)
- a colored point cloud (`.ply`)
- the camera pose of every reconstructed frame (`.json` and `.csv`)
- the raw COLMAP sparse model (binary and text)

It follows the usual photogrammetry path. Frames are extracted from the video, COLMAP runs feature extraction, sequential matching (the matcher meant for video), and mapping, images are undistorted, a dense cloud is built, and Open3D turns that cloud into a mesh.

Monocular video has no absolute scale. Camera positions and the mesh are in COLMAP's coordinate frame, defined up to a similarity transform (rotation, translation, and a single scale).

## Windows setup

These steps assume the project lives at `D:\work\cur\video2mesh`. Any folder is fine.

### 1. Python

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/windows/) and check **Add python.exe to PATH**.

```powershell
cd D:\work\cur\video2mesh
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`opencv-python-headless` reads most videos. ffmpeg is still worth installing: it is the preferred decoder, and it handles phone clips that OpenCV refuses.

### 2. ffmpeg

```powershell
winget install Gyan.FFmpeg
```

Or download the essentials zip from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/), unzip it, and add the `bin` folder (the one that contains `ffmpeg.exe`) to your user PATH.

Check:

```powershell
ffmpeg -version
```

If you would rather not touch PATH, set the full path for one terminal:

```powershell
$env:FFMPEG_EXE = "C:\tools\ffmpeg\bin\ffmpeg.exe"
```

### 3. COLMAP

Download a Windows build from the [COLMAP releases](https://github.com/colmap/colmap/releases).

- **CUDA build** if you have an NVIDIA GPU and a current driver. This is the one that can run patch-match stereo, which is the best dense cloud.
- **no-CUDA build** otherwise. Feature extraction, sequential matching, and mapping still run on the CPU. video2mesh then builds the dense cloud with its own CPU fallback (OpenCV stereo fused with the sparse points) instead of patch-match.

Release zips are named along the lines of `COLMAP-<version>-windows-cuda.zip` and `COLMAP-<version>-windows-no-cuda.zip`. Older releases used `colmap-x64-windows-cuda.zip` and `colmap-x64-windows-no-cuda.zip`. Pick the matching zip for your GPU, not both.

Unzip it, for example to `C:\tools\COLMAP`. Add the folder that contains `colmap.exe` or `colmap.bat` to PATH, then open a **new** terminal and check:

```powershell
colmap -h
```

The help banner says `with CUDA` or `without CUDA`. That is what `--gpu auto` looks at.

PATH alternative:

```powershell
$env:COLMAP_EXE = "C:\tools\COLMAP\colmap.bat"
$env:Path = "C:\tools\COLMAP;" + $env:Path
```

`colmap.bat` wrappers are supported. You can also pass `--colmap-exe` on the command line.

COLMAP 3.x and 4.x are both supported. Version 4 moved `max_image_size` and `use_gpu` from `SiftExtraction` / `SiftMatching` to `FeatureExtraction` / `FeatureMatching`. video2mesh reads each subcommand's `-h` output and passes the flag that binary still has, so a 4.2 CUDA build and an older no-CUDA build use the same command.

Optional: `pip install pycolmap` and `--backend pycolmap` if you cannot install the executable. The executable is the path this project is tested with, and it is the straightforward install on Windows.

### 4. Run it

```powershell
video2mesh reconstruct D:\videos\room.mp4 -o D:\work\cur\video2mesh\out --fps 2 --max-frames 200
```

If the `video2mesh` command is not found, either reinstall with `pip install -e .` from the project folder, or call the module:

```powershell
python -m video2mesh reconstruct D:\videos\room.mp4 -o D:\work\cur\video2mesh\out --fps 2 --max-frames 200
```

A first pass on a new clip should be cheap:

```powershell
python -m video2mesh reconstruct D:\videos\room.mp4 -o D:\work\cur\video2mesh\out_fast --fps 1 --max-frames 80 --quality fast --overwrite
```

Then raise quality once that succeeds.

Open the result (needs a desktop session, not a remote terminal without a display). The window is Open3D, so the `opencv-python-headless` install is enough; the viewer does not call `cv2.imshow`.

```powershell
python -m video2mesh view D:\work\cur\video2mesh\out
python -m video2mesh view D:\work\cur\video2mesh\out --frame 12
python -m video2mesh view --help
```

The same key list is printed in the terminal when the window opens:

| Key | Action |
| --- | --- |
| W / S | Pitch up / down around the look-at point. A held key repeats. |
| A / D | Yaw left / right around the look-at point. |
| Z / C | Roll clockwise / counter-clockwise around the view direction. Same step as pitch and yaw; a held key repeats. |
| Q / E | Zoom in / out. The eye moves and the look-at point stays. |
| Up / Down | Dolly forward / backward along the view. The eye and the look-at point move together, by a step proportional to the scene size. |
| R | Reset the view. |
| Left / Right, N / P, [ / ] | Previous / next registered frame. |
| F, or Show image | Draw that frame's photo, undistorted, on its camera plane and look through the camera. The mesh turns transparent, and only that camera stays visible. |
| H | Hide or show the projected points. |
| V | Color points by their RGB or by depth. |
| G | Switch the overlay between sparse points and `pointcloud.ply`. |

Mouse left-drag still orbits, right-drag pans, and the wheel zooms. Show image, Feature points, and the Mesh alpha slider are at the top of the side panel. Feature points draws the COLMAP sparse points in the scene. The photo in the 3D view is undistorted onto the camera plane, so those points and the mesh line up with it. Mesh alpha changes how solid the mesh is while the photo is showing. In that mode only the current camera frustum stays visible. The side panel draws the original `out/frames/<filename>` and projects points with that frame's pose and intrinsics, distortion included. `out/dense/images` are not used for the side panel. Sparse points come from the model named in `cameras.json`. The viewer prefers `<output>/sparse/<folder name>` when that folder exists, so a moved output directory still opens the right model, and falls back to the absolute path in the file. Extra COLMAP 4 files (`frames.bin`, `rigs.bin`, and the text copies) are ignored.

## What you get

```
out/
  frames/                  JPEGs that went into COLMAP
  frames_manifest.json     timestamp and blur score for each JPEG
  database.db              COLMAP feature database
  sparse/0/                raw mapper output (.bin). Extra models, if any, are sparse/1, ...
  dense/                   undistorted images and, with CUDA, patch-match workspace
  pointcloud.ply
  mesh.ply
  mesh.obj
  mesh_bpa.ply             only with --mesh both
  mesh_bpa.obj
  cameras.json
  cameras.csv
  log.txt
  run_config.json
```

`cameras.json` has one entry per registered frame:

- `filename`, timestamp, blur score, width, height
- `camera_model` and `intrinsics` (`fx`, `fy`, `cx`, `cy`, and distortion terms when the model has them)
- `rotation_quaternion_wxyz` as `(qw, qx, qy, qz)`
- `translation` `(tx, ty, tz)`
- `camera_center_world`
- `world_to_camera`, the 3×4 matrix `[R | t]`

COLMAP's convention is world-to-camera: `X_cam = R X_world + t`. The camera center is `C = -R^T t`. The same columns are in `cameras.csv`.

Frames that COLMAP could not register are listed in `unregistered_frames` and omitted from `frames`.

`sparse_model` in `cameras.json` is the model that was meshed (the largest one, when mapping produces more than one). That folder holds COLMAP's binary files and a text copy: `cameras.txt`, `images.txt`, and `points3D.txt`. COLMAP 4 also writes `frames` and `rigs` (`.bin` and `.txt`) beside those files, including under `dense/sparse`. video2mesh only reads cameras, images, and points3D, so the extra files are left untouched.

## Command

```text
video2mesh reconstruct VIDEO -o OUTPUT_DIR [options]
video2mesh view OUTPUT_DIR [--scale METERS] [--point-cloud] [--frame N]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--fps` | 2 | Frames kept per second. Never faster than the video, so frames are not duplicated. |
| `--max-frames` | 300 | Cap on images sent to COLMAP. |
| `--blur-threshold` | 0 | Drop frames with Laplacian variance below this. `0` turns the filter off. Try 40–80 for 1080p. |
| `--quality` | medium | `fast`, `medium`, or `high`: image size, feature count, match overlap, Poisson depth. |
| `--dense` / `--no-dense` | dense on | `--no-dense` meshes the sparse points only. |
| `--mesh` | poisson | `poisson`, `bpa` (ball pivoting), or `both`. |
| `--camera-model` | SIMPLE_RADIAL | COLMAP model shared by every frame. Use `OPENCV` for a wide phone lens. Avoid `PINHOLE` on short clips: separate `fx` and `fy` can drift apart. |
| `--max-image-size` | preset | Long-side cap in pixels. Overrides the preset. |
| `--poisson-trim` | 0.05 | Fraction of lowest-density Poisson vertices to cut off. |
| `--poisson-depth` | preset | Octree depth. Higher is more detail and more noise. |
| `--gpu` | auto | `auto`, `on`, or `off`. `on` requires a CUDA build of COLMAP. |
| `--backend` | auto | `cli` (the `colmap` executable) or `pycolmap`. |
| `--overwrite` | off | Replace a non-empty output directory. |
| `--view` | off | Open the viewer at the end of a successful run. |
| `--frame` | 1 | `view` only. 1-based registered frame to select when the window opens. |
| `--colmap-exe`, `--ffmpeg-exe` | PATH | Explicit tool paths. |

Exit code `2` means COLMAP or ffmpeg is missing. Exit code `1` means the video or the reconstruction failed.

Progress is plain log lines, so it stays readable in PowerShell and in `log.txt`. Each step starts with a banner such as `Stage 3 of 8: matching` and ends with how long it took. Frame extraction counts images (`extracted 42/150`). COLMAP prints processed images, match blocks, and registered images as they happen. If a COLMAP step goes quiet, a heartbeat with the elapsed time is written about every 20 seconds. The full tool output is still in `log.txt`.

A successful run ends with a summary: frames extracted, cameras registered, point-cloud size, mesh vertex and triangle counts, the output paths (`mesh.ply`, `cameras.json`, and the rest), and the wall time for the whole run and for each stage. A failed run names the stage, the command that exited, and one next step.

## Dense cloud

1. If COLMAP was built with CUDA, the dense step is `patch_match_stereo` plus `stereo_fusion`.
2. Otherwise (or if patch-match fails) consecutive undistorted frames are rectified with the recovered poses and matched with OpenCV's semi-global block matcher. Those points are fused with the sparse COLMAP points, voxel-downsampled, and cleaned.
3. `--no-dense` skips both and meshes the sparse cloud.

CPU stereo is coarser than patch-match. It is there so a machine without an NVIDIA GPU still produces a mesh.

## Mesh

Poisson reconstruction (Open3D) builds a closed surface, trims low-density vertices, crops to the point cloud, and keeps the largest connected piece. Normals are turned toward the cameras before meshing. Ball pivoting is an alternative that stays closer to the points and can leave holes.

## Capturing video that actually reconstructs

- Move slowly. Neighboring moments should show almost the same scene.
- Translate. Walking sideways or orbiting an object works. Turning in place does not, because there is no parallax.
- Overlap. A good test is that any patch of wall is visible in many frames.
- Texture. Blank walls, sky, water, glass, and mirrors produce few matches. Posters, furniture, brick, and clutter help.
- Light. Keep exposure stable. Flicker and blown-out windows break features. Avoid motion blur; raise the shutter speed rather than relying on `--blur-threshold` alone.
- Lock focus if the camera lets you. A focus hunt looks like a different scene to COLMAP.
- 720p or 1080p is enough. More sharp frames beat more blurry frames.
- Start with `--quality fast` on a 30–60 second clip before an overnight `--quality high` run.

## Linux and macOS

```bash
sudo apt install ffmpeg colmap    # Debian/Ubuntu
brew install ffmpeg colmap        # macOS
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m video2mesh reconstruct clip.mp4 -o out --fps 2 --quality fast
```

The same `--gpu auto` switch applies: distribution COLMAP packages are often built without CUDA.

## Synthetic clip and tests

`scripts/make_sample_video.py` renders a textured room with a moving pinhole camera. It is a stand-in for a real capture, not a measurement of accuracy.

```bash
python scripts/make_sample_video.py -o sample_room.mp4 --frames 16 --width 640 --height 480
python -m video2mesh reconstruct sample_room.mp4 -o out --fps 8 --max-frames 16 --quality fast --overwrite
```

Unit tests do not need COLMAP:

```bash
pip install pytest
pytest
```

`python scripts/smoke_e2e.py` runs the full pipeline on that synthetic room and checks that `mesh.ply`, `cameras.json`, and the COLMAP text model exist. It needs COLMAP on PATH.

A no-CUDA run of that clip (16 frames, 640×480, default `SIMPLE_RADIAL` model, `--quality fast`) registered every frame, estimated a focal length within about one percent of the rendered camera, and wrote a Poisson mesh from the CPU stereo cloud. Patch-match stereo is skipped unless COLMAP reports CUDA.

## Layout

| Module | Role |
| --- | --- |
| `video2mesh/frames.py` | ffmpeg or OpenCV extraction, resize, blur filter |
| `video2mesh/sfm.py` | COLMAP CLI or pycolmap: extract, sequential match, map, undistort |
| `video2mesh/dense.py` | patch-match or CPU stereo |
| `video2mesh/mesh.py` | Poisson and ball pivoting |
| `video2mesh/export.py` | `.ply`, `.obj`, JSON, CSV |
| `video2mesh/viewer.py` | Open3D window: mesh, frustums, frame overlay |
| `video2mesh/viewmath.py` | projection, orbit, and viewer key map |
| `video2mesh/cli.py` | argument parser |

## When reconstruction fails

The error text repeats the short version of this list. Usual causes:

- too few registered images (move slower, add texture, lower `--fps`)
- COLMAP not on PATH (see the Windows section above; the message includes it)
- a no-CUDA build together with `--gpu on` (use `--gpu auto` or install the CUDA zip)
- an empty Poisson mesh (try `--poisson-trim 0` or `--mesh bpa`, or capture more baseline)

The sparse model and `cameras.json` are written before meshing. If only the mesh step fails, the poses are still in the output folder.
