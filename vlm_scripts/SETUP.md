# Running this on a different machine

Two separate git repos are needed, cloned to specific locations (the scripts assume these paths):

```
~/dexproject      <- this repo (github.com/joanelizabethbino2024-dev/dex-vlm-grasping)
~/ur_gz_ws        <- github.com/mahimaapriyadharshinis/ur_gz_ws (sim world, robot/hand models, launch files)
```

They must be siblings under `$HOME` — `vlm_scripts/start_sim.sh` and `launch_all.sh` both resolve paths from `$HOME`, not from where `dexproject` itself lives.

## Prerequisites (not installed by anything in this repo)

- ROS 2 Humble
- Ignition/Gazebo (the version `ur_simulation_gz` targets)
- Ollama, with a vision model pulled (used by `vlm_overhead_node.py` — check `MODEL_NAME` in that file for which one)

## Build the sim workspace

```bash
cd ~/ur_gz_ws
colcon build
source install/setup.bash
```

## Python dependencies

```bash
pip install -r ~/dexproject/vlm_scripts/requirements.txt
```

`rclpy`, `cv_bridge`, `tf2_ros`, and the other ROS message packages come from the ROS 2 Humble install itself, not pip.

## Run

```bash
cd ~/dexproject/vlm_scripts
./launch_all.sh
```

`stop_all.sh` kills the sim and every pipeline node it started.

## Known rough edges

- `yolov8n.pt` (YOLO weights, ~6.5MB) lives in the `ur_gz_ws` repo root, not `dexproject`.
- Detection/grasp-geometry constants (e.g. `GRASP_POINT_IN_HAND_M` in `vision_ik_overhead_closed_loop.py`) were tuned against this specific sim's URDF and apple model. If either changes, those will need re-measuring, not just re-tuning by eye — see the comments at each constant for the method used to derive it.
