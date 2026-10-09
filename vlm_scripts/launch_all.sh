#!/bin/bash
# ---------------------------------------------------------------------------
# launch_all.sh
# ---------------------------------------------------------------------------
# Single entry point that replaces running these separately in several terminals:
#   1. start_sim.sh              (Gazebo sim + arm controllers + bridge)
#   2. yolo_depth_detector.py    (real-depth apple 3D localization -- primary
#                                  detection source, see ENABLE_DEPTH_DETECTOR_NODE)
#   3. vlm_overhead_node.py      (overhead camera VLM node -- alignment-check
#                                  duty when the depth detector is enabled)
#   4. vision_ik_overhead_closed_loop.py   (closed-loop IK node)
#   5. (optional) vlm_fragility_node.py -- gripper-camera VLM node
#   6. (implicitly) whatever terminal you had rqt_image_view or ollama serve in
#
# What it does:
#   - Kills any leftover processes from a previous run (same cleanup start_sim.sh
#     already does, run once up front here).
#   - Makes sure Ollama is actually running before anything else starts.
#   - Starts the simulation, then POLLS for /overhead_camera to actually have a
#     publisher before starting the nodes that depend on it -- this is exactly
#     the "Publisher count: 0" problem we chased manually earlier, now handled
#     automatically instead of you having to check by hand.
#   - Runs every node in the background, each logging to its own file under
#     ~/dexproject/vlm_scripts/logs/, and tails all of them combined to this
#     terminal so you still see everything live.
#   - On Ctrl+C (or any exit), kills every process it started, including
#     everything spawned underneath ros2 launch (sim, controllers, bridge).
#
# EDIT THESE PATHS if your files live somewhere different:
# ---------------------------------------------------------------------------
SCRIPTS_DIR="$HOME/dexproject/vlm_scripts"
START_SIM_SCRIPT="$SCRIPTS_DIR/start_sim.sh"
VLM_OVERHEAD_NODE="$SCRIPTS_DIR/vlm_overhead_node.py"
DEPTH_DETECTOR_NODE="$SCRIPTS_DIR/yolo_depth_detector.py"
IK_NODE="$SCRIPTS_DIR/vision_ik_overhead_closed_loop.py"
ADAPTIVE_GRASP_NODE="$SCRIPTS_DIR/adaptive_grasp_controller.py"
FINGER_ALIGN_NODE="$SCRIPTS_DIR/finger_alignment_node.py"

# adaptive_grasp_controller.py (force-feedback squeeze) is DISABLED for now
# (2026-09-16) -- vision_ik_overhead_closed_loop.py no longer hands off to
# it. finger_alignment_node.py is the current pipeline's hand-positioning
# stage instead: straight fingers (no curl) positioned around the object by
# size, once the arm is confirmed to have reached the target. Only turn this
# back on once finger_alignment_node.py's straight-positioning is confirmed
# working AND vision_ik_overhead_closed_loop.py's execute_grasp() handoff is
# re-wired to call it.
ENABLE_ADAPTIVE_GRASP_CONTROLLER=0

# finger_alignment_node.py: straight-finger (no curl) positioning around the
# object, scaled by its measured size, triggered once the arm is confirmed
# to have reached the target. See vision_ik_overhead_closed_loop.py's
# _on_settle_timer_fired.
ENABLE_FINGER_ALIGN_NODE=1

# yolo_depth_detector.py finds the apple's real 3D position from the overhead
# depth camera (YOLO detection + actual measured depth + gripper-position
# exclusion via forward kinematics) instead of assuming a fixed apple height.
# It becomes the initial-detection source; vlm_overhead_node.py is kept
# running alongside it purely for its on-demand alignment-check duty (it's
# started with VLM_OVERHEAD_ALIGNMENT_ONLY=1 below so the two don't race each
# other publishing to the same detection topic). Set to 0 to go back to the
# old VLM-only pixel/assumed-height pipeline.
ENABLE_DEPTH_DETECTOR_NODE=1

# Set to 1 if you also want the gripper-camera VLM node started automatically.
# Update GRIPPER_VLM_NODE to the real filename/path if you enable this.
ENABLE_GRIPPER_VLM_NODE=0
GRIPPER_VLM_NODE="$SCRIPTS_DIR/vlm_fragility_node.py"

OVERHEAD_CAMERA_TOPIC="/overhead_camera"
SIM_READY_TIMEOUT_SEC=90
OLLAMA_READY_TIMEOUT_SEC=30
# ---------------------------------------------------------------------------

LOG_DIR="$SCRIPTS_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PIDS=()

cleanup() {
    echo ""
    echo "[launch_all] Shutting down..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
        fi
    done
    pkill -9 -f "ign gazebo" 2>/dev/null
    pkill -9 -f "gz sim" 2>/dev/null
    pkill -9 -f "parameter_bridge" 2>/dev/null
    pkill -9 -f "robot_state_publisher" 2>/dev/null
    pkill -9 -f "controller_manager" 2>/dev/null
    pkill -9 -f "vlm_overhead_node.py" 2>/dev/null
    pkill -9 -f "yolo_depth_detector.py" 2>/dev/null
    pkill -9 -f "vision_ik_overhead_closed_loop.py" 2>/dev/null
    pkill -9 -f "vlm_fragility_node.py" 2>/dev/null
    pkill -9 -f "adaptive_grasp_controller.py" 2>/dev/null
    pkill -9 -f "finger_alignment_node.py" 2>/dev/null
    echo "[launch_all] Done."
}
trap cleanup EXIT INT TERM

fail() {
    echo "[launch_all] ERROR: $1"
    exit 1
}

echo "[launch_all] Cleaning up any leftover processes from a previous run..."
pkill -9 -f "ign gazebo" 2>/dev/null
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "parameter_bridge" 2>/dev/null
pkill -9 -f "robot_state_publisher" 2>/dev/null
pkill -9 -f "controller_manager" 2>/dev/null
pkill -9 -f "rviz2" 2>/dev/null
pkill -9 -f "vlm_overhead_node.py" 2>/dev/null
pkill -9 -f "yolo_depth_detector.py" 2>/dev/null
pkill -9 -f "vision_ik_overhead_closed_loop.py" 2>/dev/null
pkill -9 -f "vlm_fragility_node.py" 2>/dev/null
pkill -9 -f "adaptive_grasp_controller.py" 2>/dev/null
pkill -9 -f "finger_alignment_node.py" 2>/dev/null
sleep 2
sudo rm -f /dev/shm/fastrtps_* 2>/dev/null
sudo rm -f /dev/shm/sem.fastrtps_* 2>/dev/null

if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[launch_all] Ollama not responding, starting it..."
    nohup ollama serve > "$LOG_DIR/ollama_${TIMESTAMP}.log" 2>&1 &
    OLLAMA_PID=$!
    PIDS+=("$OLLAMA_PID")

    waited=0
    until curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$OLLAMA_READY_TIMEOUT_SEC" ]; then
            fail "Ollama did not become ready within ${OLLAMA_READY_TIMEOUT_SEC}s."
        fi
    done
    echo "[launch_all] Ollama is up."
else
    echo "[launch_all] Ollama already running."
fi

[ -f "$START_SIM_SCRIPT" ] || fail "start_sim.sh not found at $START_SIM_SCRIPT"

echo "[launch_all] Starting simulation ($START_SIM_SCRIPT)..."
setsid bash "$START_SIM_SCRIPT" > "$LOG_DIR/sim_${TIMESTAMP}.log" 2>&1 &
SIM_PID=$!
PIDS+=("$SIM_PID")

source /opt/ros/humble/setup.bash 2>/dev/null
source "$HOME/ur_gz_ws/install/setup.bash" 2>/dev/null

echo "[launch_all] Waiting for $OVERHEAD_CAMERA_TOPIC to have a publisher (up to ${SIM_READY_TIMEOUT_SEC}s)..."
waited=0
until ros2 topic info "$OVERHEAD_CAMERA_TOPIC" --verbose 2>/dev/null | grep -q "Publisher count: [1-9]"; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$SIM_READY_TIMEOUT_SEC" ]; then
        fail "Sim did not start publishing $OVERHEAD_CAMERA_TOPIC within ${SIM_READY_TIMEOUT_SEC}s. Check $LOG_DIR/sim_${TIMESTAMP}.log for errors."
    fi
done
echo "[launch_all] Simulation ready ($OVERHEAD_CAMERA_TOPIC is publishing)."

[ -f "$VLM_OVERHEAD_NODE" ] || fail "vlm_overhead_node.py not found at $VLM_OVERHEAD_NODE"

if [ "$ENABLE_DEPTH_DETECTOR_NODE" = "1" ]; then
    [ -f "$DEPTH_DETECTOR_NODE" ] || fail "yolo_depth_detector.py not found at $DEPTH_DETECTOR_NODE"
    echo "[launch_all] Starting yolo_depth_detector.py (real-depth apple localization)..."
    setsid python3 "$DEPTH_DETECTOR_NODE" > "$LOG_DIR/depth_detector_${TIMESTAMP}.log" 2>&1 &
    PIDS+=("$!")

    echo "[launch_all] Starting vlm_overhead_node.py (alignment-check duty only)..."
    setsid env VLM_OVERHEAD_ALIGNMENT_ONLY=1 python3 "$VLM_OVERHEAD_NODE" > "$LOG_DIR/vlm_overhead_${TIMESTAMP}.log" 2>&1 &
    PIDS+=("$!")
else
    echo "[launch_all] Starting vlm_overhead_node.py..."
    setsid python3 "$VLM_OVERHEAD_NODE" > "$LOG_DIR/vlm_overhead_${TIMESTAMP}.log" 2>&1 &
    PIDS+=("$!")
fi

if [ "$ENABLE_GRIPPER_VLM_NODE" = "1" ]; then
    [ -f "$GRIPPER_VLM_NODE" ] || fail "Gripper VLM node not found at $GRIPPER_VLM_NODE"
    echo "[launch_all] Starting gripper VLM node..."
    setsid python3 "$GRIPPER_VLM_NODE" > "$LOG_DIR/vlm_gripper_${TIMESTAMP}.log" 2>&1 &
    PIDS+=("$!")
fi

[ -f "$IK_NODE" ] || fail "vision_ik_overhead_closed_loop.py not found at $IK_NODE"

echo "[launch_all] Starting vision_ik_overhead_closed_loop.py..."
setsid python3 "$IK_NODE" > "$LOG_DIR/ik_node_${TIMESTAMP}.log" 2>&1 &
PIDS+=("$!")

if [ "$ENABLE_ADAPTIVE_GRASP_CONTROLLER" = "1" ]; then
    [ -f "$ADAPTIVE_GRASP_NODE" ] || fail "adaptive_grasp_controller.py not found at $ADAPTIVE_GRASP_NODE"
    echo "[launch_all] Starting adaptive_grasp_controller.py (force-adaptive finger close)..."
    setsid python3 "$ADAPTIVE_GRASP_NODE" > "$LOG_DIR/adaptive_grasp_${TIMESTAMP}.log" 2>&1 &
    PIDS+=("$!")
fi

if [ "$ENABLE_FINGER_ALIGN_NODE" = "1" ]; then
    [ -f "$FINGER_ALIGN_NODE" ] || fail "finger_alignment_node.py not found at $FINGER_ALIGN_NODE"
    echo "[launch_all] Starting finger_alignment_node.py (straight-finger positioning)..."
    setsid python3 "$FINGER_ALIGN_NODE" > "$LOG_DIR/finger_align_${TIMESTAMP}.log" 2>&1 &
    PIDS+=("$!")
fi

echo ""
echo "[launch_all] Everything is running. Logs are in: $LOG_DIR"
echo "[launch_all] Streaming combined output below. Press Ctrl+C to stop everything."
echo "-----------------------------------------------------------------------------"

TAIL_LOGS=(
    "$LOG_DIR/sim_${TIMESTAMP}.log"
    "$LOG_DIR/vlm_overhead_${TIMESTAMP}.log"
    "$LOG_DIR/ik_node_${TIMESTAMP}.log"
)
if [ "$ENABLE_DEPTH_DETECTOR_NODE" = "1" ]; then
    TAIL_LOGS+=("$LOG_DIR/depth_detector_${TIMESTAMP}.log")
fi
if [ "$ENABLE_ADAPTIVE_GRASP_CONTROLLER" = "1" ]; then
    TAIL_LOGS+=("$LOG_DIR/adaptive_grasp_${TIMESTAMP}.log")
fi
if [ "$ENABLE_FINGER_ALIGN_NODE" = "1" ]; then
    TAIL_LOGS+=("$LOG_DIR/finger_align_${TIMESTAMP}.log")
fi

tail -n +1 -F "${TAIL_LOGS[@]}" 2>/dev/null &
TAIL_PID=$!
PIDS+=("$TAIL_PID")

wait "$SIM_PID"
