# Publication Readiness — Gaps & Closure Plan

Scope: the adaptive squeeze-and-relax grasp controller (`adaptive_grasp_controller.py`),
the strongest publication candidate in this project (see prior discussion). This
tracks the four blockers to a workshop/short-paper-level submission and what's
needed to close each one. Nothing here claims the gaps are closed — items 1 and 2
now have runnable scaffolding; items 3 and 4 are still open work for you to do.

## 1. No quantitative evaluation

**The ask a reviewer makes:** "How much better is this than closing to a fixed
force?"

**What's now in place** (`adaptive_grasp_controller.py`): a `FIXED_FORCE_BASELINE`
mode and a trial-log CSV.

- Run the adaptive controller normally, then re-run with:
  ```
  FIXED_FORCE_BASELINE=1 FIXED_FORCE_CLOSURE=0.6 ros2 run <pkg> adaptive_grasp_controller
  ```
  The baseline still *detects* deformation (so you can see if fixed force damages
  fragile objects) but never reacts to it — it just closes to `FIXED_FORCE_CLOSURE`
  and holds. That's the honest "naive" comparison point.
- Every trial (adaptive or baseline) appends one row to `~/dexproject/vlm_scripts/trial_log.csv`:
  `timestamp, object_name, mode, used_memory, time_to_lock_sec, deformation_events, final_closure_mean, final_effort_mean`.

**What you still have to do:**
- Pick 5–10 objects spanning a fragility range (e.g. apple, egg, rigid block, soft
  fruit, thin-walled cup) and run N trials each (10+ per object per condition is a
  reasonable minimum for a workshop paper).
- `deformation_events` and `final_effort_mean` are objective, logged automatically.
  **Success/no-damage is not** — the controller has no way to know if it crushed the
  object. You need an external ground-truth check per trial (a human rating a 0–3
  damage scale from the gripper-camera frame, or a simple vision check in sim for a
  known deformable mesh) and add that as a column you fill in per trial (or a
  post-hoc script keyed on `timestamp`).
- Build the actual success-rate / mean-force / deformation-incident table from the
  CSV once trials are run — that table is the paper's central result.

## 2. No ablation

**The ask:** "Does the derivative trigger matter over a plain ceiling? Does the
MNDF memory actually help?"

**What's now in place:**
- `DEFORMATION_TRIGGER_MODE=absolute_only` drops the effort-derivative trigger and
  keeps only `EFFORT_ABS_CEILING`, so you can run the same object set through both
  modes and compare `deformation_events` / final grasp quality.
- `MNDF_MEMORY_ENABLED=0` disables memory lookup/save entirely, forcing every trial
  through the full squeeze→relax→fine-approach search, so you can compare
  `time_to_lock_sec` and outcome with vs. without memory on repeat presentations of
  the same object.

**What you still have to do:** run the 2×2 (trigger mode × memory on/off) grid
across your object set and report it as an ablation table — this is what turns
"we implemented X" into "we measured that X matters."

## 3. Simulation-only, and the sim is under active calibration

The codebase is honest about this in its own comments — that's good, but it means
the current sim is not yet a platform you can run "real" trials on and defend to
reviewers. A non-exhaustive list of open calibration items, so you can track them
to closure or explicitly scope them as future work:

| File | Line(s) | Issue |
|---|---|---|
| `vision_ik_overhead_closed_loop.py` | 44 | `HAND_JOINT_LIMITS_RAD` is a placeholder `(0, pi/2)` for every hand joint — real per-joint limits (already correctly sourced in `adaptive_grasp_controller.py` and `manual_target_node.py` from the URDF) haven't been ported here yet. |
| `vlm_overhead_node.py` | 24 | `IMAGE_TOPIC` is flagged as an unconfirmed guess. |
| `arm_approach_node.py` | 25, 43 | Descend pose is a repeated placeholder, explicitly marked `TODO: replace`. |
| `manual_target_node.py` | 103–104 | The contact-detection signal is described as "a proxy for contact, not a true force sensor — consider it a placeholder for real [F/T] hardware." This is close to the paper's core claim (sensorless contact/deformation sensing), so this one especially needs to be nailed down and stated precisely, not left as a comment. |
| `query_apple_vlm.py` / `vision_ik_overhead_closed_loop.py` | `CAMERA_WORLD_*`, `HORIZONTAL_FOV_RAD`, `APPLE_WORLD_Z` | Hand-tuned scene constants duplicated across files rather than read from the URDF/SDF or camera intrinsics — fine for a demo, but a reviewer will ask how these generalize past this one scene. |

**Recommendation:** before running the trials in §1–2, do a calibration pass and
either (a) resolve each item above with a stated verification method (matches
`manual_target_node.py`'s pattern of "confirmed via live testing on <date>"), or
(b) if resolving all of them isn't feasible, scope the paper's claims to exactly
the validated subset (e.g. the adaptive grasp controller's effort-derivative logic
alone, tested on real joint-effort data even if the perception/positioning stack
around it stays a known-limited prototype) and say so explicitly in a Limitations
section. Reviewers penalize unstated gaps far more than stated ones. If real
hardware is available at all, even a handful of real-robot trials will carry much
more weight than additional sim trials.

## 4. No related-work positioning

The differentiator — "no dedicated tactile/F-T hardware, purely off
`/joint_states` effort" — is real, but it has to be argued against specific prior
work, not asserted. Grounded starting points (found via search, verify relevance
before citing):

**Sensorless / proprioceptive contact & slip sensing** (directly competing claim —
this is the related work to differentiate hardest against):
- *Current as Touch: Proprioceptive Contact Feedback for Compliant Dexterous
  Manipulation* — https://arxiv.org/pdf/2607.03529 — learns contact feedback from
  motor current/joint states without external tactile/F-T sensing; converts it into
  position references via a standard PD controller. Closest prior work to your
  "no dedicated tactile hardware" claim — you need to state precisely what MNDF
  memory adds over this (per-object persistence and a simple derivative-trigger
  rule vs. a learned model).

**Tactile-sensor-based slip/force approaches** (useful as the "what people usually
do instead" contrast):
- *TacDexGrasp: Compliant and Robust Dexterous Grasping with Tactile Feedback* —
  https://arxiv.org/pdf/2603.07040
- *Reactive Slip Control in Multifingered Grasping: Hybrid Tactile Sensing and
  Internal-Force Optimization* — https://arxiv.org/html/2602.16127v2
- *FORTE: Tactile Force and Slip Sensing on Compliant Fingers for Delicate
  Manipulation* — https://arxiv.org/pdf/2506.18960 (92% success on
  slippery/fragile/deformable objects — a concrete number to compare your
  success-rate table against once you have one)
- *Calibration-free per-finger force-feedback slip control ... tri-axial tactile
  sensors* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12926654/

**Experience-based / per-object grasp memory** (directly competing claim for the
MNDF-memory half of the contribution):
- *DGCM-Net: Dense Geometrical Correspondence Matching Network for Incremental
  Experience-based Robotic Grasping* — https://arxiv.org/pdf/2001.05279 — stores
  every successful grasp and retrieves by visual similarity. Contrast: your memory
  is keyed by VLM-predicted object *name*, not visual/geometric similarity — cheaper
  but brittle to misclassification; say so.
- *A system of robotic grasping with experience acquisition* — Sci China Inf Sci —
  https://link.springer.com/article/10.1007/s11432-014-5208-3

**Fragile/deformable object grasping specifically** (motivates why this matters,
good for the intro, not the core differentiation):
- *Towards Damage-Less Robotic Fragile Fruit Grasping: A Systematic Review* — J.
  Field Robotics 2025 — https://onlinelibrary.wiley.com/doi/10.1002/rob.70021 —
  good survey to cite for the problem statement and to check whether "apple
  grasping in sim" has already been heavily covered (it likely has — differentiate
  on the sensorless/memory angle, not the object choice).
- *Quantitative Hardness Assessment with Vision-based Tactile Sensing for Fruit
  Classification and Grasping* — https://arxiv.org/pdf/2505.05725

**Action item:** read the "Current as Touch" paper (arxiv 2607.03529) first and
closely — it's the single closest piece of prior work to this project's central
claim, and the paper's contribution statement should be written as a delta against
it, not written first and reconciled later.
