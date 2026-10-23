# Changelog

## Cluster release

The widget cockpit is replaced by a GPU-rendered instrument cluster built from the ADAS Cluster UI/UX spec. The previous cockpit is still available with `--ui classic`.

### Added
- **Cluster UI** (`src/ui/qml/Cluster.qml`, `src/ui/cluster_app.py`): violet-bezel cluster with speed, posted limit, telltales, function status, alert banner, front-camera tile and full-camera view, perception status, trip / FPS / latency, SoC temperature. Authored at 1920 × 976 and scaled to any window.
- **Threat model** (`src/adas/threat.py`): green / yellow / red per tracked object from time headway, TTC, closing speed, adjacent-lane proximity, cut-in and lane-change conflict. Includes hysteresis, banner priority and audio cues, with unit tests in `tests/`.
- **3D scene upgrades** (`BevScene.qml`, `bev_quick3d.py`):
  - neutral-grey traffic tinted by threat, with pulsing halos;
  - a gap band to the lead vehicle, with chevrons;
  - lane lines that turn yellow or red with LDW, and an ego that drifts toward the line;
  - a distance / time-gap / closing-speed tag;
  - a soft-edged scene mask.
- **Earcons**: synthesized collision, lane-departure (panned to the departing side), caution and over-speed tones through QtMultimedia.
- **Inputs**: USB cameras (`--camera 0`), CSI cameras through Argus (`--camera csi`) and any GStreamer pipeline. A camera that fails to open is retried.
- **Speed sources** (`src/utils/speed_source.py`): JSON log, fixed value or OBD-II.
- **Simulation** (`--sim`): a 60 s scripted scenario through every alert, using the real LDW / FCW logic. `--screenshots` renders each alert state to PNG.
- **Deployment**: `scripts/run_cluster.sh` launcher, `deploy/adas-cluster.service` for kiosk autostart, and new `setup.sh --system-opencv` option for CSI.

### Changed
- `InferenceWorker` takes a speed source and reads speed even when engines are missing. It now reports model availability and road status to the UI, and resets the frame index when a video loops so speed logs stay aligned.
- `BevQuick3DWidget` is split into a shared `_BevDriver` plus `BevQuick3DWidget` (classic) and `BevSceneDriver` (cluster).
- `run_pyside6_app.py` defaults to the cluster and resolves relative paths from where it is launched.
- `setup.sh` now also installs `libxcb-cursor0`, `libxkbcommon-x11-0`, `libpulse0` and `v4l-utils`.
