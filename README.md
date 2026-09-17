# SIH 26126 — TerrainSight UGV

Offline-first, vision-based autonomous navigation for an outdoor UGV in GPS-denied environments.

## Baseline pipeline

`RGB-D → traversability segmentation → depth geometry → fusion → visual SLAM → costmap → Nav2 → safety gate → cmd_vel`

The system is deliberately modular: each stage exposes a ROS 2 interface, timestamps and health/confidence information. The safety gate is the only route from planning to motor commands.

## Foundation decisions

- **Project name:** TerrainSight UGV
- **Middleware:** ROS 2 Jazzy Jalisco
- **Simulator:** Gazebo Harmonic
- **Reference sensor for simulation:** Intel RealSense D435i RGB-D model
- **Reference vehicle model:** differential-drive UGV; provisional 0.60 m × 0.45 m chassis footprint

These are simulation and software-baseline decisions, not claims about physically available hardware. See [hardware inventory](docs/architecture/hardware-inventory.md) before a real-robot run.

## Repository workflow

- `main` is protected/release-ready integration.
- Work happens in short-lived branches named `feature/<area>-<purpose>`, `fix/<area>-<issue>`, or `docs/<topic>`.
- A pull request must include a focused test or verification note; safety/configuration changes need explicit peer review.

## Quick start

The reproducible development environment is defined in `docker-compose.yml` and `docker/ros/Dockerfile`. On Ubuntu with Docker installed:

```bash
docker compose build ros
docker compose run --rm ros bash
colcon build --symlink-install
```

Host hardware is intentionally not baked into the image. Record GPU, CUDA, camera SDK, and actual robot parameters in `docs/architecture/hardware-inventory.md` before hardware integration.

## Layout

- `src/` — ROS 2 packages, one module per responsibility
- `config/` — versioned module configuration
- `simulation/` — Gazebo assets and launch configuration
- `tests/` — unit and integration checks
- `docs/` — architecture, experiments, and demo evidence

## Current phase

Phase 0 is scaffolded. The physical-hardware verification tasks remain explicitly blocked until the team records the available equipment.
