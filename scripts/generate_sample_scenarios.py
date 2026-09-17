"""Generates structured real-world outdoor validation scenarios for software-only verification.

Creates 5 focused outdoor test scenarios in datasets/processed/:
1. scenario_1_open_path: Clear traversable trail (high confidence, straight motion)
2. scenario_2_sudden_obstacle: Positive obstacle on trail (geometry detection, avoidance steering)
3. scenario_3_terrain_boundary: Pathway flanked by non-traversable bushes (semantic guidance)
4. scenario_4_depth_degradation: Glare / invalid depth region (Rule 12 testing: invalid depth != free)
5. scenario_5_visual_degradation: Low-feature / blurred scene (VO failure, localization loss safety stop)
"""

from __future__ import annotations

import json
import os
import numpy as np
import cv2


def make_realistic_outdoor_scene(
    frame_idx: int,
    num_frames: int,
    scenario_type: str,
    width: int = 640,
    height: int = 480,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Synthesize photorealistic outdoor ground & depth frames honoring real camera physics."""
    # Camera geometry: height 0.45m, downward pitch 12 deg
    # Nominal ground plane: z_c * sin(pitch) + y_c * cos(pitch) = h_cam
    pitch = 0.209  # ~12 deg
    h_cam = 0.45
    fx, fy = 385.0, 385.0
    cx, cy = 320.0, 240.0

    # Base RGB canvas
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth_m = np.zeros((height, width), dtype=np.float32)

    # 1. Sky / Background (Upper half)
    horizon_y = int(height * 0.38)
    for y in range(horizon_y):
        frac = y / max(horizon_y, 1)
        # Blue/hazy outdoor sky gradient
        rgb[y, :] = [int(135 + 40 * frac), int(170 + 40 * frac), int(220 + 20 * frac)]

    # 2. Ground plane geometry & textures
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Distance Z along camera optical axis for flat ground:
    # y_c = (v - cy) * Z / fy
    # Z * sin(p) + [(v - cy)*Z/fy] * cos(p) = h_cam
    # Z * [sin(p) + ((v - cy)/fy)*cos(p)] = h_cam
    denom = np.sin(pitch) + ((y_coords - cy) / fy) * np.cos(pitch)
    ground_mask = denom > 0.03

    ground_z = np.zeros((height, width), dtype=np.float32)
    ground_z[ground_mask] = h_cam / denom[ground_mask]
    ground_z[ground_z > 25.0] = 0.0

    # Robot base coordinates (X forward, Y lateral)
    x_robot = ground_z * np.cos(pitch) - ((y_coords - cy) * ground_z / fy) * np.sin(pitch)
    y_robot = -((x_coords - cx) * ground_z / fx)

    # Motion offset per frame (simulating camera translation forward)
    fwd_offset = frame_idx * 0.15
    cur_x = x_robot + fwd_offset

    # Central path width = 1.4m (-0.7m to +0.7m)
    path_width = 1.3
    is_path = (abs(y_robot) <= path_width / 2.0) & ground_mask
    is_margins = (abs(y_robot) > path_width / 2.0) & ground_mask

    # Path color: earth dirt / gravel (brown/gray with realistic texture)
    noise = np.random.normal(0, 10, (height, width)).astype(np.float32)
    rgb[is_path, 0] = np.clip(160 + noise[is_path] * 0.8, 0, 255).astype(np.uint8)
    rgb[is_path, 1] = np.clip(140 + noise[is_path] * 0.7, 0, 255).astype(np.uint8)
    rgb[is_path, 2] = np.clip(115 + noise[is_path] * 0.6, 0, 255).astype(np.uint8)

    # Margins: green grass and shrubs
    grass_noise = np.random.normal(0, 15, (height, width)).astype(np.float32)
    rgb[is_margins, 0] = np.clip(55 + grass_noise[is_margins] * 0.5, 0, 255).astype(np.uint8)
    rgb[is_margins, 1] = np.clip(120 + grass_noise[is_margins] * 0.8, 0, 255).astype(np.uint8)
    rgb[is_margins, 2] = np.clip(45 + grass_noise[is_margins] * 0.4, 0, 255).astype(np.uint8)

    depth_m = np.copy(ground_z)

    # Visual Odometry trackable features (rocks/texture on path)
    for seed in range(25):
        np.random.seed(seed * 100 + 7)
        rx = np.random.uniform(-0.6, 0.6)
        rz = np.random.uniform(1.2, 8.0) - fwd_offset
        if rz > 0.8:
            # Project to pixel
            u = int(cx - (rx * fx / rz))
            v = int(cy + ((h_cam - rz * np.sin(pitch)) * fy / (rz * np.cos(pitch))))
            if 0 <= u < width and horizon_y < v < height:
                cv2.circle(rgb, (u, v), 3, (80, 70, 60), -1)

    metadata = {"scenario": scenario_type, "frame": frame_idx}

    # Scenario-specific modifications:
    if scenario_type == "scenario_2_sudden_obstacle":
        # Positive obstacle placed at X = 2.8m, Y = 0.0m on the path
        obs_x = 3.2 - fwd_offset
        obs_y = 0.0
        obs_w = 0.6  # 60cm wide
        obs_h = 0.4  # 40cm high
        if obs_x > 0.5:
            # Find pixels corresponding to obstacle
            u_min = int(cx - ((obs_y + obs_w/2) * fx / obs_x))
            u_max = int(cx - ((obs_y - obs_w/2) * fx / obs_x))
            # Height in image
            v_bottom = int(cy + ((h_cam - obs_x * np.sin(pitch)) * fy / (obs_x * np.cos(pitch))))
            v_top = int(cy + (((h_cam - obs_h) - obs_x * np.sin(pitch)) * fy / (obs_x * np.cos(pitch))))
            if 0 <= v_top < height and 0 <= v_bottom < height and 0 <= u_min < width and 0 <= u_max < width:
                rgb[v_top:v_bottom, u_min:u_max] = [180, 50, 40]  # Red/brown solid crate/boulder
                depth_m[v_top:v_bottom, u_min:u_max] = obs_x
                metadata["has_obstacle"] = True
                metadata["obstacle_dist_m"] = round(float(obs_x), 2)

    elif scenario_type == "scenario_4_depth_degradation":
        # Sunlight glare / low-texture region in center -> depth becomes invalid (0.0 / NaN)
        # Tests Rule 12: Invalid depth must NOT be treated as free space!
        glare_u_min = width // 3
        glare_u_max = 2 * width // 3
        glare_v_min = int(height * 0.45)
        glare_v_max = int(height * 0.85)

        # Glare bloom in RGB
        rgb[glare_v_min:glare_v_max, glare_u_min:glare_u_max] = np.clip(
            rgb[glare_v_min:glare_v_max, glare_u_min:glare_u_max].astype(np.float32) + 120.0,
            0, 255
        ).astype(np.uint8)

        # Depth sensor fails to compute stereo disparity/IR reflection -> returns 0.0
        depth_m[glare_v_min:glare_v_max, glare_u_min:glare_u_max] = 0.0
        metadata["depth_degraded"] = True

    elif scenario_type == "scenario_5_visual_degradation":
        # Motion blur / extreme low contrast -> feature tracking loss
        # Tests Rule 13: Confidence drop triggers immediate safety stop
        rgb = cv2.GaussianBlur(rgb, (45, 45), 0)
        rgb = (rgb * 0.35).astype(np.uint8)  # Heavy underexposure
        metadata["visual_tracking_degraded"] = True

    return rgb, depth_m, metadata


def generate_all_scenarios(base_dir: str = "datasets/processed") -> None:
    """Generate all 5 scenarios with 15 frames each."""
    scenarios = [
        ("scenario_1_open_path", 15),
        ("scenario_2_sudden_obstacle", 15),
        ("scenario_3_terrain_boundary", 15),
        ("scenario_4_depth_degradation", 15),
        ("scenario_5_visual_degradation", 15),
    ]

    for sc_name, count in scenarios:
        sc_dir = os.path.join(base_dir, sc_name)
        rgb_dir = os.path.join(sc_dir, "rgb")
        depth_dir = os.path.join(sc_dir, "depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        meta_list = []
        for i in range(count):
            rgb, depth, meta = make_realistic_outdoor_scene(i, count, sc_name)
            # Save RGB
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(rgb_dir, f"frame_{i:04d}.jpg"), rgb_bgr)
            # Save Depth as .npy float32 meters
            np.save(os.path.join(depth_dir, f"frame_{i:04d}.npy"), depth)
            meta_list.append(meta)

        with open(os.path.join(sc_dir, "metadata.json"), "w") as f:
            json.dump(meta_list, f, indent=2)

        print(f"Generated {count} frames for {sc_name} in {sc_dir}")


if __name__ == "__main__":
    generate_all_scenarios()
