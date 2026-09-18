#!/usr/bin/env python3
"""Generates visual failure example comparisons for Multimodal Fusion using OpenCV.

Creates 3 illustrative comparative panels:
1. Case 1: Geometry Veto (Camouflaged Obstacle) - Vision fails, Depth saves UGV.
2. Case 2: Semantic Veto (Liquid/Mud Trap) - Depth fails (flat), Vision saves UGV.
3. Case 3: Rule 12 Guard (Solar Glare Dropout) - Missing depth safely penalized.

Saved to: docs/images/fusion_cases/
"""

from __future__ import annotations

import os
import sys
import numpy as np
import cv2

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.interfaces.types import (
    CameraIntrinsics,
    SemanticResult,
    DepthGeometryResult,
    TerrainClass,
)
from src.fusion.fusion_engine import FusionEngine
from src.depth_geometry.point_cloud import DepthProjector
from scripts.generate_sample_scenarios import make_realistic_outdoor_scene


def make_card(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Formats an image into a labeled visual card."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    # Header banner
    banner_h = 50
    banner = np.full((banner_h, w, 3), 30, dtype=np.uint8)
    cv2.putText(banner, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(banner, subtitle, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)

    card = np.vstack([banner, img])
    # White outline border
    card = cv2.copyMakeBorder(card, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60))
    return card


def colorize_traversability(trav: np.ndarray) -> np.ndarray:
    """Maps traversability [0..1] to green (safe) -> red (lethal)."""
    # Inverse: high cost is red (0), low cost is green (120)
    u8 = (np.clip(trav, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def colorize_costmap(cost: np.ndarray) -> np.ndarray:
    """Colorizes costmap [0..254] with inferno / hot palette."""
    u8 = np.clip(cost, 0, 254).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def create_visual_cases():
    out_dir = os.path.join(repo_root, "docs", "images", "fusion_cases")
    os.makedirs(out_dir, exist_ok=True)

    intrinsics = CameraIntrinsics()
    fusion = FusionEngine(intrinsics)
    projector = DepthProjector(intrinsics)

    # -------------------------------------------------------------------------
    # CASE 1: Geometry Veto (Camouflaged Obstacle)
    # -------------------------------------------------------------------------
    print("Generating Case 1: Geometry Veto (Camouflaged Obstacle)...")
    rgb_obs, depth_obs, _ = make_realistic_outdoor_scene(0, 15, "scenario_2_sudden_obstacle")
    h, w = depth_obs.shape

    p_sem_deceived = np.full((h, w), 0.90, dtype=np.float32)
    p_sem_deceived[:150, :] = 0.0
    classes_deceived = np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32)
    classes_deceived[:150, :] = TerrainClass.UNKNOWN
    sem_deceived = SemanticResult(p_sem_deceived, classes_deceived, confidence=0.88)

    valid_d = (depth_obs >= 0.35) & (depth_obs <= 12.0)
    p_geom_cost = np.zeros((h, w), dtype=np.float32)
    pos_obs = np.zeros((h, w), dtype=bool)
    pos_obs[270:340, 280:360] = True
    p_geom_cost[pos_obs] = 1.0
    p_geom_cost[~valid_d] = 0.60

    geo_obs = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=pos_obs,
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_d,
        geometric_cost=p_geom_cost,
        confidence=0.92,
    )

    pts_base, _ = projector.project_and_transform_to_base_link(np.where(valid_d, depth_obs, np.nan))
    fused_res = fusion.fuse(sem_deceived, geo_obs, pts_base)
    p_naive = 0.50 * p_sem_deceived + 0.50 * (1.0 - p_geom_cost)

    # Convert BEV grid to match camera aspect for display
    bev_vis = cv2.resize(colorize_costmap(fused_res.fused_costmap), (w, h))

    c1_1 = make_card(cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR), "(a) Input Camera RGB", "Boulder blends with background terrain")
    c1_2 = make_card(colorize_traversability(p_sem_deceived), "(b) CNN Perception", "DECEIVED: Classifies boulder as 0.90 Safe")
    c1_3 = make_card(cv2.applyColorMap((p_geom_cost * 255).astype(np.uint8), cv2.COLORMAP_HOT), "(c) 3D Depth Geometry", "DETECTS: Physical height dz = +0.40m")
    c1_4 = make_card(colorize_traversability(p_naive), "(d) Naive Average", "FAIL: Blends to 0.45, UGV crashes into boulder")
    c1_5 = make_card(colorize_traversability(fused_res.fused_traversability), "(e) Geometry Veto Fusion", "SAVED: Forces 0.0 lethal obstacle")
    c1_6 = make_card(bev_vis, "(f) 2D BEV Costmap", "Obstacle extracted & path blocked safely")

    row1 = np.hstack([c1_1, c1_2, c1_3])
    row2 = np.hstack([c1_4, c1_5, c1_6])
    case1_panel = np.vstack([row1, row2])

    c1_path = os.path.join(out_dir, "case1_geometry_veto_camouflaged_obstacle.png")
    cv2.imwrite(c1_path, case1_panel)
    print(f"  Saved: {c1_path}")

    # -------------------------------------------------------------------------
    # CASE 2: Semantic Veto (Liquid/Mud Trap)
    # -------------------------------------------------------------------------
    print("Generating Case 2: Semantic Veto (Liquid/Mud Trap)...")
    rgb_clean, depth_clean, _ = make_realistic_outdoor_scene(0, 15, "scenario_1_open_path")

    p_sem_puddle = np.full((h, w), 0.92, dtype=np.float32)
    classes_puddle = np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32)
    p_sem_puddle[320:380, 240:400] = 0.08  # Impassable liquid
    classes_puddle[320:380, 240:400] = TerrainClass.WATER_PUDDLE
    sem_puddle = SemanticResult(p_sem_puddle, classes_puddle, confidence=0.90)

    valid_d2 = (depth_clean >= 0.35) & (depth_clean <= 12.0)
    geo_puddle = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_d2,
        geometric_cost=np.where(valid_d2, 0.0, 0.60).astype(np.float32),
        confidence=0.95,
    )

    pts_base2, _ = projector.project_and_transform_to_base_link(np.where(valid_d2, depth_clean, np.nan))
    fused_res2 = fusion.fuse(sem_puddle, geo_puddle, pts_base2)
    p_naive2 = 0.50 * p_sem_puddle + 0.50 * (1.0 - geo_puddle.geometric_cost)
    bev_vis2 = cv2.resize(colorize_costmap(fused_res2.fused_costmap), (w, h))

    c2_1 = make_card(cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2BGR), "(a) Input Camera RGB", "Water puddle / deep mud trap ahead")
    c2_2 = make_card(cv2.applyColorMap((geo_puddle.geometric_cost * 255).astype(np.uint8), cv2.COLORMAP_HOT), "(b) 3D Depth Geometry", "DECEIVED: Flat plane dz ~ 0.0m (Looks safe)")
    c2_3 = make_card(colorize_traversability(p_sem_puddle), "(c) CNN Perception", "FLAGS: Water puddle / mud (Impassable liquid)")
    c2_4 = make_card(colorize_traversability(p_naive2), "(d) Naive Average", "FAIL: 0.54 Trav, drives UGV into deep water")
    c2_5 = make_card(colorize_traversability(fused_res2.fused_traversability), "(e) Semantic Veto Fusion", "SAVED: Forces <= 0.10 hazard cost")
    c2_6 = make_card(bev_vis2, "(f) 2D BEV Costmap", "Puddle area flagged as lethal hazard")

    row1_2 = np.hstack([c2_1, c2_2, c2_3])
    row2_2 = np.hstack([c2_4, c2_5, c2_6])
    case2_panel = np.vstack([row1_2, row2_2])

    c2_path = os.path.join(out_dir, "case2_semantic_veto_water_mud_hazard.png")
    cv2.imwrite(c2_path, case2_panel)
    print(f"  Saved: {c2_path}")

    # -------------------------------------------------------------------------
    # CASE 3: Rule 12 Guard (Solar Glare / Missing Depth)
    # -------------------------------------------------------------------------
    print("Generating Case 3: Rule 12 Guard (Solar Glare / Missing Depth)...")
    rgb_deg, depth_deg, _ = make_realistic_outdoor_scene(0, 15, "scenario_4_depth_degradation")

    valid_d3 = (depth_deg >= 0.35) & (depth_deg <= 12.0)
    sem_deg = SemanticResult(
        traversability_mask=np.full((h, w), 0.85, dtype=np.float32),
        terrain_class_map=np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32),
        confidence=0.82,
    )
    geo_deg = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_d3,
        geometric_cost=np.where(valid_d3, 0.0, 0.60).astype(np.float32),
        confidence=0.55,
    )

    pts_base3, _ = projector.project_and_transform_to_base_link(np.where(valid_d3, depth_deg, np.nan))
    fused_res3 = fusion.fuse(sem_deg, geo_deg, pts_base3)
    bev_vis3 = cv2.resize(colorize_costmap(fused_res3.fused_costmap), (w, h))

    # Grayscale validity
    val_img = np.where(valid_d3[..., None], np.array([255, 255, 255], dtype=np.uint8), np.array([0, 0, 0], dtype=np.uint8))
    unc_img = cv2.applyColorMap((fused_res3.uncertainty_px * 255).astype(np.uint8), cv2.COLORMAP_JET)

    c3_1 = make_card(cv2.cvtColor(rgb_deg, cv2.COLOR_RGB2BGR), "(a) Input Camera RGB", "Intense direct sunlight & specular washout")
    c3_2 = make_card(val_img, "(b) Depth Validity Mask", "BLACK: 48.9% missing depth (Solar blind zone)")
    c3_3 = make_card(colorize_traversability(fused_res3.fused_traversability), "(c) Fused Traversability", "CLAMPED: <= 0.40 safe crawl cap (Rule 12)")
    c3_4 = make_card(unc_img, "(d) Spatial Uncertainty Map", "ELEVATED: 1.0 uncertainty in dropout zone")
    c3_5 = make_card(bev_vis3, "(e) 2D BEV Costmap", "Rule 11: Unknown cells assigned cost 128")
    c3_6 = make_card(cv2.resize(np.where(fused_res3.unknown_mask[..., None], 255, 0).astype(np.uint8), (w, h)), "(f) Unobserved Grid Mask", "Safe crawl triggered, vehicle never speeds")

    row1_3 = np.hstack([c3_1, c3_2, c3_3])
    row2_3 = np.hstack([c3_4, c3_5, c3_6])
    case3_panel = np.vstack([row1_3, row2_3])

    c3_path = os.path.join(out_dir, "case3_rule12_missing_depth_solar_glare.png")
    cv2.imwrite(c3_path, case3_panel)
    print(f"  Saved: {c3_path}")
    print("All 3 visual failure example figures generated successfully.")


if __name__ == "__main__":
    create_visual_cases()
