# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import csv
import os


# ========================= 可配置参数 =========================
input_path = r"D:\ah\sy_dy\stage1_triclass_rgb_fast - Cloud.pcd"
rock_output_path = r"D:\ah\sy_dy\JG\stage_rock.pcd"
vegetation_output_path = r"D:\ah\sy_dy\JG\stage_vegetatioon.pcd"
metrics_csv_path = r"D:\ah\sy_dy\JG\metrics_4_1_4_2.csv"

voxel_size = 0.04

normal_radius = 0.05
normal_max_nn = 100
ks = [20, 50, 100]

dbscan_eps = 0.2
dbscan_min_points = 10
random_state = 42
# ============================================================


# ----------------- 工具函数：统计 / 评估 -----------------
def bbox_xy_area(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    minb = np.asarray(aabb.get_min_bound())
    maxb = np.asarray(aabb.get_max_bound())
    area_xy = max(
        1e-12,
        (maxb[0] - minb[0]) * (maxb[1] - minb[1])
    )
    return area_xy, minb, maxb


def nominal_spacing_and_density(n_points, area_xy):
    if n_points <= 0 or area_xy <= 0:
        return np.nan, np.nan
    ps = np.sqrt(area_xy / n_points)
    pd = n_points / area_xy
    return ps, pd


def mahalanobis_separability(X, labels):
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2:
        return np.nan

    X0 = X[labels == 0]
    X1 = X[labels == 1]

    if len(X0) < 3 or len(X1) < 3:
        return np.nan

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    S = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])

    try:
        Sinv = np.linalg.inv(S)
        d2 = (mu0 - mu1).T @ Sinv @ (mu0 - mu1)
        return float(np.sqrt(d2))
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(mu0 - mu1))


def connectivity_and_noise_stats(labels_dbscan):
    labels = np.asarray(labels_dbscan)
    n = len(labels)

    if n == 0:
        return np.nan, np.nan, 0, 0

    noise_mask = (labels == -1)
    n_noise = int(noise_mask.sum())
    core_labels = labels[~noise_mask]

    if core_labels.size == 0:
        return 0.0, n_noise / n, 0, n_noise

    _, counts = np.unique(core_labels, return_counts=True)
    largest_cluster = counts.max()

    conn_coeff = largest_cluster / (n - n_noise)
    noise_ratio = n_noise / n

    return (
        float(conn_coeff),
        float(noise_ratio),
        int(largest_cluster),
        int(n_noise)
    )


def compute_normal_consistency_metrics(pcd, radius=0.05, max_nn=50):
    if len(pcd.points) == 0:
        return np.nan, np.nan

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )

    normals = np.asarray(pcd.normals)
    normals = normals / (
        np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    )

    R = np.linalg.norm(np.sum(normals, axis=0)) / max(1, len(normals))

    tree = o3d.geometry.KDTreeFlann(pcd)
    cos_vals = []

    for i, n in enumerate(normals):
        _, idx, _ = tree.search_radius_vector_3d(
            pcd.points[i], radius
        )
        if len(idx) >= 3:
            nn = normals[idx]
            cos_vals.append(
                float(np.mean(np.abs(nn @ n)))
            )

    local_cos = (
        float(np.mean(cos_vals))
        if len(cos_vals) > 0 else np.nan
    )

    return float(R), local_cos


def z_stats(pcd):
    if len(pcd.points) == 0:
        return np.nan, np.nan, np.nan, np.nan

    z = np.asarray(pcd.points)[:, 2]
    return (
        float(np.mean(z)),
        float(np.std(z)),
        float(np.percentile(z, 25)),
        float(np.percentile(z, 75))
    )


# ----------------- 核心：曲率计算 -----------------
def compute_curvature(neighbor_pts):
    cov = np.cov(neighbor_pts.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)
    return eigvals[0] / (np.sum(eigvals) + 1e-12)


# ========================= 主流程 =========================

# 1) 读取与降采样
pcd_raw = o3d.io.read_point_cloud(input_path)
N_raw = len(pcd_raw.points)

pcd = pcd_raw.voxel_down_sample(voxel_size=voxel_size)
N_down = len(pcd.points)

# 2) 法向估计
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius,
        max_nn=normal_max_nn
    )
)
pcd.orient_normals_consistent_tangent_plane(100)

points = np.asarray(pcd.points)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# 3) 多尺度特征
features = []
valid_indices = []

for i in range(len(points)):
    total_curvature = 0.0
    total_density = 0.0
    valid_ks = 0

    for k in ks:
        _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], k)
        if len(idx) < 5:
            continue

        neighbor_pts = points[idx]
        total_curvature += compute_curvature(neighbor_pts)

        volume = np.prod(np.ptp(neighbor_pts, axis=0)) + 1e-12
        total_density += len(idx) / volume
        valid_ks += 1

    if valid_ks == 0:
        continue

    features.append([
        total_curvature / valid_ks,
        total_density / valid_ks,
        points[i][2]
    ])
    valid_indices.append(i)

features = np.array(features)

# 4) KMeans 聚类
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=2,
    random_state=random_state,
    n_init=10,
    max_iter=300
).fit(X_scaled)

labels = kmeans.labels_

cluster_0_mean = np.mean(X_scaled[labels == 0], axis=0)
cluster_1_mean = np.mean(X_scaled[labels == 1], axis=0)

rock_label = (
    0 if cluster_0_mean[1] > cluster_1_mean[1] else 1
)

rock_indices = [
    valid_indices[i]
    for i in range(len(labels))
    if labels[i] == rock_label
]
vegetation_indices = [
    valid_indices[i]
    for i in range(len(labels))
    if labels[i] != rock_label
]

rock_pcd_before_db = pcd.select_by_index(rock_indices)
vegetation_pcd = pcd.select_by_index(vegetation_indices)

print(f"初步岩石点数: {len(rock_pcd_before_db.points)}")
print(f"草木点数: {len(vegetation_pcd.points)}")

# 5) DBSCAN 清洗
rock_labels_db = np.array(
    rock_pcd_before_db.cluster_dbscan(
        eps=dbscan_eps,
        min_points=dbscan_min_points,
        print_progress=False
    )
)

conn_coeff_pre, noise_ratio_pre, largest_pre, noise_pre = (
    connectivity_and_noise_stats(rock_labels_db)
)

if np.any(rock_labels_db >= 0):
    largest_cluster = np.argmax(
        np.bincount(rock_labels_db[rock_labels_db >= 0])
    )
    filtered_indices = np.where(
        rock_labels_db == largest_cluster
    )[0]
else:
    filtered_indices = np.array([], dtype=int)

rock_pcd = rock_pcd_before_db.select_by_index(filtered_indices)

# ----------------- 保存与可视化 -----------------
o3d.io.write_point_cloud(rock_output_path, rock_pcd)
o3d.io.write_point_cloud(vegetation_output_path, vegetation_pcd)

o3d.visualization.draw_geometries(
    [rock_pcd, vegetation_pcd],
    window_name="KMeans聚类 + DBSCAN清洗结果（含4.1/4.2指标）"
)
