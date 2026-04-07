import os
import time
import math
import json
import numpy as np
import open3d as o3d


# =========================
# 工具函数
# =========================

def timer():
    """简单计时器"""
    t0 = time.time()
    return lambda: time.time() - t0


def legacy_mesh_to_t(mesh_legacy: o3d.geometry.TriangleMesh):
    """将 legacy TriangleMesh 转为 tensor 版 TriangleMesh（RaycastingScene 需要）"""
    return o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)


def point_to_mesh_distance(points_np: np.ndarray,
                           mesh_legacy: o3d.geometry.TriangleMesh,
                           chunk: int = 200_000) -> np.ndarray:
    """
    使用 RaycastingScene 计算 '点到网格' 的 unsigned 真实距离。
    若 tensor API 不可用，回退为点到点最近邻的近似（会偏大），并提示用户。
    """
    try:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(legacy_mesh_to_t(mesh_legacy))
        pts = points_np.astype(np.float32)
        dists = np.empty((len(pts),), dtype=np.float32)
        # 分块，避免一次性占用太多显存/内存
        for s in range(0, len(pts), chunk):
            e = min(s + chunk, len(pts))
            query = o3d.core.Tensor(pts[s:e], dtype=o3d.core.float32)
            dists[s:e] = scene.compute_distance(query).numpy()
        return dists.astype(np.float64)
    except Exception as e:
        print("⚠️ RaycastingScene 不可用，回退为点到点最近邻距离（会偏保守/偏大）。原因：", repr(e))
        # 回退：点到点最近邻（近似）
        # 构建一个稠密采样的网格点云用于近似距离（注意：这仍然会高估）
        mesh_dense = mesh_legacy.sample_points_poisson_disk(number_of_points=500_000)
        tree = o3d.geometry.KDTreeFlann(mesh_dense)
        dists = np.empty((len(points_np),), dtype=np.float64)
        for i, p in enumerate(points_np):
            _, idx, d2 = tree.search_knn_vector_3d(p, 1)
            if len(d2) > 0:
                dists[i] = math.sqrt(d2[0])
            else:
                dists[i] = np.nan
        return dists


def robust_threshold_from_dist(dist: np.ndarray,
                               mode: str = "quantile",
                               q: float = 0.98,
                               k_mad: float = 2.0) -> float:
    """
    从距离分布自适应得到阈值：
      - mode='quantile': q 分位数
      - mode='mad': median + k * MAD
    """
    dist = np.asarray(dist)
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0:
        return 0.0
    if mode == "quantile":
        return float(np.quantile(dist, q))
    # median + k * MAD
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) + 1e-12
    return med + k_mad * mad


def mesh_surface_area(mesh_legacy: o3d.geometry.TriangleMesh) -> float:
    """返回网格表面积（m²）"""
    return float(mesh_legacy.get_surface_area())


def aabb_xy_area(pcd: o3d.geometry.PointCloud) -> float:
    """点云 XY 投影的 AABB 面积（m²）"""
    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    return float(extent[0] * extent[1])


def format_num(n):
    if isinstance(n, int):
        return f"{n:,}"
    try:
        i = int(n)
        return f"{i:,}"
    except Exception:
        return f"{n}"


# =========================
# 1) 泊松重建 + 采样
# =========================

def mesh_reconstruct_and_resample(pcd: o3d.geometry.PointCloud,
                                  poisson_depth: int = 10,
                                  sample_points: int = 400_000,
                                  normal_radius: float = 0.1,
                                  normal_max_nn: int = 30):
    tic = timer()
    print("⛏ Poisson 网格重建...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius, max_nn=normal_max_nn))
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)
    mesh.compute_vertex_normals()

    print("🧹 网格清理...")
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_triangle_normals()
    print(f"✅ 清理后三角面数量: {format_num(len(mesh.triangles))}")

    print("🔄 网格采样（Poisson Disk）...")
    filled_pcd = mesh.sample_points_poisson_disk(number_of_points=sample_points)

    # 颜色回传（如果有）
    if pcd.has_colors():
        print("🎨 回传颜色中...")
        src_tree = o3d.geometry.KDTreeFlann(pcd)
        src_colors = np.asarray(pcd.colors)
        tgt_points = np.asarray(filled_pcd.points)
        new_colors = []
        for pt in tgt_points:
            _, idx, _ = src_tree.search_knn_vector_3d(pt, 1)
            new_colors.append(src_colors[idx[0]])
        filled_pcd.colors = o3d.utility.Vector3dVector(np.array(new_colors))

    print(f"⏱️ 重建+采样耗时: {tic():.2f} s")
    return mesh, filled_pcd


# =========================
# 2) 自适应两阶段裁剪
# =========================

def trim_poisson_shell_adaptive(filled_pcd: o3d.geometry.PointCloud,
                                original_pcd: o3d.geometry.PointCloud,
                                mesh_legacy: o3d.geometry.TriangleMesh,
                                r_support: float = 0.12,
                                dist_mode: str = "quantile",
                                q_dist: float = 0.98,
                                k_mad: float = 2.0,
                                probe_samples: int = 2000):
    """
    自适应两阶段裁剪：
      (A) 到网格的真实距离 d_mesh <= d_tau
      (B) 到原始点云的支持度：r_support 半径内的邻居数 >= k_min（邻居分布 P5%，但不小于 5）
    """
    print("🔍 自适应两阶段裁剪（Mesh 距离 + 原始支持度）...")
    filled_np = np.asarray(filled_pcd.points)

    # A) 点到网格距离阈值
    d_mesh = point_to_mesh_distance(filled_np, mesh_legacy)
    d_tau = robust_threshold_from_dist(d_mesh, mode=dist_mode, q=q_dist, k_mad=k_mad)
    keep_mesh = (d_mesh <= d_tau)

    # B) 原始点云支持度阈值（自适应 k_min）
    tree = o3d.geometry.KDTreeFlann(original_pcd)
    # 估计邻居分布
    if len(filled_np) <= probe_samples:
        probe_idx = np.arange(len(filled_np))
    else:
        probe_idx = np.random.choice(len(filled_np), size=probe_samples, replace=False)
    neighbor_counts = []
    for i in probe_idx:
        _, idx, _ = tree.search_radius_vector_3d(filled_pcd.points[i], r_support)
        neighbor_counts.append(len(idx))
    k_min = max(5, int(np.percentile(neighbor_counts, 5)))

    keep_support = np.zeros(len(filled_np), dtype=bool)
    for i, pt in enumerate(filled_pcd.points):
        if not keep_mesh[i]:
            continue
        _, idx, _ = tree.search_radius_vector_3d(pt, r_support)
        if len(idx) >= k_min:
            keep_support[i] = True

    keep_idx = np.where(keep_support)[0]
    trimmed_pcd = filled_pcd.select_by_index(keep_idx)

    removed_ratio = 1.0 - len(trimmed_pcd.points) / max(1, len(filled_pcd.points))
    print(f"✅ 最终保留点数: {format_num(len(trimmed_pcd.points))} / {format_num(len(filled_pcd.points))}")
    print(f"🚫 剔除比例: {removed_ratio*100:.2f}%")
    print(f"📏 Mesh 距离阈值 d_tau ≈ {d_tau*1000:.2f} mm, 支持度阈值 k_min = {k_min}, r_support = {r_support} m")
    return trimmed_pcd, removed_ratio, d_tau, d_mesh


# =========================
# 3) 评估指标（用于论文）
# =========================

def compute_surface_metrics(original_pcd: o3d.geometry.PointCloud,
                            mesh_legacy: o3d.geometry.TriangleMesh,
                            filled_pcd: o3d.geometry.PointCloud,
                            trimmed_pcd: o3d.geometry.PointCloud,
                            save_json: str = None):
    """
    - 网格表面积（m²）
    - AABB XY 投影面积（m²）
    - 原始/补全/修剪 点数与密度（pts/m²）
    - 原始点 → 网格 表面 RMSE（真实 point-to-mesh）
    - 误差分位数（P50, P95, P99），用于稳健报告
    """
    # 面积
    area_mesh = mesh_surface_area(mesh_legacy)
    area_trimmed_xy = aabb_xy_area(trimmed_pcd)
    area_orig_xy = aabb_xy_area(original_pcd)

    # 点数/密度
    n_orig = len(original_pcd.points)
    n_filled = len(filled_pcd.points)
    n_trim = len(trimmed_pcd.points)

    pd_orig = n_orig / max(1e-12, area_orig_xy)
    pd_trim = n_trim / max(1e-12, area_trimmed_xy)

    # 原始点 → 网格 表面距离（真实）
    d_orig_mesh = point_to_mesh_distance(np.asarray(original_pcd.points), mesh_legacy)
    d_orig_mesh = d_orig_mesh[np.isfinite(d_orig_mesh)]
    rmse = float(np.sqrt(np.mean(d_orig_mesh ** 2)))
    p50 = float(np.quantile(d_orig_mesh, 0.50))
    p95 = float(np.quantile(d_orig_mesh, 0.95))
    p99 = float(np.quantile(d_orig_mesh, 0.99))

    print("\n—— 重建与修补统计（改进评估）——")
    print(f"网格表面积：{area_mesh:.3f} m²；AABB XY（原始/修剪）：{area_orig_xy:.3f} / {area_trimmed_xy:.3f} m²")
    print(f"点数（原始/补全/修剪）：{format_num(n_orig)} / {format_num(n_filled)} / {format_num(n_trim)}")
    print(f"密度（原始/修剪）：{pd_orig:.1f} / {pd_trim:.1f} pts/m²")
    print(f"原始点 → 网格 RMSE：{rmse*1000:.2f} mm；P50/P95/P99：{p50*1000:.2f} / {p95*1000:.2f} / {p99*1000:.2f} mm\n")

    out = {
        "mesh_surface_area_m2": area_mesh,
        "aabb_xy_area_orig_m2": area_orig_xy,
        "aabb_xy_area_trim_m2": area_trimmed_xy,
        "n_points": {"orig": n_orig, "filled": n_filled, "trimmed": n_trim},
        "density_pts_per_m2": {"orig": pd_orig, "trimmed": pd_trim},
        "orig_to_mesh_rmse_m": rmse,
        "orig_to_mesh_quantiles_m": {"p50": p50, "p95": p95, "p99": p99},
    }
    if save_json:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"💾 指标已保存：{save_json}")
    return out


# =========================
# 入口：按需修改路径与参数
# =========================

if __name__ == "__main__":
    # ---- 路径（按需修改）----
    input_path = r"D:\ah\sy_dy\JG\A_ZXY.pcd"
    output_mesh_path = r"D:\ah\sy_dy\JG\cleaned_ZXY.ply"
    output_filled_path = r"D:\ah\sy_dy\JG\filled_ZXY.pcd"
    output_trimmed_path = r"D:\ah\sy_dy\JG\final_trimmed_ZXY.pcd"
    output_metrics_json = r"D:\ah\sy_dy\JG\reconstruct_metrics_ZXY.json"

    # ---- 参数（建议先用这组）----
    poisson_depth = 10
    sample_points = 400_000
    r_support = 0.12
    dist_mode = "quantile"   # "quantile" 或 "mad"
    q_dist = 0.98            # 若使用 quantile 模式，建议 0.95–0.99 之间试验
    k_mad = 2.0              # 若使用 mad 模式，建议 1.5–2.5
    normal_radius = 0.1
    normal_max_nn = 30

    # ---- 加载点云 ----
    print("📥 加载点云：", input_path)
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"  点数：{format_num(len(pcd.points))}")

    # ---- 泊松重建 + 采样 ----
    mesh, filled_pcd = mesh_reconstruct_and_resample(
        pcd,
        poisson_depth=poisson_depth,
        sample_points=sample_points,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn
    )

    # ---- 自适应裁剪 ----
    trimmed_pcd, removed_ratio, d_tau, d_mesh_all = trim_poisson_shell_adaptive(
        filled_pcd,
        pcd,
        mesh,
        r_support=r_support,
        dist_mode=dist_mode,
        q_dist=q_dist,
        k_mad=k_mad,
        probe_samples=2000
    )

    # ---- 保存中间与最终结果 ----
    print("💾 保存结果中...")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    o3d.io.write_point_cloud(output_filled_path, filled_pcd)
    o3d.io.write_point_cloud(output_trimmed_path, trimmed_pcd)
    print(f"  网格：{output_mesh_path}")
    print(f"  采样点云：{output_filled_path}")
    print(f"  修剪点云：{output_trimmed_path}")

    # ---- 评估指标 ----
    _ = compute_surface_metrics(
        original_pcd=pcd,
        mesh_legacy=mesh,
        filled_pcd=filled_pcd,
        trimmed_pcd=trimmed_pcd,
        save_json=output_metrics_json
    )

    # ---- 可视化（可选）----
    try:
        o3d.visualization.draw_geometries(
            [trimmed_pcd], window_name="最终补全结果（自适应裁剪后）")
    except Exception as e:
        print("⚠️ 可视化失败：", repr(e))
