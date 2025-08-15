import open3d as o3d
import teaserpp_python
import numpy as np
import copy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import time

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn)  # Removed `n_jobs=-1`
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1

def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def Rt2T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def evaluate_error(T_pred, T_gt):
    # 提取预测变换矩阵 T_pred 和真实变换矩阵 T_gt 中的旋转和平移部分
    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3, 3]

    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]

    # 计算平移误差（欧几里得距离）以及分别在 x, y, z 方向上的误差
    delta_t = t_pred - t_gt
    trans_error = np.linalg.norm(delta_t)  # 欧几里得距离（总平移误差）
    dx, dy, dz = delta_t  # x, y, z 方向上的误差

    # 计算旋转误差（角度差）
    R_diff = np.dot(R_pred.T, R_gt)  # R_pred.T @ R_gt 计算两个旋转矩阵之间的差异
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    angle_deg = np.degrees(angle)  # 转换为度

    # 输出预测的旋转矩阵和平移向量
    print("[预测的变换矩阵]:")
    print(f"  旋转矩阵 (R_pred): \n{R_pred}")
    print(f"  平移向量 (t_pred): {t_pred}")

    # 输出误差评估
    print("[误差评估]:")
    print(f"  平移误差: {trans_error:.4f} m")
    print(f"    x 方向的平移误差: {dx:.4f} m")
    print(f"    y 方向的平移误差: {dy:.4f} m")
    print(f"    z 方向的平移误差: {dz:.4f} m")
    print(f"  旋转误差: {angle_deg:.4f}°")


# ====== 自动搜索 / 自适应 voxel size ======
def run_pipeline_once_sync(source, target,
                           voxel_size_coarse,
                           voxel_size_fine=None):
    """
    同步你的口径：
    1) FPFH/TEASER++ 用 coarse 下采样
    2) ICP 用 fine 下采样
    3) ICP 的 max_correspondence_distance = NOISE_BOUND = voxel_size_coarse
    4) 评估用 icp_sol.fitness / icp_sol.inlier_rmse
    """
    if voxel_size_fine is None:
        voxel_size_fine = max(0.02, voxel_size_coarse * 0.25)

    times = {}

    # 下采样
    t0 = time.time()
    src_down_coarse = source.voxel_down_sample(voxel_size=voxel_size_coarse)
    tgt_down_coarse = target.voxel_down_sample(voxel_size=voxel_size_coarse)
    src_down_fine   = source.voxel_down_sample(voxel_size=voxel_size_fine)
    tgt_down_fine   = target.voxel_down_sample(voxel_size=voxel_size_fine)
    times['downsample'] = time.time() - t0

    # FPFH + 对应
    t0 = time.time()
    src_feats = extract_fpfh(src_down_coarse, voxel_size_coarse)
    tgt_feats = extract_fpfh(tgt_down_coarse, voxel_size_coarse)
    src_xyz = np.asarray(src_down_coarse.points).T
    tgt_xyz = np.asarray(tgt_down_coarse.points).T
    corrs_source, corrs_target = find_correspondences(src_feats, tgt_feats, mutual_filter=True)
    if len(corrs_source) < 6:
        return None  # 对应太少，跳过
    source_corr = src_xyz[:, corrs_source]
    target_corr = tgt_xyz[:, corrs_target]
    times['fpfh'] = time.time() - t0

    # TEASER++（NOISE_BOUND = coarse voxel）
    t0 = time.time()
    noise_bound = voxel_size_coarse
    solver = get_teaser_solver(noise_bound)
    solver.solve(source_corr, target_corr)
    sol = solver.getSolution()
    R_teaser, t_teaser = sol.rotation, sol.translation
    T_teaser = Rt2T(R_teaser, t_teaser)
    times['teaser'] = time.time() - t0

    # ICP（与你一致：fine 点云 + max_corr = NOISE_BOUND = coarse voxel）
    t0 = time.time()
    max_corr = noise_bound
    icp_sol = o3d.pipelines.registration.registration_icp(
        src_down_fine, tgt_down_fine, max_corr, T_teaser,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    T_icp = icp_sol.transformation
    times['icp'] = time.time() - t0

    # —— 与你完全一致的“口径”输出 —— #
    fitness = icp_sol.fitness
    rmse    = icp_sol.inlier_rmse

    metrics = {
        'fitness': fitness,
        'rmse': rmse,
        'num_corrs': source_corr.shape[1],
        'times': times,
        'voxel_coarse': voxel_size_coarse,
        'voxel_fine': voxel_size_fine,
        'max_corr_used': max_corr,  # = voxel_size_coarse
    }
    return T_teaser, T_icp, metrics



def auto_select_voxel_sync(source, target,
                           voxel_candidates=None,
                           min_fitness=0.3,
                           max_rmse=0.5,
                           prefer='rmse'):
    if voxel_candidates is None:
        voxel_candidates = list(np.geomspace(0.05, 1.0, num=8))

    best_feasible = None
    best_any = None
    best_any_score = float('inf')

    print("\n[AutoVoxel-SYNC] 候选:", [round(v, 4) for v in voxel_candidates])
    for vs in voxel_candidates:
        out = run_pipeline_once_sync(source, target, voxel_size_coarse=vs)
        if out is None:
            print(f"[AutoVoxel-SYNC] voxel={vs:.4f} -> 对应点不足，跳过。")
            continue

        T_teaser, T_icp, m = out
        print(f"[AutoVoxel-SYNC] voxel={vs:.4f} -> fitness={m['fitness']:.3f}, rmse={m['rmse']:.3f}, "
              f"corrs={m['num_corrs']}, time(teaser/icp)={m['times']['teaser']:.2f}/{m['times']['icp']:.2f}s")

        feasible = (m['fitness'] >= min_fitness) and (m['rmse'] <= max_rmse)
        if feasible:
            if best_feasible is None:
                best_feasible = (vs, T_teaser, T_icp, m)
            else:
                if prefer == 'rmse':
                    if m['rmse'] < best_feasible[3]['rmse'] - 1e-9 or \
                       (abs(m['rmse'] - best_feasible[3]['rmse']) < 1e-9 and m['fitness'] > best_feasible[3]['fitness']):
                        best_feasible = (vs, T_teaser, T_icp, m)
                else:
                    if m['fitness'] > best_feasible[3]['fitness'] + 1e-9 or \
                       (abs(m['fitness'] - best_feasible[3]['fitness']) < 1e-9 and m['rmse'] < best_feasible[3]['rmse']):
                        best_feasible = (vs, T_teaser, T_icp, m)

        score = m['rmse'] - 0.5 * m['fitness']
        if score < best_any_score:
            best_any_score = score
            best_any = (vs, T_teaser, T_icp, m)

    if best_feasible:
        print(f"\n[AutoVoxel-SYNC] 选择达标 voxel={best_feasible[0]:.4f} "
              f"(fitness={best_feasible[3]['fitness']:.3f}, rmse={best_feasible[3]['rmse']:.3f})")
        return best_feasible
    else:
        print(f"\n[AutoVoxel-SYNC] 无达标，回退到综合评分最优 voxel={best_any[0]:.4f} "
              f"(fitness={best_any[3]['fitness']:.3f}, rmse={best_any[3]['rmse']:.3f})")
        return best_any


VOXEL_SIZE = 0.5
VOXEL_SIZE_FINE = 0.1
VISUALIZE = True

# Load your source and target point clouds
source = o3d.io.read_point_cloud("HKUST03/frame_3150_lidar.pcd")
target = o3d.io.read_point_cloud("HKUST03/HKUST03.pcd")
source.paint_uniform_color([0.0, 0.0, 1.0]) # show source in blue
target.paint_uniform_color([1.0, 0.0, 0.0]) # show target in red

# Visualize original point clouds
if VISUALIZE:
    o3d.visualization.draw_geometries([source, target])  # plot original source and target

start_icp = time.time()

# 自动选择 coarse voxel
chosen = auto_select_voxel_sync(
    source, target,
    voxel_candidates=np.geomspace(0.2, 2.0, num=20),
    min_fitness=0.85,
    max_rmse=0.5,
    prefer='rmse'
)

end_voxel = time.time()
print(f"[选定 voxel 耗时]: {end_voxel - start_icp:.2f} 秒")

voxel_coarse = chosen[0]  # 返回的 coarse voxel
T_teaser = chosen[1]      # 全局配准位姿
T_icp = chosen[2]         # 局部配准位姿
metrics = chosen[3]

# 按选择的 coarse voxel 重新下采样
source_down = source.voxel_down_sample(voxel_size=voxel_coarse)
target_down = target.voxel_down_sample(voxel_size=voxel_coarse)

# Color ICP fine downsample
voxel_size_fine = voxel_coarse * 0.25  # Fine voxel is typically a smaller fraction of coarse voxel
source_down_fine = source.voxel_down_sample(voxel_size=voxel_size_fine)
target_down_fine = target.voxel_down_sample(voxel_size=voxel_size_fine)

# Visualize downsampled point clouds
if VISUALIZE:
    o3d.visualization.draw_geometries([source_down, target_down])  # plot downsampled source and target

# Convert point clouds to numpy arrays for easier manipulation
source_xyz = np.asarray(source_down.points).T  # 3 x N
target_xyz = np.asarray(target_down.points).T  # 3 x M

start_FPFH = time.time()
# Extract FPFH features
source_feats = extract_fpfh(source_down, voxel_coarse)  # Use coarse voxel size here for FPFH
target_feats = extract_fpfh(target_down, voxel_coarse)

# Establish correspondences by nearest neighbor search in feature space
corrs_source, corrs_target = find_correspondences(source_feats, target_feats, mutual_filter=True)
source_corr = source_xyz[:, corrs_source]  # 3 x num_corrs
target_corr = target_xyz[:, corrs_target]  # 3 x num_corrs

num_corrs = source_corr.shape[1]
print(f'FPFH generates {num_corrs} putative correspondences.')
end_FPFH = time.time()
print(f"[FPFH提取 耗时]: {end_FPFH - start_FPFH:.2f} 秒")

# Visualize the point clouds together with feature correspondences
points = np.concatenate((source_corr.T, target_corr.T), axis=0)
lines = []
for i in range(num_corrs):
    lines.append([i, i + num_corrs])
colors = [[0, 1, 0] for i in range(len(lines))]  # lines are shown in green
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([source_down, target_down, line_set])

start_teaser = time.time()
# Robust global registration using TEASER++
NOISE_BOUND = voxel_coarse  # Use coarse voxel size as noise bound
teaser_solver = get_teaser_solver(NOISE_BOUND)
teaser_solver.solve(source_corr, target_corr)
solution = teaser_solver.getSolution()
R_teaser = solution.rotation
t_teaser = solution.translation
T_teaser = Rt2T(R_teaser, t_teaser)
end_teaser = time.time()
print(f"[TEASER++ 耗时]: {end_teaser - start_teaser:.2f} 秒")

# Visualize the registration results
source_transformed = copy.deepcopy(source_down).transform(T_teaser)
o3d.visualization.draw_geometries([source_transformed, target_down])

start_icp = time.time()
# Local refinement using ICP
max_corr = voxel_coarse * 2.0  # max_corr = voxel_size * 2 (a typical heuristic)
icp_sol = o3d.pipelines.registration.registration_icp(
    source_down_fine, target_down_fine, max_corr, T_teaser,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
T_icp = icp_sol.transformation
end_icp = time.time()
print(f"[ICP 耗时]: {end_icp - start_icp:.2f} 秒")

# Visualize the registration after ICP refinement
source_transformed_icp = copy.deepcopy(source_down).transform(T_icp)
o3d.visualization.draw_geometries([source_transformed_icp, target_down])

# Ground truth transformation
t_gt = np.array([1.6169, 1.15465, 5.53517])
q_gt = np.array([-0.117136, 0.0107066, 0.99147, 0.0561426])  # [w, x, y, z]

# 将四元数转换为旋转矩阵
rot = R.from_quat([q_gt[1], q_gt[2], q_gt[3], q_gt[0]]).as_matrix()  # 注意四元数的顺序是 [x, y, z, w]
T_gt = np.eye(4)
T_gt[:3, :3] = rot
T_gt[:3, 3] = t_gt

# 从 TEASER++ 获取预测的变换矩阵
T_pred = T_teaser  # TEASER++ 预测的变换矩阵

# 评估预测的位姿与真实位姿之间的误差
evaluate_error(T_pred, T_gt)

T_pred2 = T_icp # ICP 预测的变换矩阵

# 评估预测的位姿与真实位姿之间的误差
evaluate_error(T_pred2, T_gt)

print("Fitness:", icp_sol.fitness)
print("RMSE:", icp_sol.inlier_rmse)
