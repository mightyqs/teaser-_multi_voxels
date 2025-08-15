import open3d as o3d
import os


def merge_pcd_files(pcd_files, output_file):
    # 创建一个空的点云对象
    merged_pcd = o3d.geometry.PointCloud()

    for pcd_file in pcd_files:
        # 读取每个 PCD 文件
        pcd = o3d.io.read_point_cloud(pcd_file)

        # 将当前点云合并到 merged_pcd
        merged_pcd += pcd

    # 保存合并后的点云为一个新的文件
    o3d.io.write_point_cloud(output_file, merged_pcd)
    print(f"合并后的点云已保存到 {output_file}")


if __name__ == "__main__":
    # 输入多个 PCD 文件的路径
    pcd_files = [
        "hku_campus_seq_00/frame_898_lidar.pcd",
        "hku_campus_seq_00/frame_899_lidar.pcd",
        "hku_campus_seq_00/frame_900_lidar.pcd",
        "hku_campus_seq_00/frame_901_lidar.pcd",
        "hku_campus_seq_00/frame_902_lidar.pcd"
        # 添加更多的文件路径
    ]

    # 输出合并后的 PCD 文件路径
    output_file = "merged_point_cloud.pcd"

    # 调用合并函数
    merge_pcd_files(pcd_files, output_file)
