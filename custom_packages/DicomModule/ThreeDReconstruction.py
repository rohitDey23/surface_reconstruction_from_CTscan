import numpy as np
import open3d as o3d


class ThreeDReconstructor:
    def __init__(self):
        self.layered_contours = []

    def read_npy_file(self, npy_file_path):
        pcd_point = np.load(npy_file_path)
        pcd_point = pcd_point.reshape((-1, 3))
        return pcd_point[1:, :]

    def get_pcd_filtered(self, npy_pcd):
        pcd = o3d.t.geometry.PointCloud(npy_pcd)
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        # pcd = pcd.uniform_down_sample(every_k_points=50)
        cl, ind = pcd.remove_statistical_outliers(nb_neighbors=100, std_ratio=1)
        pcd_trimmed = pcd.select_by_mask(ind)
        return pcd_trimmed, ind

    def visualize_inlier_outlier(self, cloud: o3d.t.geometry.PointCloud, mask: o3d.core.Tensor):
        inlier_cloud = cloud.select_by_mask(mask)
        outlier_cloud = cloud.select_by_mask(mask, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud = outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    def visualize_pcd(self, pcd, normals=True):
        pcd.estimate_normals(max_nn=100, radius=0.2)
        pcd.orient_normals_consistent_tangent_plane(1000)
        o3d.visualization.draw_geometries([pcd.to_legacy()],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024],
                                          point_show_normal=normals)

    def create_triangular_mesh(self, pcd_points: o3d.t.geometry.PointCloud):
        pcd_points.estimate_normals(max_nn=100, radius=0.2)
        pcd_points.orient_normals_consistent_tangent_plane(500)
        print(f"The average distance between each point is 0.2")

        # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_points.to_legacy(),
        #                                                                            o3d.utility.DoubleVector([0.1, 0.4, 1.2, 1.5]))
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd_points.to_legacy(), depth=11, width=0, scale=1.5, linear_fit=False)[0]

        return bpa_mesh


