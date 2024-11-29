import numpy as np
import open3d as o3d

pcd_file = "stacked_contour.csv"
result_path = "Mesh/mesh.obj"

if __name__ == '__main__':

    point_cloud_data = np.loadtxt(pcd_file, delimiter=",")

    # Convert the numpy array to an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    downsampled_point_cloud  = point_cloud#point_cloud.voxel_down_sample(2)

    # Apply statistical outlier removal
    cl, ind = downsampled_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    downsampled_point_cloud = downsampled_point_cloud.select_by_index(ind)
    print(len(downsampled_point_cloud.points))

    # Estimate normals for the point cloud
    downsampled_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=500))
    print("Estimate Normals done!!")
    downsampled_point_cloud.orient_normals_consistent_tangent_plane(30)
    print("Orienting to normals done!!")

    ## To invert the normals facing inside out
    # downsampled_point_cloud.normals = o3d.utility.Vector3dVector(np.asarray(downsampled_point_cloud.normals) * -1)
    # o3d.visualization.draw_geometries([downsampled_point_cloud])

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downsampled_point_cloud, depth=12)
    ## filter out low-density vertices to clean up the mesh
    vertices_to_remove = densities < np.quantile(densities, 0.0003)  # Remove 5% of vertices with the lowest densities
    mesh.compute_vertex_normals(True)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if mesh.has_vertex_normals():
        print("Good to GO")

    mesh.paint_uniform_color([0.1, 0.5, 0.5])
    # Save as .obj file
    o3d.io.write_triangle_mesh(result_path, mesh)
    o3d.visualization.draw_geometries([mesh])