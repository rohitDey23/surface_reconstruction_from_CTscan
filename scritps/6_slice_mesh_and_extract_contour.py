import numpy as np
import open3d as o3d
import copy
import glob
import os

PI = 3.14

class MeshSlicer:
    def __init__(self, mesh_path, plane_eqs):
        self.mesh, self.obb = self.pre_process_mesh(mesh_path)
        self.plane_eqs = plane_eqs
        self.all_contours = []
        self.colors = self.get_color_list(10)

    @staticmethod
    def pre_process_mesh(mesh_path):
        mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())

        obb: o3d.geometry.OrientedBoundingBox = mesh.get_oriented_bounding_box(robust=True)
        obb.color = (0, 0, 0)
        rotation_matrix = np.linalg.inv(obb.R)
        mesh.rotate(rotation_matrix)

        return mesh, obb

    @staticmethod
    def get_color_list(num):
        np.random.seed(10)
        color = [tuple(np.random.random(3) * 256) for i in range(10)]
        return color

    @staticmethod
    def signed_distance_from_plane(point, plane_eq):
        signed_dist = ((point[0] * plane_eq[0]) +
                       (point[1] * plane_eq[1]) +
                       (point[2] * plane_eq[2]) -
                       plane_eq[3])

        if signed_dist > 0:
            return 1
        elif signed_dist < 0:
            return -1
        else:
            return 0

    @staticmethod
    def get_in_plane_intersecting_vertices(plane, inter_verts):
        n = np.asarray(plane[:3]) / np.linalg.norm(np.asarray(plane[:3]))
        O = np.array([-plane[3], 0, 0])
        A = inter_verts[0, :]
        B = inter_verts[1, :]
        C = inter_verts[2, :]

        # A + K*(AB/||AB|) = b     K = (AO.n)/(AB.n)
        # A + K*(AC/||AC|) = c     K = (AO.n)/(AC.n)

        AB_norm = (B - A) / np.linalg.norm(B - A)
        AC_norm = (C - A) / np.linalg.norm(C - A)
        AO = (O - A)

        point1 = A + (np.dot(AO, n) / np.dot(AB_norm, n)) * AB_norm
        point2 = A + (np.dot(AO, n) / np.dot(AC_norm, n)) * AC_norm

        # inPlanePoints = np.r_['0, 2', point1, point2]

        return point1, point2

    @staticmethod
    def apply_inverse_plane_transform(pcd_3d, plane_eq):
        H = np.identity(4)
        ref_normal = np.array([0.0, 1.0, 0.0])
        plane_norm = np.array(plane_eq[:3]) / np.linalg.norm(np.array(plane_eq[:3]))

        axis_aligned = np.cross(ref_normal, plane_norm)
        axis_aligned = axis_aligned / np.linalg.norm(axis_aligned)

        rot_mat = o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle(
            axis_aligned * np.arccos(np.dot(ref_normal, plane_norm)))
        H[:3, :3] = rot_mat
        H[:3, 3] = (plane_norm * plane_eq[3])

        pcd_3d.transform(np.linalg.inv(H))
        pcd_3d.translate(-pcd_3d.get_center())
        pcd_3d.estimate_normals()
        pcd_2d = copy.deepcopy(pcd_3d)

        return pcd_2d


    def get_sliced_contour_points(self, plane_eq):
        trigs = self.mesh.triangles  # Triangles are sets of three vertex IDs
        verts = self.mesh.vertices   # Vertices are tuple of 3D point coordinate
        intersecting_mesh = copy.deepcopy(self.mesh)

        intersecting_trigs = []
        contour_list = []

        for trig_id, trig in enumerate(trigs):
            vert_sds = np.empty(3) # Stores the values of signed distance function of each vertex in triangle

            for id, vert_id in enumerate(trig):
                vert_sds[id] = self.signed_distance_from_plane(verts[vert_id], plane_eq)

            if not np.all(vert_sds == vert_sds[0]):  # Checking if the plane crosses this triangles
                intersecting_trigs.append(trig)
                # Determining which vertex lies on the other side of the plane isolated
                if (vert_sds == 1).sum() == 1:
                    inter_verts = np.array([verts[trig[vert_sds == 1][0]],
                                            verts[trig[vert_sds == -1][0]],
                                            verts[trig[vert_sds == -1][1]]])
                else:
                    inter_verts = np.array([verts[trig[vert_sds == -1][0]],
                                            verts[trig[vert_sds == 1][0]],
                                            verts[trig[vert_sds == 1][1]]])

                p1, p2 = self.get_new_in_plane_vertices(plane_eq, inter_verts)
                contour_list.append(p1)
                contour_list.append(p2)

        intersecting_mesh.triangles = o3d.utility.Vector3iVector(np.array(intersecting_trigs))
        intersecting_mesh.compute_vertex_normals()
        intersecting_mesh.paint_uniform_color([0.0, 1.0, 0.0])

        return intersecting_mesh, contour_list

    def run_slicer(self, pre_translation=(0,0,0), pre_rotation=(0,0,0), slicing_plane_list=[], save_path=None):
        self.mesh.translate(pre_translation)
        self.mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(pre_rotation))
        self.obb: o3d.geometry.OrientedBoundingBox = self.mesh.get_oriented_bounding_box(robust=True)
        self.obb.color = (0, 0, 0)
        intersecting_meshes = []
        all_contours = []

        for idx, plane_eq in enumerate(slicing_plane_list):
            intersect_mesh, contour_points = self.get_sliced_contour_points(plane_eq)

            contour_points = np.asarray(contour_points)
            intersecting_meshes.append(intersect_mesh)
            all_contours.append(contour_points)

            self.all_contours = all_contours

            print(f"Intersection with plane: {idx} is {len(contour_points)}")
            if save_path:
                np.savetxt(save_path + f"contour_grind_{str.zfill(str(idx), 4)}.csv", contour_points,
                           delimiter=',')

        return intersecting_meshes, all_contours

    def run_projector(self, pcd_3d_path=None, save_path=None):
        if pcd_3d_path:
            contour_file_list = [file for file in sorted(glob.glob(pcd_3d_path), key=lambda s: int(os.path.splitext(os.path.basename(s))[0][-4:]))]
            contours = [np.genfromtxt(conts_path, delimiter=',') for conts_path in contour_file_list]
        else:
            contours = copy.deepcopy(self.all_contours)

        pcd_set = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points)) for pcd_points in contours]
        pcd_projected = []
        colors = iter(self.colors)
        for pcd, plane_eq in zip(pcd_set, self.plane_eqs):
            pcd = self.apply_inverse_plane_transform(pcd, plane_eq)
            pcd.paint_uniform_color(next(colors))
            pcd_projected.append(pcd)

        if save_path:
            # cont_angles = [22,45,65,90,122,135, 157]
            # cont_angles = [600,850,1100,1350,1600,1850,2100]
            for idx, pointClouds in enumerate(pcd_projected):
                cont_name = save_path + 'flattened_contour_' + str.zfill(str(idx + 1), 4) + '.csv'
                np.savetxt(cont_name, np.asarray(pointClouds.points), delimiter=',')

        return pcd_projected


if __name__ == '__main__':
    mesh_path   = ""
    result_path_sliced_3d = ""
    result_path_sliced_2d = ""
    plane_eqs   = [(np.sin(i * PI / 8), np.cos(i * PI / 8), 0, 0) for i in range(1,8)] # 3mm curved channel after grinding
    # plane_eqs   = [(-1, 0, 0, d) for d in range(600, 2150, 250)] # 3mm tapered channel after grinding
    # plane_eqs   = [(-1, 0, 0, d) for d in range(0, -1750, -250)]
    # angle = 15  # 56: [20, -9, 15] #78[-10,-10,15]]
    # plane_eqs   =  [(np.cos(angle*PI/180), 0, np.sin(angle*PI/180), 0)]

    mesh_slicer = MeshSlicer(mesh_path, plane_eqs)
    intersecting_meshes, contour_points  =mesh_slicer.run_slicer(pre_translation=(0,0,0),
                                                                 pre_rotation=(0,0,0),
                                                                 slicing_plane_list=plane_eqs,
                                                                 save_path=result_path_sliced_3d)

    pcd_flattened = mesh_slicer.run_projector(result_path_sliced_3d, result_path_sliced_2d)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0,0,0], size=5000)
    all_mesh = [mesh_slicer.obb, coord] + intersecting_meshes +  contour_points
    o3d.visualization.draw_geometries(all_mesh, lookat = [0, 0, 0] , up = [-1, 0, 0], front=[0, 1, 0], zoom=0.75)
    o3d.visualization.draw_geometries(pcd_flattened + coord, zoom=2, lookat=[0, 0, 0], up=[0, 0, 1], front=[0, 1, 0])