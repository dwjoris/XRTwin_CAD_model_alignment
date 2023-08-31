import open3d as o3d
import numpy as np

def txt_to_array(filename):
    # :: Read .txt file and turn into array
    # Link: https://stackoverflow.com/questions/14676265/how-to-read-a-text-file-into-a-list-or-an-array-with-python
    
    list_of_lists = []

    with open(filename) as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split(' ')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            list_of_lists.append(inner_list)

    array = np.array(list_of_lists)
    return array


def array_to_ply(array):

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(array[:,0:3])
    point_cloud.normals = o3d.utility.Vector3dVector(array[:,3:6])
    return point_cloud

FileName = "Base-Top_Plate_1_BB_1_Normals.txt"
newFileName = "Base-Top_Plate_1_Source.ply"
pointsArray = txt_to_array(FileName)
pointsPLY = array_to_ply(pointsArray)
o3d.io.write_point_cloud(newFileName, pointsPLY)

pcd = o3d.io.read_point_cloud(newFileName)
o3d.visualization.draw_geometries([pcd])