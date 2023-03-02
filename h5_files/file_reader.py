"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



file reader

Read different types of files (.txt, .hdf5, .stl) and turn into array

Inputs:
    - .hdf5 files c
    - .txt files
    - .stl files

Output:
    - array with data

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import h5py
import open3d as o3d
import numpy as np
import torch
#import ptlk

"""
=============================================================================
-----------------------------READING FUNCTIONS-------------------------------
=============================================================================
"""
def read_stl(file_loc, Nmb_points, Normals = False):                 # Read, scale and estimate normals
    # Create point cloud
    mesh = o3d.io.read_triangle_mesh(file_loc)                       # Read Mesh file
    pcd = mesh.sample_points_uniformly(number_of_points=Nmb_points)  # Sample Mesh uniformly
    
    # Create array + scale + center
    data_array = np.asarray(pcd.points)                              # Turn PC to array of points
    
    data_mean = np.mean(data_array,0)                                # Calculate PC mean
    data_array = data_array - data_mean                              # Remove mean from PC to center
                                                                     # on (0,0,0)
                                                                     
    max = np.max(np.max(np.abs(data_array),0))
    #print(max)
    data_array = data_array/(max)
    
    # Add normals if required
    if(Normals):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        normals_array = np.asarray(pcd.normals)
        data_array = np.append(data_array,normals_array,1)
    
    return data_array


def read_Test_txt(file_loc):                # Test should be in the form of a 2D array
    file = open(file_loc,'r')
    T_list = []
    T_est = []
    lines = file.readlines()
    count = 0
    
    for line in lines:
                                            # Check for start of new matrix (indicated by '[[')
        if(line[0:1] == "["):
            L = line.split('[[')[-1]        # Split line at '[[' and keep only last element from list
            L = L.split(']')[0]             # Split line at ']' and keep only first element from list
            L = L.split(' ')                # Split at spaces 
            L = [x for x in L if x != '']   # Remove empty characters from list
            L = [float(x) for x in L]       # Transfrom strings to floats
            # print(line)
            # print(L)
            
            T_est.append(L)                 # Add line to new list 
            count = count + 1               # Increase count
            
        if(line[0] == " "):                 # Check for other lines of matrix (indicated by ' ')
            L = line.split('[')[-1]         # Split line at '[' and keep only last element from list
            L = L.split(']')[0]             # Split line at ']' and keep only first element from list
            L = L.split(' ')                # Split at spaces
            L = [x for x in L if x != '']   # Remove empty characters from list
            L = [float(x) for x in L]       # Transform strings to floats
            # print(line)
            # print(L)
            
            T_est.append(L)                 # Add line to new list
            count = count + 1               # Increase count
            
        if(count == 4):                     # When count is 4, matrix is finished (1x4x4)
            T_list.append(T_est)            # Add found matrix to list
            T_est = []                      # Reset list for new matrix
            count = 0                       # Reset count
    
    # Turn list to tensor (correct format)
    transfo_array = np.array(T_list)
    return transfo_array


def h5reader(file_dict,key):
    #Link: https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
    with h5py.File(file_dict,"r") as f:
        if(key in f.keys()):
            dataset = f[key][()]
        else:
            raise Exception("Key not included in %s" % f.keys())
    return dataset

def h5file_to_torch(file_loc):
    # Turn basic h5 file into tensors
    templ_array = h5reader(file_loc,'template')
    src_array = h5reader(file_loc,'source')
    gt_array = h5reader(file_loc,'transformation')
    
    gt_symm_array = h5reader(file_loc,'transformation_all')
    
    templ_tensor = torch.tensor(templ_array)
    src_tensor = torch.tensor(src_array)
    gt_tensor = torch.tensor(gt_array)
    gt_symm_tensor = torch.tensor(gt_symm_array)
    
    # print(templ_tensor.shape)
    # print(src_tensor.shape)
    # print(gt_tensor.shape)
    
    return templ_tensor, src_tensor, gt_tensor, gt_symm_tensor

def txt_to_array(filename):
    #Link: https://stackoverflow.com/questions/14676265/how-to-read-a-text-file-into-a-list-or-an-array-with-python
    list_of_lists = []

    with open(filename) as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split(' ')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            list_of_lists.append(inner_list)

    array = np.array(list_of_lists)
    return array

"""
=============================================================================
------------------------------DISPLAY & OTHER--------------------------------
=============================================================================
"""

def show_open3d(template,source, name = "Open3D", index = 0):
    # Create point clouds
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template[index][:][:])
    source_.points = o3d.utility.Vector3dVector(source[index][:][:])
    
    # Add color
    template_.paint_uniform_color([0,0,1])
    source_.paint_uniform_color([1,0,0])
    
    #bounding box template/source
    templ_bb = template_.get_oriented_bounding_box()
    templ_bb.color = (1, 0, 0)
    
    src_bb = source_.get_oriented_bounding_box()
    src_bb.color = (0,1,0)
    
    # Add axis system
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    # o3d.visualization.draw_geometries([template_,source_,templ_bb,src_bb])
    o3d.visualization.draw_geometries([template_,source_,mesh_frame],window_name=name)
    # volume_ratio(templ_bb, src_bb)

def volume_ratio(bb_1,bb_2):
    
    corners_1 = np.array(bb_1.get_box_points())
    corners_2 = np.array(bb_2.get_box_points())
    
    V1 = rect_volume(corners_1)
    print(":: Template volume is: ",V1)
    V2 = rect_volume(corners_2)
    print(":: Source volume is: ",V2)
    
    ratio = V2/V1
    print(":: The volume ratio of source & template is: ",ratio)
    return

def rect_volume(corners):
    l1 = np.linalg.norm(corners[1]-corners[0])
    l2 = np.linalg.norm(corners[2]-corners[0])
    l3 = np.linalg.norm(corners[3]-corners[0])
    V = l1*l2*l3
    print(V)
    return V

# """
# -------------------FROM PointNetLK-------------------
# """

# def offread(filepath, points_only=True):
#     """ read Geomview OFF file. """
#     with open(filepath, 'r') as fin:
#         mesh, fixme = _load_off(fin, points_only)
#     return mesh

# def _load_off(fin, points_only):
#     """ read Geomview OFF file. """
#     mesh = ptlk.data.mesh.Mesh()

#     fixme = False
#     sig = fin.readline().strip()
#     if sig == 'OFF':
#         line = fin.readline().strip()
#         num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
#     elif sig[0:3] == 'OFF': # ...broken data in ModelNet (missing '\n')...
#         line = sig[3:]
#         num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
#         fixme = True
#     else:
#         raise RuntimeError('unknown format')

#     for v in range(num_verts):
#         vp = tuple(float(s) for s in fin.readline().strip().split(' '))
#         mesh._vertices.append(vp)
#         #print(mesh)
        

#     if points_only:
#         return mesh, fixme

#     for f in range(num_faces):
#         fc = tuple([int(s) for s in fin.readline().strip().split(' ')][1:])
#         mesh._faces.append(fc)

#     return mesh, fixme

# """
# -----------------------------------------------------
# """