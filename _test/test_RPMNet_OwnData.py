"""
=============================================================================
-----------------------------------CREDITS-----------------------------------
=============================================================================

RPMNet Code by vinits5 as part of the Learning3D library 
Link: https://github.com/vinits5/learning3d#use-your-own-data

Changes/additions by Menthy Denayer (2023)

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

# General imports
import os
import argparse
import numpy as np

# Array related operations
import torch
import torch.utils.data

# Data loading import
from misc.dataloader import dataset_loader
from torch.utils.data import DataLoader

# Testing imports
from _test.tester import test_one_epoch

BASE_DIR = os.getcwd() #Parent folder -> Thesis
# print(BASE_DIR)
   
# Learning3d toolbox imports
from toolboxes.learning3d.models import RPMNet, PPFNet
# from toolboxes.learning3d.data_utils import RegistrationData, ModelNet40Data, UserData

# h5_files
from h5_files.h5_writer import write_h5_result

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_rpmnet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--pretrained', default='learning3d/pretrained/exp_rpmnet/models/partial-trained.pth', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

def test(args, model, test_loader, DIR):
    reg_time = test_one_epoch(args.device, model, test_loader, DIR,algo="RPMNet")
    return reg_time

def main(h5_file_loc,object_name,zero_mean=True,voxel_size=0):
    
    DIR = write_h5_result(h5_file_loc,"RPMNet",np.zeros((0,4,4)),
                          FolderName = "results/RPMNet/"+object_name)
    
    torch.cuda.empty_cache()
    args = options()
    
    #Change directory of pretrained
    PRE_DIR = "toolboxes/learning3d/checkpoints/exp_rpmnet/models/best_model.t7"
    # PRE_DIR = "toolboxes/learning3d/checkpoints/best_model.t7"
    args.pretrained = os.path.join(BASE_DIR, PRE_DIR)

    #testset = RegistrationData('RPMNet', ModelNet40Data(train=False, num_points=args.num_points, 
    # use_normals=True), partial_source=True, partial_template=False)
    
    dataset = dataset_loader(h5_file_loc,zero_mean=zero_mean,normals=False,voxel_size=voxel_size)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create RPMNet Model.
    model = RPMNet(feature_model=PPFNet())
    model = model.to(args.device)

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
        #model.load_state_dict(torch.load(args.pretrained, map_location='cpu')['state_dict'],strict=False)
    model.to(args.device)
    
    # Solve with RPMNet
    reg_time = test(args, model, test_loader, DIR)
    return reg_time

if __name__ == '__main__':
    main()