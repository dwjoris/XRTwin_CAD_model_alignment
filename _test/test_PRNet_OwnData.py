"""
==================================================================================================
ADDED:

    * possibility to use own datasets (UserData)
    * calculation of general metrics
    * time measurements for different sections
    * removed unused imports/variables

==================================================================================================
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General imports
import os

import argparse

#Array related operations
import torch
import numpy as np
import torch.utils.data

# Data loading import
from misc.dataloader import dataset_loader
from torch.utils.data import DataLoader

#testing imports
from _test.tester import test_one_epoch

BASE_DIR = os.getcwd() #Parent folder -> Thesis
# sys.path.append(BASE_DIR)
# os.chdir(BASE_DIR)
# print(BASE_DIR)
   
#learning3d toolbox imports 
from toolboxes.learning3d.models import PRNet
# from toolboxes.learning3d.data_utils import RegistrationData, ModelNet40Data, UserData

# h5_files
from h5_files.h5_writer import write_h5_result

"""
=============================================================================
----------------------------------PARAMETERS---------------------------------
=============================================================================
"""

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_prnet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')

    # settings for PointNet
    parser.add_argument('--emb_dims', default=512, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--num_iterations', default=3, type=int,
                        help='Number of Iterations')

    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--pretrained', default='learning3d/pretrained/exp_prnet/models/best_model.t7', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

"""
=============================================================================
-------------------------------------TEST------------------------------------
=============================================================================
"""

def test(args, model, test_loader, DIR):
    reg_time = test_one_epoch(args.device, model, test_loader, DIR, algo="PRNet")
    return reg_time

def main(h5_file_loc, object_name, zero_mean = False, voxel_size = 0):
    
    DIR = write_h5_result(h5_file_loc,"PRNet",voxel_size,np.zeros((0,4,4)),
                          FolderName = "results/PRNet/"+object_name)
    
    torch.cuda.empty_cache()
    args = options()
    
    #Change directory of pretrained
    # args.pretrained = os.path.join(BASE_DIR,'toolboxes/' + args.pretrained)
    
    PRE_DIR = "toolboxes/learning3d/checkpoints/exp_prnet/models/best_model.t7"
    args.pretrained = os.path.join(BASE_DIR, PRE_DIR)
    
    torch.backends.cudnn.deterministic = True
    
    # testset = RegistrationData('PRNet', ModelNet40Data(train=False),additional_params={'use_masknet': False},noise=False)
    dataset = dataset_loader(h5_file_loc,zero_mean=zero_mean,normals=False,voxel_size=voxel_size)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                             drop_last=False, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create PointNet Model.
    model = PRNet(emb_dims=args.emb_dims, num_iters=args.num_iterations)
    model = model.to(args.device)

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained), strict=False)
    model.to(args.device)

    reg_time = test(args, model, test_loader, DIR)
    
    return reg_time
    
if __name__ == '__main__':
    main()