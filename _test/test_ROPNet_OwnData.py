"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General imports
import numpy as np
import os
import random
import time
import torch

BASE_DIR = os.getcwd() #Parent folder -> Thesis
# sys.path.append(BASE_DIR)
# os.chdir(BASE_DIR)
#print(BASE_DIR)

#ROPNet toolbox imports
import toolboxes.ROPNet.src.vis as vis
from toolboxes.ROPNet.src.configs import eval_config_params
from toolboxes.ROPNet.src.data import ModelNet40
from toolboxes.ROPNet.src.models import ROPNet, gather_points
from toolboxes.ROPNet.src.utils import npy2pcd, pcd2npy, inv_R_t, batch_transform, square_dists, \
    format_lines, vis_pcds
from toolboxes.ROPNet.src.metrics import compute_metrics, summary_metrics, print_metrics

# Data loading import
from misc.dataloader import dataset_loader
from torch.utils.data import DataLoader

# h5_files
from h5_files.h5_writer import write_h5_result, append_h5

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        # print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

# Function to save the estimated transformation in the correct format
def save_predicted_transformation(R,t,DIR):
    T = np.zeros((4,4))
    T[3,3] = 1
    T[0:3,0:3] = np.array(R.tolist()[0])
    T[:3,3] = np.array(t.tolist()[0])
    append_h5(DIR,'Test',T)

def evaluate_ROPNet(args, test_loader, DIR):
    model = ROPNet(args)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    dura = []
    count = 0
    tot_reg_time = 0
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    src_recalls_op, src_precs_op = [], []
    src_recalls_rop, src_precs_rop = [], []
    with torch.no_grad():
        # for i, (tgt_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
        for i, (tgt_cloud, src_cloud, gt) in enumerate(test_loader):
            
            print(src_cloud.shape)
            gtR = gt[:, 0:3, 0:3]
            gtt = gt[:, 0:3, 3]
            
            if args.cuda:
                tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()
            tic = time.time()
            B, N, _ = src_cloud.size()
            results = model(src=src_cloud,
                            tgt=tgt_cloud,
                            num_iter=2)
            toc = time.time()
            reg_time = toc - tic
            dura.append(reg_time)
            pred_Ts = results['pred_Ts']
            
            R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
            
            # Save estimated transformations
            save_predicted_transformation(R,t,DIR)
            count = count + 1
            tot_reg_time = tot_reg_time + reg_time

            # for overlap evaluation
            src_op = results['src_ol1']
            src_rop = results['src_ol2']
            inv_R, inv_t = inv_R_t(gtR, gtt)
            dist_thresh = 0.05
            gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                                 inv_t)
            dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
            src_ol_gt = torch.min(dists, dim=-1)[0] < dist_thresh * dist_thresh

            gt_transformed_src_op = batch_transform(src_op[..., :3], inv_R,
                                                     inv_t)
            dists_op = square_dists(gt_transformed_src_op, tgt_cloud[..., :3])
            src_op_pred = torch.min(dists_op, dim=-1)[0] < dist_thresh * dist_thresh
            src_prec = torch.sum(src_op_pred, dim=1) / \
                       torch.sum(torch.min(dists_op, dim=-1)[0] > -1)
            src_recall = torch.sum(src_op_pred, dim=1) / torch.sum(src_ol_gt, dim=1)
            src_precs_op.append(src_prec.cpu().numpy())
            src_recalls_op.append(src_recall.cpu().numpy())

            gt_transformed_src_rop = batch_transform(src_rop[..., :3], inv_R,
                                                 inv_t)
            dists_rop = square_dists(gt_transformed_src_rop, tgt_cloud[..., :3])
            src_rop_pred = torch.min(dists_rop, dim=-1)[0] < dist_thresh * dist_thresh
            src_prec = torch.sum(src_rop_pred, dim=1) / \
                       torch.sum(torch.min(dists_rop, dim=-1)[0] > -1)
            src_recall = torch.sum(src_rop_pred, dim=1) / \
                         torch.sum(src_ol_gt, dim=1)
            src_precs_rop.append(src_prec.cpu().numpy())
            src_recalls_rop.append(src_recall.cpu().numpy())

            # src_op = npy2pcd(torch.squeeze(gt_transformed_src_op).cpu().numpy())
            # src_rop = npy2pcd(torch.squeeze(gt_transformed_src_rop).cpu().numpy())
            # tgt_cloud = npy2pcd(torch.squeeze(tgt_cloud[..., :3]).cpu().numpy())
            # vis_pcds([tgt_cloud, src_op])
            # vis_pcds([tgt_cloud, src_rop])

            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic)
            t_isotropic.append(cur_t_isotropic)

    print('=' * 20, 'Overlap', '=' * 20)
    print('OP overlap precision: ', np.mean(src_precs_op))
    print('OP overlap recall: ', np.mean(src_recalls_op))
    print('ROP overlap precision: ', np.mean(src_precs_rop))
    print('ROP overlap recall: ', np.mean(src_recalls_rop))

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, tot_reg_time/count


def main(h5_file_loc, object_name, zero_mean = True, voxel_size = 0, p_keep = [1, 1]):
    
    DIR = write_h5_result(h5_file_loc,"ROPNet", voxel_size, np.zeros((0,4,4)),
                          FolderName = "results/ROPNet/"+object_name)
    
    seed = 222
    random.seed(seed)
    np.random.seed(seed)

    args = eval_config_params()
    
    args.cuda = True
    args.normal = True
    args.p_keep = p_keep #If only one value => source partial, template complete
    
    #Change directory of pretrained
    # checkpoint_dir = "ROPNet/src/work_dirs/models/checkpoints/min_rot_error.pth"
    checkpoint_dir = "ROPNet/src/work_dirs/partial_0.7_noisy_0.01_floor/models/checkpoints/min_rot_error.pth"
    args.checkpoint = os.path.join(os.getcwd(),'toolboxes/' + checkpoint_dir)
    
    #Input root directory
    args.root = "C:/Users/menth/Documents/Python Scripts/Thesis/toolboxes/learning3d/data/modelnet40_ply_hdf5_2048"
    
    print(args)

    # test_set = ModelNet40(root=args.root,
    #                       split='test',
    #                       npts=args.npts,
    #                       p_keep=args.p_keep,
    #                       noise=args.noise,
    #                       unseen=args.unseen,
    #                       ao=args.ao,
    #                       normal=args.normal)
    
    # Create dataset with desired properties
    dataset = dataset_loader(h5_file_loc,zero_mean=zero_mean,normals=True,voxel_size=voxel_size)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, tot_reg_time = \
        evaluate_ROPNet(args, test_loader,DIR)
    print_metrics('ROPNet', dura, r_mse, r_mae, t_mse, t_mae,
                  r_isotropic,
                  t_isotropic)
    # vis.vis_ROPNet(args, test_loader)
    
    return tot_reg_time

if __name__ == '__main__':
    main()