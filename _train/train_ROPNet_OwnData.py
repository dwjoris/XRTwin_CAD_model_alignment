"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



train ROPNet OwnData

Train ROPNet, with given parameters, for given training dataset

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - trained model

Credits: 
    ROPNet Code by zhulf0804
    Link: https://github.com/zhulf0804/ROPNet

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General imports
import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

BASE_DIR = os.getcwd() #Parent folder -> Thesis
# sys.path.append("C:/Users/menth/Documents/Python Scripts/Thesis")
# os.chdir(BASE_DIR)
print(BASE_DIR)

#ROPNet toolbox imports
from toolboxes.ROPNet.src.data import ModelNet40
from toolboxes.ROPNet.src.models import ROPNet
from toolboxes.ROPNet.src.loss import cal_loss
from toolboxes.ROPNet.src.metrics import compute_metrics, summary_metrics, print_train_info
from toolboxes.ROPNet.src.utils import time_calc, inv_R_t, batch_transform, setup_seed, square_dists
from toolboxes.ROPNet.src.configs import train_config_params as config_params

#learning3d toolbox imports
from toolboxes.learning3d.data_utils import UserData

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

test_min_loss, test_min_r_mse_error, test_min_rot_error = \
        float('inf'), float('inf'), float('inf')


def save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse, global_step, tag,
                 lr=None):
    for k, v in loss_all.items():
        loss = np.mean(v.item())
        writer.add_scalar(f'{k}/{tag}', loss, global_step)
    cur_r_mse = np.mean(cur_r_mse)
    writer.add_scalar(f'RError/{tag}', cur_r_mse, global_step)
    cur_r_isotropic = np.mean(cur_r_isotropic)
    writer.add_scalar(f'rotError/{tag}', cur_r_isotropic, global_step)
    if lr is not None:
        writer.add_scalar('Lr', lr, global_step)


@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer, epoch, log_freq, writer):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    
    global test_min_loss, test_min_r_mse_error, test_min_rot_error
    # for step, (tgt_cloud, src_cloud, gtR, gtt) in enumerate(tqdm(train_loader)):
    for step, (tgt_cloud, src_cloud, gt) in enumerate(tqdm(train_loader)):    
        
        gtR = gt[:, 0:3, 0:3]
        gtt = gt[:, 0:3, 3]
        
        np.random.seed((epoch + 1) * (step + 1))
        tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
                                         gtR.cuda(), gtt.cuda()

        optimizer.zero_grad()
        results = model(src=src_cloud,
                        tgt=tgt_cloud,
                        num_iter=1,
                        train=True)
        pred_Ts = results['pred_Ts']
        pred_src = results['pred_src']
        x_ol = results['x_ol']
        y_ol = results['y_ol']
        inv_R, inv_t = inv_R_t(gtR, gtt)
        gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                             inv_t)
        dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
        loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                           pred_transformed_src=pred_src,
                           dists=dists,
                           x_ol=x_ol,
                           y_ol=y_ol)

        loss = loss_all['total']
        loss.backward()
        optimizer.step()

        R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        global_step = epoch * len(train_loader) + step + 1

        if global_step % log_freq == 0:
            save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                         global_step, tag='train',
                         lr=optimizer.param_groups[0]['lr'])

        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic)
        t_isotropic.append(cur_t_isotropic)
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn, epoch, log_freq, writer):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        # for step, (tgt_cloud, src_cloud, gtR, gtt) in enumerate(tqdm(test_loader)):
        for step, (tgt_cloud, src_cloud, gt) in enumerate(tqdm(test_loader)):
            
            gtR = gt[:, 0:3, 0:3]
            gtt = gt[:, 0:3, 3]
            
            tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()

            results = model(src=src_cloud,
                            tgt=tgt_cloud,
                            num_iter=1)
            pred_Ts = results['pred_Ts']
            pred_src = results['pred_src']
            x_ol = results['x_ol']
            y_ol = results['y_ol']
            inv_R, inv_t = inv_R_t(gtR, gtt)
            gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                                 inv_t)
            dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
            loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                               pred_transformed_src=pred_src,
                               dists=dists,
                               x_ol=x_ol,
                               y_ol=y_ol)
            loss = loss_all['total']

            R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            global_step = epoch * len(test_loader) + step + 1
            if global_step % log_freq == 0:
                save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                             global_step, tag='test')

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic)
            t_isotropic.append(cur_t_isotropic)
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


def main():
    args = config_params() #Initialize training parameters (Cf. other toolbox)
    
    #Input root directory
    args.root = "C:/Users/menth/Documents/Python Scripts/Thesis/toolboxes/learning3d/data/modelnet40_ply_hdf5_2048"
    args.saved_path = os.path.join(os.getcwd(),"toolboxes/ROPNet/src/work_dirs/models")
    args.normal = True
    args.p_keep = [1, 0.7]
    
    print(args)

    setup_seed(args.seed)
    
    #Change directory to output files
    
    if not os.path.exists(args.saved_path): #Check if path for saving training checkpoints exists
        os.makedirs(args.saved_path)
        with open(os.path.join(args.saved_path, 'args.json'), 'w') as f: #Create file for saving all arguments
            json.dump(args.__dict__, f, ensure_ascii=False, indent=2)
    summary_path = os.path.join(args.saved_path, 'summary') #Check if path for summary exists
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path): #Check if path for checkpoints exists
        os.makedirs(checkpoints_path)
        
    file_train = "h5_files/output/files_train/ply_data_train_820IT_partial_0.7_normalinfo.hdf5"
    file_test = "h5_files/output/files_test/ply_data_test_205IT_partial_0.7_normalinfo.hdf5"
    
    train_set = UserData('registration',file_train)
    test_set = UserData('registration',file_test)
    
    # train_set = ModelNet40(root=args.root,
    #                         split='train',
    #                         npts=args.npts,
    #                         p_keep=args.p_keep, #Keep ratio for partial registration
    #                         noise=args.noise,
    #                         unseen=args.unseen, #Use unseen mode?
    #                         ao=args.ao, #Assymetric objects?
    #                         normal=args.normal #Use normal data?
    #                         )
    # test_set = ModelNet40(root=args.root,
    #                       split='val',
    #                       npts=args.npts,
    #                       p_keep=args.p_keep,
    #                       noise=args.noise,
    #                       unseen=args.unseen,
    #                       ao=args.ao,
    #                       normal=args.normal
    #                       )
    
    train_loader = DataLoader(train_set, batch_size=args.batchsize, #same as other toolbox
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)

    model = ROPNet(args)
    model = model.cuda()
    loss_fn = cal_loss #For computing losses
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    if args.resume: #Continue training from last checkpoint
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        
    #Scheduler is used for adjusting the learning rate over time, here the
    #Cosine Annealing Warm Restarts scheduler is used
    #Info: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=40,
                                                                     T_mult=2,
                                                                     eta_min=1e-6,
                                                                     last_epoch=-1)

    writer = SummaryWriter(summary_path)

    for i in tqdm(range(epoch)):
        for _ in train_loader:
            pass
        for _ in test_loader:
            pass
        scheduler.step() #Next step in scheduler
        
    global test_min_loss, test_min_r_mse_error, test_min_rot_error #Global variables
    
    for epoch in range(epoch, args.epoches):
        print('=' * 20, epoch + 1, '=' * 20) #Print in console
        train_results = train_one_epoch(train_loader=train_loader,
                                        model=model,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        epoch=epoch,
                                        log_freq=args.log_freq,
                                        writer=writer)
        print_train_info(train_results)

        test_results = test_one_epoch(test_loader=test_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
        print_train_info(test_results)
        
        test_loss, test_r_error, test_rot_error = \
            test_results['loss'], test_results['r_mse'], \
            test_results['r_isotropic']
        if test_loss < test_min_loss:
            saved_path = os.path.join(checkpoints_path,
                                      "min_loss.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_loss = test_loss
        if test_rot_error < test_min_rot_error:
            saved_path = os.path.join(checkpoints_path,
                                      "min_rot_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_rot_error = test_rot_error

        scheduler.step()


if __name__ == '__main__':
    main()

