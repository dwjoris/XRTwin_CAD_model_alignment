"""
==================================================================================================
CHANGELOG:

    * added definition of test/train accuracies based on test/train loss < threshold
    * added global variable "threshold" for defining accuracy
    * added accuracies for saving
    * removed unused imports/variables
    * added epoch to test_one_epoch and train_one_epoch functions
    * added code for saving unusual cases to .h5 file
    * added global variable "Err_Thres" and "Epoch_Lim" for saving errors to investigate
    * added global variable "Exp_Name" for using args.exp_name in other functions

==================================================================================================
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General imports
import h5py
import os

from tensorboardX import SummaryWriter
import argparse

#Array related operations
import numpy as np 
import torch
import torch.utils.data

from torch.utils.data import DataLoader
from tqdm import tqdm

#Memory related imports
from GPUtil import showUtilization as gpu_usage
from numba import cuda

BASE_DIR = os.getcwd() #Parent folder -> Thesis
l3D_DIR = "toolboxes/learning3d/"
#print(BASE_DIR)

#learning3d toolbox imports
from toolboxes.learning3d.models import RPMNet, PPFNet

from toolboxes.learning3d.losses import FrobeniusNormLoss, RMSEFeaturesLoss
from toolboxes.learning3d.data_utils import RegistrationData, ModelNet40Data, UserData

#Misc. imports
from h5_files.h5_writer import uniquify

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def free_gpu_cache(): #Code from: https://www.kaggle.com/getting-started/140636 by Moradnejad
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def _init_(args):
    if not os.path.exists(l3D_DIR+'checkpoints'):
        os.makedirs(l3D_DIR+'checkpoints') #Creates directory with checkpoints folder if it does not yet exist
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name) #Same for directory to exp_name folder
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'models') #Same for errors
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'errors'):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'errors') #Same for errors
    # os.system('copy main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    # os.system('copy model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')

def write_error_h5(template,source,igt,est_T,transfo_source, epoch, part):
    if not os.path.exists(l3D_DIR+'checkpoints/' + Exp_Name + '/' + 'errors/EP'+str(epoch)):
        os.makedirs(l3D_DIR+'checkpoints/' + Exp_Name + '/' + 'errors/EP'+str(epoch)) #Same for errors
    
    DIR = l3D_DIR +'checkpoints/' + Exp_Name + '/errors/EP'+str(epoch)+'/'
    DIR = DIR + part +'_' + 'error' + ".hdf5"
    DIR = uniquify(DIR)
    
    template = template.detach().cpu().numpy()
    source = source.detach().cpu().numpy()
    transfo_source = transfo_source.detach().cpu().numpy()
    igt = igt.detach().cpu().numpy()
    est_T = est_T.detach().cpu().numpy()
    
    with h5py.File(DIR,"w") as data_file:
        data_file.create_dataset("template",data=template)
        data_file.create_dataset("source",data=source)
        data_file.create_dataset("gt",data=igt)
        data_file.create_dataset("transformed_source",data=transfo_source)
        data_file.create_dataset("est_T",data=est_T)
    return

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

"""
=============================================================================
--------------------------------------TESTING--------------------------------
=============================================================================
"""

def test_one_epoch(device, model, test_loader, epoch):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        output = model(template, source, 2)
        
        # loss_val = FrobeniusNormLoss()(output['est_T'], igt) + RMSEFeaturesLoss()(output['r'])
        loss_val = FrobeniusNormLoss()(output['est_T'], igt)
        
        if(loss_val.item() > Err_Thres and epoch > Epoch_Lim): #Error cases to investigate later are saved
            write_error_h5(template,source,igt,output['est_T'],output['transformed_source'], 
                           epoch, 'test')

        if(loss_val.item() < threshold):
            test_acc += 1
            
        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss)/count
    test_acc = float(test_acc)/count
    return test_loss,test_acc

def test(args, model, test_loader, textio):
    test_loss, test_acc = test_one_epoch(args.device, model, test_loader)
    textio.cprint('Validation Loss: %f & Validation Accuracy: %f'%(test_loss, test_acc))

"""
=============================================================================
-------------------------------------TRAINING--------------------------------
=============================================================================
"""

def train_one_epoch(device, model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        output = model(template, source)
        
        loss_val = FrobeniusNormLoss()(output['est_T'], igt)
        # loss_val = FrobeniusNormLoss()(output['est_T'], igt) + RMSEFeaturesLoss()(output['r'])
        
        if(loss_val.item() > Err_Thres and epoch > Epoch_Lim): #Error cases to investigate later are saved
            write_error_h5(template,source,igt,output['est_T'],output['transformed_source'], 
                           epoch, 'train')
        
        # forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if(loss_val.item() < threshold):
            train_acc += 1
        
        train_loss += loss_val.item()
        count += 1

    train_loss = float(train_loss)/count
    train_acc = float(train_acc)/count
    return train_loss, train_acc

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(args.device, model, train_loader, optimizer, epoch+1)
        test_loss, test_acc = test_one_epoch(args.device, model, test_loader, epoch+1)

        if test_loss<best_test_loss:
            best_test_loss = test_loss
            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer' : optimizer.state_dict(),}
            torch.save(snap,l3D_DIR+ 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
            torch.save(model.state_dict(),l3D_DIR+ 'checkpoints/%s/models/best_model.t7' % (args.exp_name))
            torch.save(model.feature_model.state_dict(),l3D_DIR+ 'checkpoints/%s/models/best_ptnet_model.t7' % (args.exp_name))

        torch.save(snap, l3D_DIR+'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
        torch.save(model.state_dict(), l3D_DIR+'checkpoints/%s/models/model.t7' % (args.exp_name))
        torch.save(model.feature_model.state_dict(), l3D_DIR+'checkpoints/%s/models/ptnet_model.t7' % (args.exp_name))

        boardio.add_scalar('Train Loss', train_loss, epoch+1)
        boardio.add_scalar('Test Loss', test_loss, epoch+1)
        boardio.add_scalar('Best Test Loss', best_test_loss, epoch+1)
        boardio.add_scalar('Train Acc', train_acc, epoch+1)
        boardio.add_scalar('Test Acc', test_acc, epoch+1)

        textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f, Training Acc: %f, Testing Acc: %f'%(epoch+1, train_loss, test_loss, best_test_loss, train_acc, test_acc))

"""
=============================================================================
------------------------------------PARAMETERS-------------------------------
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
    parser.add_argument('--fine_tune_pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--transfer_ptnet_weights', default='./checkpoints/exp_classifier/models/best_ptnet_model.t7', type=str,
                        metavar='PATH', help='path to pointnet features file')
    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=400, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

def main():
    file_train = "h5_files/output/files_train/ply_data_train_820IT_partial_0.5.hdf5"
    file_test = "h5_files/output/files_test/ply_data_test_205IT_partial_0.5.hdf5"
    
    args = options()
    free_gpu_cache()
    
    "------DEFINE GLOBALS------"
    global Exp_Name
    Exp_Name = args.exp_name
    
    global threshold #Threshold for accuracy (acc ++ if loss < threshold)
    threshold = 0.01
     
    global Err_Thres #Threshold beyond which errors are saved when epoch > Epoch_Lim
    Err_Thres = 1
    
    global Epoch_Lim #Threshold afterwhich errors are saved to investigate
    Epoch_Lim = 50
    
    "--------------------------"
    
    "------SETUP LEARNING------"
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    "--------------------------"
    
    "------SETUP WRITING------"
    boardio = SummaryWriter(log_dir=l3D_DIR+'checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream(l3D_DIR+'checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))
    "-------------------------"
    
    "------SETUP LOADING------"
    trainset = UserData('registration',file_train)
    testset = UserData('registration',file_test)

    # trainset = RegistrationData('RPMNet', 
    #                             ModelNet40Data(train=True, num_points=args.num_points, 
    #                                            use_normals=True), 
    #                             partial_source=True, partial_template=True,
    #                             additional_params={'use_masknet': False})
    
    # testset = RegistrationData('RPMNet', 
    #                            ModelNet40Data(train=False, num_points=args.num_points, 
    #                                           use_normals=True), 
    #                            partial_source=True, partial_template=True,
    #                            additional_params={'use_masknet': False})
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(testset, batch_size=4, shuffle=False, 
                             drop_last=False, num_workers=args.workers)
    
    "-------------------------"
    
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create RPMNet Model.
    model = RPMNet(feature_model=PPFNet())
    model = model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    if args.eval:
        test(args, model, test_loader, textio)
    else:
        train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
    main()