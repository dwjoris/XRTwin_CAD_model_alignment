"""
==================================================================================================
ADDED:

    * definition of test/train accuracies based on test/train loss < threshold
    * global variable "threshold" for defining accuracy
    * arrays for saving train/test accuracies/losses
    * code for plotting (and saving) evolution of accuracies/losses as a function of #epochs
    * removed unused imports/variables

==================================================================================================
"""

"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



train PointNet OwnData

Train PointNet, with given parameters, for given training dataset

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Ground Truth

Output:
    - trained model

Credits: 
    PointNet Code by vinits5 as part of the Learning3D library 
    Link: https://github.com/vinits5/learning3d#use-your-own-data

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General imports
import os
import sys

import argparse

#Array related operations
import numpy as np 
import torch
import torch.utils.data
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd())) #Parent folder -> Thesis
l3D_DIR = "toolboxes/learning3d/"
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)
print(os.getcwd())
    
from toolboxes.learning3d.models import PointNet
from toolboxes.learning3d.models import Classifier
from toolboxes.learning3d.data_utils import ClassificationData, ModelNet40Data, UserData

"""
=============================================================================
------------------------------------CODE-------------------------------------
=============================================================================
"""

"""
=============================================================================
--------------------------------FILE CREATION--------------------------------
=============================================================================
"""

def _init_(args):
    if not os.path.exists(l3D_DIR+'checkpoints'):
        os.makedirs(l3D_DIR+'checkpoints') #Creates directory with checkpoints folder if it does not yet exist
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name) #Same for directory to exp_name folder
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'models') #Same for errors
    os.system('copy main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('copy model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')


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

def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    pred  = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        points, target = data
        target = target[:,0]

        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1), target, size_average=False)

        test_loss += loss_val.item()
        count += output.size(0)

        _, pred1 = output.max(dim=1)
        ag = (pred1 == target)
        am = ag.sum()
        pred += am.item()

    test_loss = float(test_loss)/count
    accuracy = float(pred)/count
    return test_loss, accuracy

def test(args, model, test_loader, textio):
    test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader)
    textio.cprint('Validation Loss: %f & Validation Accuracy: %f'%(test_loss, test_accuracy))

"""
=============================================================================
-------------------------------------TRAINING--------------------------------
=============================================================================
"""

def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    pred  = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        points, target = data
        target = target[:,0]

        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1), target, size_average=False)
        # print(loss_val.item())

        # forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += output.size(0)

        _, pred1 = output.max(dim=1)
        ag = (pred1 == target)
        am = ag.sum()
        pred += am.item()

    train_loss = float(train_loss)/count
    accuracy = float(pred)/count
    return train_loss, accuracy

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
        train_loss, train_accuracy = train_one_epoch(args.device, model, train_loader, optimizer)
        test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader)

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
        boardio.add_scalar('Train Accuracy', train_accuracy, epoch+1)
        boardio.add_scalar('Test Accuracy', test_accuracy, epoch+1)

        textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f'%(epoch+1, train_loss, test_loss, best_test_loss))
        textio.cprint('EPOCH:: %d, Traininig Accuracy: %f, Testing Accuracy: %f'%(epoch+1, train_accuracy, test_accuracy))

"""
=============================================================================
------------------------------------PARAMETERS-------------------------------
=============================================================================
"""

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_classifier', metavar='N',
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
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=200, type=int,
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

"""
=============================================================================
----------------------------------INITIALIZATION-----------------------------
=============================================================================
"""

def main():    
    args = options()
    #args.dataset_path = os.path.join(os.getcwd(), os.pardir, os.pardir, 'ModelNet40', 'ModelNet40')
    
    "------SETUP LEARNING------"
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    "--------------------------"

    "------SETUP WRITING------"
    boardio = SummaryWriter(log_dir=l3D_DIR + 'checkpoints/' + args.exp_name) #Create files in Checkpoints/Exp_name folder for acces by TensorBoard during training
    _init_(args) #Call __init__ function

    textio = IOStream(l3D_DIR+'checkpoints/' + args.exp_name + '/run.log') #append to run.log file
    textio.cprint(str(args)) #write args used into run.log file
    
    "-------------------------"

    "-------------------------"
    
    "------SETUP LOADING------"
    # trainset = ClassificationData(ModelNet40Data(train=True))
    # testset = ClassificationData(ModelNet40Data(train=False))
    
    file_loc = BASE_DIR + "/h5_files/output/files_train/ply_data_train_820IT_normal_labeled.hdf5"
    trainset = UserData('classification',file_loc)
    
    file_loc = BASE_DIR + "/h5_files/output/files_test/ply_data_test_205IT_normal_labeled.hdf5"
    testset = UserData('classification',file_loc)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
    
    "-------------------------"
    
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True)
    model = Classifier(feature_model=ptnet)

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