"""
==================================================================================================
ADDED:

    * definition of test/train accuracies based on test/train loss < threshold
    * additional argument in train(test)_one_epoch for threshold variable
    * arrays for saving train/test accuracies/losses
    * code for plotting (and saving) evolution of accuracies/losses as a function of #epochs
    * removed unused imports/variables

==================================================================================================
"""

"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



train PRNet OwnData

Train PRNet, with given parameters, for given training dataset

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - trained model

Credits: 
    PRNet Code by vinits5 as part of the Learning3D library 
    Link: https://github.com/vinits5/learning3d#use-your-own-data

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
#General imports
import h5py
import os
import argparse
import time

from tensorboardX import SummaryWriter

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

#General imports
from toolboxes.learning3d.data_utils import RegistrationData, ModelNet40Data, UserData  
from toolboxes.learning3d.models import PRNet

#Misc. imports
from h5_files.h5_writer import uniquify

"""
=============================================================================
------------------------------------CODE-------------------------------------
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

def get_transformations(igt):
    R_ba = igt[:, 0:3, 0:3]                                # Ps = R_ba * Pt
    translation_ba = igt[:, 0:3, 3].unsqueeze(2)           # Ps = Pt + t_ba
    R_ab = R_ba.permute(0, 2, 1)                           # Pt = R_ab * Ps
    translation_ab = -torch.bmm(R_ab, translation_ba)      # Pt = Ps + t_ab
    return R_ab, translation_ab #, R_ba, translation_ba

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


class IOStream: #For writing to run file in checkpoints/exp_name/run.log
    def __init__(self, path):
        self.f = open(path, 'a') #open file and append info

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n') #Write text to file run.log
        self.f.flush() #clear internal buffer of the file

    def close(self):
        self.f.close() #close file

"""
=============================================================================
--------------------------------------TESTING--------------------------------
=============================================================================
"""

def test_one_epoch(device, model, test_loader, epoch):
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        count = 0
        for i, data in enumerate(tqdm(test_loader)):
            #template, source, igt = data
            
            template, source, R_ab, translation_ab = data
            
            # transformations = get_transformations(igt)
            # transformations = [t.to(device) for t in transformations]
            # R_ab, translation_ab = transformations
            #R_ab, translation_ab, R_ba, translation_ba = transformations
            
            template = template.to(device)
            source = source.to(device)
            #igt = igt.to(device)
            R_ab = R_ab.to(device)
            translation_ab = translation_ab.to(device)

            output = model(source, template, R_ab, translation_ab.squeeze(2))
            loss_val = output['loss']
            
            if(loss_val.item() > Err_Thres and epoch > Epoch_Lim): #Error cases to investigate later are saved
                write_error_h5(template,source,igt,output['est_T'],output['transformed_source'], 
                               epoch, 'test')
            
            if(loss_val.item() < threshold):
                test_acc += 1
    
            test_loss += loss_val.item()
            count += 1
    
        test_loss = float(test_loss)/count
        test_acc = float(test_acc)/count
    return test_loss, test_acc

def test(args, model, test_loader, textio):
    test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader)
    textio.cprint('Validation Loss: %f & Validation Accuracy: %f'%(test_loss, test_accuracy))

"""
=============================================================================
-------------------------------------TRAINING--------------------------------
=============================================================================
"""

def train_one_epoch(device, model, train_loader, optimizer, epoch): #similar to test_one_epoch    
    model.train() #signal we are training NOT testing
    train_loss = 0.0
    train_acc = 0.0
    count = 0
    accum_iter = 4
    
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt = data
        
        transformations = get_transformations(igt)
        transformations = [t.to(device) for t in transformations]
        R_ab, translation_ab = transformations
        #R_ab, translation_ab, R_ba, translation_ba = transformations
        
        template = template.to(device)
        source = source.to(device)
        #igt = igt.to(device)
        
        with torch.set_grad_enabled(True):
        
            output = model(source, template, R_ab, translation_ab.squeeze(2))
            loss_val = output['loss']
            
            loss_val = loss_val/accum_iter
            
            if(loss_val.item() > Err_Thres and epoch > Epoch_Lim): #Error cases to investigate later are saved
                write_error_h5(template,source,igt,output['est_T'],output['transformed_source'], 
                               epoch, 'train')
            
            loss_val.backward()   #calculate partial derivative from loss to parameters (for which requires_grad = True)
            
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                optimizer.step()      #Update parameters according to optimizer configuration
                optimizer.zero_grad() #zero gradient to prevent accumulation of losses

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

    best_test_loss = np.inf #initialize best loss as infinity

    for epoch in range(args.start_epoch, args.epochs): #loop over all epochs
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

        boardio.add_scalar('Train Loss', train_loss, epoch+1)
        boardio.add_scalar('Test Loss', test_loss, epoch+1)
        boardio.add_scalar('Best Test Loss', best_test_loss, epoch+1)
        boardio.add_scalar('Train Acc', train_acc, epoch+1)
        boardio.add_scalar('Test Acc', test_acc, epoch+1)

        textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f, Training Acc: %f, Testing Acc: %f'%(epoch+1, train_loss, test_loss, best_test_loss, train_acc, test_acc))

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
    parser.add_argument('--emb_dims', default=512, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--num_iterations', default=3, type=int,
                        help='Number of Iterations')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

"""
=============================================================================
----------------------------------INITIALIZATION-----------------------------
=============================================================================
"""

def main():
    
    file_train = "h5_files/output/files_train/ply_data_train_820IT_normal.hdf5"
    file_test = "h5_files/output/files_test/ply_data_test_205IT_normal.hdf5"
    
    args = options()
    free_gpu_cache()
    
    "------DEFINE GLOBALS------"
    global Exp_Name
    Exp_Name = args.exp_name
    
    global threshold #Threshold for accuracy (acc ++ if loss < threshold)
    threshold = 0.5
     
    global Err_Thres #Threshold beyond which errors are saved when epoch > Epoch_Lim
    Err_Thres = 2
    
    global Epoch_Lim #Threshold afterwhich errors are saved to investigate
    Epoch_Lim = 100
    "--------------------------"
    
    "------SETUP LEARNING------"
    torch.backends.cudnn.deterministic = True #To create reproducable results, only use deterministic convulution algorithms
    torch.manual_seed(args.seed) #Seed for generating random numbers
    torch.cuda.manual_seed_all(args.seed) #Sets the seed for generating random numbers on all GPUs.
    np.random.seed(args.seed) #Sets seed for numpy random generation

    "--------------------------"
    
    "------SETUP WRITING------"
    boardio = SummaryWriter(log_dir=l3D_DIR + 'checkpoints/' + args.exp_name) #Create files in Checkpoints/Exp_name folder for acces by TensorBoard during training
    _init_(args) #Call __init__ function

    textio = IOStream(l3D_DIR+'checkpoints/' + args.exp_name + '/run.log') #append to run.log file
    textio.cprint(str(args)) #write args used into run.log file
    
    "-------------------------"

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    "------SETUP LOADING------"
    trainset = UserData('registration',file_train)
    testset = UserData('registration',file_test)
    
    #trainset = RegistrationData('PRNet', ModelNet40Data(train=True), partial_source=True, partial_template=True)
    #testset = RegistrationData('PRNet', ModelNet40Data(train=False), partial_source=True, partial_template=True)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(testset, batch_size=4, shuffle=False, drop_last=False, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'

    args.device = torch.device(args.device)
    
    # Create PointNet Model.
    model = PRNet(emb_dims=args.emb_dims, num_iters=args.num_iterations)
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