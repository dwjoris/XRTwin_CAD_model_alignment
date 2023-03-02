"""
=============================================================================
-----------------------------------CREDITS-----------------------------------
=============================================================================

PointNetLK/RPMNet/PRNet Code by vinits5 as part of the Learning3D library 
Link: https://github.com/vinits5/learning3d#use-your-own-data

Changes/additions by Menthy Denayer (2023)

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#Drawing point clouds
import open3d as o3d

#Array related operations
import torch
import time
from tqdm import tqdm

# h5_files
from h5_files.h5_writer import append_h5

# misc imports
from misc.transformer import camera_model


"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def display_open3d(template, source, transformed_source):
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()
    
    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source)
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    
    template_.paint_uniform_color([0,0,1])
    source_.paint_uniform_color([1,0,0])
    transformed_source_.paint_uniform_color([0,1,0])
    
    o3d.visualization.draw_geometries([transformed_source_,template_])

def get_transformations(igt):
    R_ba = igt[:, 0:3, 0:3]                             # Ps = R_ba * Pt
    translation_ba = igt[:, 0:3, 3].unsqueeze(2)        # Ps = Pt + t_ba
    R_ab = R_ba.permute(0, 2, 1)                        # Pt = R_ab * Ps
    translation_ab = -torch.bmm(R_ab, translation_ba)   # Pt = Ps + t_ab
    return R_ab, translation_ab, R_ba, translation_ba

def test_one_epoch(device, model, test_loader,DIR,algo):
    
    with torch.no_grad():
        model.eval()
        count = 0
        
        starting_time = 0
        prep_time = 0
        device_time = 0
        reg_time = 0
        
        start = time.time()
        # for i, data in enumerate(tqdm(test_loader)):
        for i, data in enumerate(test_loader):
            starting_time = starting_time + time.time() - start

            "--------------------preparing data--------------------"
            start = time.time()
            template, source, igt = data
            prep_time = prep_time + time.time() - start

            "--------------------assigning device--------------------"
            start = time.time()
            template = template.to(device)
            source = source.to(device)
            igt = igt.to(device)
            device_time = device_time + time.time() - start
            
            # print(template)
            # print(source)
            # print(igt)

            "--------------------performing registration--------------------"
            start = time.time()
            
            if(algo == "PointNetLK"):
                output = model(template, source)
            elif(algo == "RPMNet"):
                output = model(template, source)
            elif(algo == "PRNet"):
                transformations = get_transformations(igt)
                transformations = [t.to(device) for t in transformations]
                R_ab, translation_ab, R_ba, translation_ba = transformations
                # template - source switched because otherwise opposite result
                output = model(source, template, R_ab, translation_ab.squeeze(2)) 
                # print(output['est_T'])
                # print(R_ab)
                
            reg_time = reg_time + time.time() - start
            
            "--------------------saving results--------------------"
            
            append_h5(DIR,'Test',output['est_T'].detach().cpu().numpy()[0])
            
            "--------------------displaying results--------------------"
            # start = time.time()
            display_open3d(template.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], output['transformed_source'].detach().cpu().numpy()[0])
            # print(":: Presenting data took %.3f sec.\n" % (time.time() - start))
            count = count + 1
            
        starting_time = starting_time/count
        device_time = device_time/count
        reg_time = reg_time/count
        prep_time = prep_time/count
        
        print("\n:: Starting enumeration took %.3f sec.\n" % (starting_time))
        print(":: Preparing data took %.3f sec.\n" % (prep_time))
        print(":: Assigning device took %.3f sec.\n" % (device_time))
        print(":: Registration took %.3f sec.\n" % (reg_time))
    return reg_time

def test_one_epoch_PNLK_partial(device, model, test_loader, nmb_it):
    with torch.no_grad():
        model.eval() #Turn off certain layers during evaluation (link: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)

        #tqdm to create progress bar, enumerate returns number of current item + current item
        #Progress bar shows number of iterations in one epoch (depends on batch-size)
        start = time.time()
        for i, data in enumerate(tqdm(test_loader)):

            print("\n:: Starting enumeration took %.3f sec.\n" % (time.time() - start))

            "--------------------preparing data--------------------"
            start = time.time()
            template, source, igt = data
            print(":: Preparing data took %.3f sec.\n" % (time.time() - start))

            "--------------------assigning device--------------------"
            start = time.time()
            template = template.to(device)
            source = source.to(device)
            igt = igt.to(device)
            print(":: Assigning device took %.3f sec.\n" % (time.time() - start))

            "--------------------performing registration--------------------"
            start = time.time()

            cam_dir = torch.mean(source.detach().cpu(),1).tolist()[0]
            cam_pos = [0,0,0]

            for i in range(nmb_it):

                template_new = camera_model(template.detach().cpu(),cam_dir,cam_pos)
                template_new = template_new.expand(1,template_new.size(0),3).to(device)
                #print(template_new.shape)

                output = model(template_new, source,maxiter=1)
                #display_open3d(template_new.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], output['transformed_source'].detach().cpu().numpy()[0])

                source = output['transformed_source']

                src_mean = torch.mean(source.detach().cpu(),1).tolist()[0]
                cam_dir = src_mean
                t = output['est_t'].detach().cpu().numpy()[0]
                cam_poss = t

            print(":: Registration took %.3f sec.\n" % (time.time() - start))

            "--------------------displaying results--------------------"
            start = time.time()
            display_open3d(template.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], output['transformed_source'].detach().cpu().numpy()[0])
            print(":: Presenting data took %.3f sec.\n" % (time.time() - start))

    return output