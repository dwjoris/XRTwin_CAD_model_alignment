# XRTwin CAD model alignment

This repository is part of a project on XR environments. The goal is to compare different point cloud registration methods, when applied to real-world scans obtained using a RealSense depth camera. The codes are written specifically to make the comparison efficiently. The following functionalities are implemented:
- Process point cloud (.ply) files, obtained from a RealSense camera into a text file (.txt).
- Rewrite the processed point cloud (.txt) into an h5 file (.hdf5) containing the original template, source & ground truth.
- Add symmetric solutions to the obtained h5 files, when symmetry is present inside the objects.
- Test different point cloud registration methods with your own datasets, including: PointNetLK, RPMNet, PRNet, ROPNet, GO-ICP, ICP, RANSAC & FGR.
- Create files (.hdf5) to train different registration methods with different additional options (including noise, partiality...).
- Train different point cloud registration methods with your own datasets, including: PointNetLK, RPMNet, PRNet, ROPNet.
- Compute error metrics for the obtained results (from .hdf5 file result).
- Visualise the resulting transformations (from .hdf5 file result).

The functions are tested on the Cranfield Benchmark dataset and the ModelNet40 dataset.

Training datasets for PointNetLK, RPMNet, ROPNet and PRNet can be found [here](https://vub-my.sharepoint.com/:f:/g/personal/menthy_denayer_vub_be/EgztyhoVz5JLianKSp7KcxEBhoGzQ2AWnmX_uOmPsXBKbQ?e=U3EBTC)

# Documentation
The __.main__ files allow to execute the experiments performed. 
- _main_preprocess_ is used to process the raw point clouds and create the required input files (.hdf5)
- _main_train_ can be used to train the different PCR methods
- _main_test_ performs the registration for the selected PCR method, for every object, for all scans
- _main_optimize_ allows to perform the registration, for a selected PCR method and object, with varying voxel sizes
- _main_errors_ computes the errors based on the found transformations, stored in the .hdf5 result files

The __settings.json__ file contains the used settings of the D435i camera.

The __test__ and __train__ folders contain the individual codes to test and train the PCR methods.

The **datasets** folder stores the different used CAD models for the templates.

**h5_files** contains functions to process the data and create the correct .hdf5 files, as well as the input files used for the experiments, raw point clouds and result files.

Inside the **misc** folder functions are added to compute errors, visualise the results, load the data etc.

# Current Progress (Update 21/04/2023)
- [x] Installation of considered registration methods
- [x] Creation of code infrastructure to test & train methods
- [x] Creation of real-world scans for Cranfield benchmark dataset with D435i camera
- [x] Creation of real-world scans for ModelNet40 dataset with D435i camera
- [X] Training of considered methods with various training sets
- [X] Testing of learning-based methods
- [X] Testing of non-learning-based methods
- [X] Guidelines for registration & discussion of the results

# Acknowledgements
Special thanks go to the authors of the available code for the used registration methods and learning3d repository.
- [Learning3d by vinit5](https://github.com/vinits5/learning3d#use-your-own-data) (RPMNet, PointNetLK, PRNet)
- [GO-ICP by aalavandhaann](https://github.com/aalavandhaann/go-icp_cython)
- [ROPNet by zhulf0804](https://github.com/zhulf0804/ROPNet)
- [Open3D](http://www.open3d.org/) (ICP, RANSAC, FGR)
