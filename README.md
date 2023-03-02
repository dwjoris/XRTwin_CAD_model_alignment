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

# Acknowledgements
Special thanks go to the authors of the available code for the used registration methods and learning3d repository.
- [Learning3d by vinit5](https://github.com/vinits5/learning3d#use-your-own-data) (RPMNet, PointNetLK, PRNet)
- [GO-ICP by aalavandhaann](https://github.com/aalavandhaann/go-icp_cython)
- [ROPNet by zhulf0804](https://github.com/zhulf0804/ROPNet)
- [Open3D](http://www.open3d.org/) (ICP, RANSAC, FGR)
