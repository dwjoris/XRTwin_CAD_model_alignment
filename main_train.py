"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



main train

Trains selected method (RPMNet, PointNetLK, PRNet, ROPNet)

Credits: 
    PointNetLK, RPMNet, ROPNet & PRNet Code by vinits5 as part of the Learning3D library 
    Link: https://github.com/vinits5/learning3d#use-your-own-data

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

# import _train.train_PointNetLK_OwnData as PNLK_train
# import _train.train_PointNet_OwnData as PN_train
import _train.train_RPMNet_OwnData as RPMN_train
# import _train.train_PRNet_OwnData as PRNet_train
# import _train.train_ROPNet_OwnData as ROPN_train

"""
=============================================================================
------------------------------------MAIN-------------------------------------
=============================================================================
"""

if __name__ == '__main__':
    # PNLK_train.main()
    # PN_train.main()
    RPMN_train.main()
    # PRNet_train.main()
    # ROPN_train.main()
    # ROPN_train.main()