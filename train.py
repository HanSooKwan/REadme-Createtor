from Data2hdf5 import convert_data_2_hdf5
from ensemble import ensemble
from mv_best_models import mv_best_models
from train_pspnet import training
from train_saunet import train_saunet_final




# Make HDF5 Files
convert_data_2_hdf5(test_directory='validation', rootdir = "echocardiography/")


# Train SAUNet
train_saunet_final(max_epoch=60)


# Train PSPNet
training()


# Move Best Models to Safe Place
mv_best_models()



# # # Ensemble Models
ensemble()