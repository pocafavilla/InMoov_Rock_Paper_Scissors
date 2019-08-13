#!/bin/bash

#cd Inmoov




#########################################################################################################
########################################## modify this: #################################################
#########################################################################################################

#1)
#if you want to keep training the model and already have checkpoints, enter the path of the PARENT DIRECTORY.
#if you want to train a new model and don't have any checkpoints to restore from, put a "#" in front of the following line:
#input_model_from=./checkpoints/checkpoint.h5


#########################################################################################################
#########################################################################################################
#########################################################################################################








if [ -z "$input_model_from" ]
then
      python model.py
else
      python model.py -restore $input_model_from
fi




#  S T A R T   F R O M   S C R A T C H  

#  A N D   C R E A T E   C H E C K P O I N T S:

#python ~/Documents/theirs/model.py -path_prediction_in $input_data_from -path_prediction $output_data -checkpoint_dir $output_model_to -nn True -upsampling_factor 2


#  A P P L Y   P R E - T R A I I N E D:

#python ~/Documents/theirs/model.py -path_prediction_in $input_data_from -path_prediction $output_data_to -checkpoint_dir $output_model_to -nn True -upsampling_factor 2 -apply $apply -restore $input_model_from



#  R E S U M E   P R E - T R A I I N E D:

#python ~/Documents/theirs/model.py -path_prediction_in $input_data_from -path_prediction $output_data -checkpoint_dir $output_model_to -nn True -upsampling_factor 2 -restore $input_model_from
























####################################################################
####################################################################
#OLD CODE
####################################################################
####################################################################


#  S T A R T   F R O M   S C R A T C H  

#  A N D   C R E A T E   C H E C K P O I N T S:

#python ~/Documents/theirs/model.py -path_prediction predictions -checkpoint_dir ./checkpoints/now_2/checkpoints -nn True -upsampling_factor 2


#  A P P L Y   P R E - T R A I I N E D:

#python ~/Documents/theirs/model.py -path_prediction predictions -checkpoint_dir ./checkpoints/now_2/checkpoints -nn True -upsampling_factor 2 -apply '/localdata/Leona_Maehler/ADNI_brain' -restore ./checkpoints/now_2/



#  R E S U M E   P R E - T R A I I N E D:

#python ~/Documents/theirs/model.py -path_prediction predictions -checkpoint_dir ./checkpoints/now_2/checkpoints -nn True -upsampling_factor 2 -restore ./checkpoints/checkpoints/

