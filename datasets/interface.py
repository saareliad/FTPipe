import abc


class ModelParallelDatasetInterface(abc.ABC):
    # TODO:

    def get_dataloader_length(self):
        pass

    def get_separate_just_x_or_y_train_test_dl(self, just):
        pass


# Desired abilities

################
# Transforms
################

################
# Get DS
################

################
# Get DL
################

############################
# Generic "get" functions
############################

############################
# Simplified. dataset by name.
############################

##########################################################
# Distributed. (v2): Using a modified DistributedSampler
###########################################################

#############################################
# Distributed. (v3)
# get x separate from y, both with same seed
#############################################

###################################
# from args and key words.
###################################