
def linear_lr_scaling(bs_train, BASE_LR, BASE_BS_TRAIN, downscale=False):

    if bs_train < BASE_BS_TRAIN:
        if not downscale:
            return BASE_LR
        else:
            lr = BASE_LR / (BASE_BS_TRAIN / bs_train)
    else:
        lr = BASE_LR * (bs_train / BASE_BS_TRAIN)
    
    assert(lr > 0)
    return lr
