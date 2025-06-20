ShuffleNet_cfg = {
    'model_type': 'ShuffleNet',
    #'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    #'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr': 0.004,
    'milestones': [0,1,2,3,4,5,],
    'num_out': 136,
    'num_epoch': 6,
    'momentum': 0.9,
    'weight_decay': 0,
    
}