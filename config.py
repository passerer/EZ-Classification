class DefaultConfigs(object):
    #1.string parameters
    root = "../data/imet-2020/"
    train_data = "../data/imet-2020/train/"
    test_data =  None
    val_data = "../data/imet-2020/test/"
    model_name = "resnet"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    suffix = ".png"
    gpu = "cuda:9" 
    
    fp16 = False          # use float16 to train the model
    opt_level = "O1"      # if use fp16, "O0" means fp32，"O1" means mixed，"O2" means except BN，"O3" means only fp16
    keep_batchnorm_fp32 = False  # if use fp16,keep BN layer as fp32
    
    negtive_score = 0.05 #label smooth
    positive_score = 0.9
    
    #2.numeric parameters
    epochs = 40
    warmup_peak_epoch = 4
    batch_size = 16
    step = 2
    display_interval = 50 #display every "display_interval" iters
    img_height = 224
    img_width = 224
    size = 224
    num_classes = 3474 #number of classes
    
    seed = 888
    
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()