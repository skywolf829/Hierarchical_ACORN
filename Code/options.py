import os
import json

class Options():
    def get_default():
        opt = {}
        # Input info
        opt["mode"]                    = "2D"      # What SinGAN to use - 2D, 2DTV, or 3D
        opt["target_signal"]           = "TrainingData/Snickers.h5"
        opt["save_folder"]             = "SavedModels"
        opt["save_name"]               = "Temp"    # Folder that the model will be saved to
        opt["device"]                  = "cuda:0"

        opt["num_channels"]            = 3

        opt['feat_grid_channels']       = 16
        opt['feat_grid_x']              = 32
        opt['feat_grid_y']              = 32
        opt['feat_grid_z']              = 32
        opt['local_queries_per_iter']   = 100000
        opt['min_queries_per_block']    = 16
        opt['max_blocks_per_iter']      = 16384
        opt['num_positional_encoding_terms'] = 6
        opt['FC_size_exp_start']        = 1
        opt['FC_size_exp_grow']        = 1.25
        opt['use_residual']            = False

        opt["octree_depth_start"]      = 0
        opt["octree_depth_end"]        = 6
        opt['error_bound_split']       = True
        opt['error_bound']             = 40

        opt["train_distributed"]       = False
        opt["gpus_per_node"]           = 8
        opt["num_nodes"]               = 1
        opt["node_num"]                = 0

        opt["epochs"]                  = 10000
        opt["lr"]                      = 0.001
        opt["beta_1"]                  = 0.5
        opt["beta_2"]                  = 0.999

        # Info during training (to continue if it stopped)
        opt["epoch"]                   = 0
        opt["save_every"]              = 1000
        opt["log_every"]               = 1000
        opt['log_loss']                = True
        opt['log_img']                 = False
        opt['log_CDF']                 = False
        opt['log_psnr']                = True
        opt['log_ssim']                = True

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    
def load_options(load_location):
    opt = Options.get_default()
    print(load_location)
    if not os.path.exists(load_location):
        print("%s doesn't exist, load failed" % load_location)
        return
        
    if os.path.exists(os.path.join(load_location, "options.json")):
        with open(os.path.join(load_location, "options.json"), 'r') as fp:
            opt2 = json.load(fp)
    else:
        print("%s doesn't exist, load failed" % "options.json")
        return
    
    # For forward compatibility with new attributes in the options file
    for attr in opt2.keys():
        opt[attr] = opt2[attr]

    return opt
