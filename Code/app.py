from flask import Flask, render_template, Response, jsonify, request, json
import os
from datetime import datetime
import sys
import base64
from options import load_options
from models import load_model
import random
from utility_functions import to_img, to_pixel_samples, PSNR
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from torchvision.utils import make_grid
from models import PositionalEncoding

file_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(file_folder_path, "..")

input_folder = os.path.join(project_folder_path, "TrainingData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

load_from = "pluto_dynamic"#"Snick_5levels_dynamic_error"

template_folder = os.path.join(file_folder_path, 'InteractiveWebapp', 
    'templates')
static_folder = os.path.join(file_folder_path, 'InteractiveWebapp', 
    'static')
app = Flask(__name__, template_folder=template_folder, 
    static_folder=static_folder)

global model
global item

model_name = "Cat_"


def load_model_and_item(name):
    global model
    opt = load_options(os.path.join(save_folder, name))
    opt["device"] = "cuda:0"
    opt["save_name"] = name

    item = h5py.File(os.path.join(project_folder_path, opt['target_signal']), 'r')['data']
    item = torch.tensor(item).unsqueeze(0).to(opt['device'])

    model = load_model(opt, opt['device'])
    return model, item

def log_visitor():
    visitor_ip = request.remote_addr
    visitor_requested_path = request.full_path
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")

    pth = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(pth,"log.txt"), "a")
    f.write(dt + ": " + str(visitor_ip) + " " + str(visitor_requested_path) + "\n")
    f.close()

@app.route('/')
def index():
    log_visitor()
    return render_template('index.html')

@app.route('/reconstruct_target_signal')
def reconstruct_target_signal():
    global model
    global item
    with torch.no_grad():        
        reconstruction = model.get_full_img_no_residual()    
        im = reconstruction.clamp_(0,1)[0].permute(1, 2, 0).cpu().numpy()    
        psnr = PSNR(reconstruction, item, 1.0).item()
    success, reconstructed_im = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    reconstructed_im = reconstructed_im.tobytes()
    return jsonify(
            {
                "reconstructed_im":str(base64.b64encode(reconstructed_im)),
                "psnr": "%0.02f" % psnr
            }
        )

@app.route('/reconstruct_target_signal_blocks')
def reconstruct_target_signal_blocks():
    global model
    global item
    with torch.no_grad():        
        reconstruction = model.get_full_img_no_residual()    
        octree_blocks = model.octree.get_octree_block_img(model.opt['device'])  
        im = (reconstruction.clamp_(0,1)*octree_blocks)[0].permute(1, 2, 0).cpu().numpy()    
    success, reconstructed_im = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    reconstructed_im = reconstructed_im.tobytes()
    return jsonify(
            {
                "reconstructed_im_blocks":str(base64.b64encode(reconstructed_im))
            }
        )

@app.route('/get_target_signal')
def get_target_signal():
    global item
    im = item[0].permute(1, 2, 0).cpu().numpy()    
    success, im = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    im = im.tobytes()
    return jsonify(
            {
                "im":str(base64.b64encode(im))
            }
        )

@app.route('/get_block_zoom')
def get_block_zoom():
    global model
    block_x = float(request.args.get('x_click'))
    block_y = float(request.args.get('y_click'))
    block_zoom = float(request.args.get('zoom'))

    with torch.no_grad():
        blocks = []

        for depth in range(model.octree.min_depth(), model.octree.max_depth()+1):
            for block in model.octree.depth_to_nodes[depth]:
                if(block.pos[0] <= block_x and block.pos[0]+block.shape[2] >= block_x and \
                    block.pos[1] <= block_y and block.pos[1]+block.shape[3] >= block_y):
                    blocks.append(block)
        
        smallest_block = blocks[-1]
        
        final_size = []
        for i in range(2, len(smallest_block.shape)):
            final_size.apend(smallest_block[i]*block_zoom)

        out = torch.zeros(final_size, dtype=torch.float32, device=model.opt['device'])
        model_no = 0

        pe = PositionalEncoding(model.opt)
        for block in blocks:
            block_position = torch.tensor([block.pos], 
                            device=model.opt['device'])
            block_position = pe(block_position)
            feat_grid = model.models[model_no].feature_encoder(block_position)
            model.models[model_no].feat_grid_shape[0] = 1
            feat_grid = feat_grid.reshape(model.models[model_no].feat_grid_shape)

            block_output = F.interpolate(feat_grid, size=block.shape[2:], 
                            mode='bilinear' if "2D" in model.opt['mode'] else "trilinear", 
                            align_corners=False)
            offset_x = smallest_block.pos[0] - block.pos[0]
            offset_y = smallest_block.pos[1] - block.pos[1]

            block_output = block_output[:,:,
                offset_x:offset_x+smallest_block.shape[2], 
                offset_y:offset_y+smallest_block.shape[3]]

            block_output = F.interpolate(block_output, scale_factor=block_zoom, 
                            mode='bilinear' if "2D" in model.opt['mode'] else "trilinear", 
                            align_corners=False)

            block_output = model.models[model_no].feature_decoder(block_output)
            out += block_output
        block_im = out.clamp_(0,1)[0].permute(1, 2, 0).cpu().numpy()

    success, block_im = cv2.imencode(".png", cv2.cvtColor(block_im, cv2.COLOR_BGR2RGB))
    reconstructed_im = block_im.tobytes()
    return jsonify(
            {
                "block_im":str(base64.b64encode(block_im))
            }
        )

if __name__ == '__main__':
    
    model, item = load_model_and_item()
    
    app.run(host='127.0.0.1',debug=True,port="12345")
    #app.run(host='0.0.0.0',debug=False,port="80")