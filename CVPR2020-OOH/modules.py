import time
import os
from utils.logger import Logger, savefig
import yaml
from utils.uv_map_generator import UV_Map_Generator
from utils.smpl_torch_batch import SMPLModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.imutils import uv_to_torch_noModifyChannel, img_reshape
import cv2
from utils.resample import resample_torch, resample_np
import numpy as np
import torch.utils.data as data
from utils.imutils import im_to_torch

def init(note='occlusion', dtype=torch.float32, **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join('output', note)
    out_dir = os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the log for the current experiment
    logger = Logger(os.path.join(out_dir, 'log.txt'), title="occlusion")
    logger.set_names([note])
    logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./data/model.pkl', 
                        data_type=dtype,
                    )
    # load UV generator
    generator = UV_Map_Generator(
        UV_height=256,
        UV_pickle='./data/param.pkl' #separate UV map
        #totalhuman.pickle       #connecting UV map
    )

    # load virtual occlusion
    if kwargs.get('virtual_mask'):
        occlusion_folder = os.path.join(kwargs.get('data_folder'), 'occlusion/images')
        occlusions = [os.path.join(occlusion_folder, k) for k in os.listdir(occlusion_folder)]
    else:
        occlusions = None

    return out_dir, logger, model_smpl, generator, occlusions

class ImageLoader(data.Dataset):
    def __init__(self, data_folder='./data', **kwargs):
        self.images = [os.path.join(data_folder, img) for img in os.listdir(data_folder)]
        self.len = len(self.images)

    def create_UV_maps(self, index=0):
        data = {}
        image_path = self.images[index]
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        if h != 256 or w != 256:
            max_size = max(h, w)
            ratio = 256/max_size
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            image = img_reshape(image)
        assert image.shape[0] == 256 and image.shape[1] == 256 , "The image size must be 256*256*3"

        dst_image = image
        inp = im_to_torch(dst_image)
        data['img'] = inp

        return data

    def __getitem__(self, index):
        data = self.create_UV_maps(index)
        return data

    def __len__(self):
        return self.len


class ModelLoader():
    def __init__(self, model=None, lr=0.001, device=torch.device('cpu'), pretrain=False, pretrain_dir='', output='', smpl=None, generator=None, uv_mask=None, batchsize=10, **kwargs):
        self.smpl = smpl
        self.generator = generator
        self.output = output
        self.batchsize = batchsize
        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)()
        self.device = device
        #if uv_mask:
        self.uv_mask = cv2.imread('./data/MASK.png')
        if self.uv_mask.max() > 1:
            self.uv_mask = self.uv_mask / 255.

        print('load model: %s' %self.model_type)

        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=1, verbose=True)

        # load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            premodel_dict = torch.load(pretrain_dir).state_dict()
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print("load pretrain parameters from %s" %pretrain_dir)

        # load fixed model
        if kwargs.get('task') == 'latent':
            fixmodel_dir = kwargs.pop('fixmodel_dir')
            exec('from model.inpainting import inpainting')
            self.inpainting = eval('inpainting')()
            inpainting_dict = self.inpainting.state_dict()
            fixmodel_dict = torch.load(fixmodel_dir).state_dict()
            fixmodel_dict = {k: v for k, v in fixmodel_dict.items() if k in inpainting_dict}
            inpainting_dict.update(fixmodel_dict)
            self.inpainting.load_state_dict(inpainting_dict)
            for param in self.inpainting.parameters():
                param.requires_grad = False
            self.inpainting.to(self.device)
            print("load fixed model from %s" %fixmodel_dir)

    def save_results(self, results, iter):
        """
        object order: 
        """
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        for item in results:
            index = 0
            opt = results[item]

            for img in opt:
                img_name = "%05d_%s.jpg" % (iter * self.batchsize + index, item)
                img = img.transpose(1, 2, 0)  # H*W*C

                # save mesh
                if item == 'pred' or item == 'uv_gt':
                    resample_img = img.copy()
                    resample_img = resample_img * self.uv_mask
                    resampled_mesh = resample_np(self.generator, resample_img)
                    self.smpl.write_obj(
                        resampled_mesh, os.path.join(output, '%05d_%s_mesh.obj' %(iter * self.batchsize + index, item) )
                    )
                # save img
                if item == 'pred' or item == 'uv_gt' or item == 'uv_in':
                    img = img * self.uv_mask
                    img = (img + 0.5) * 255
                else:
                    img = img * 255
                cv2.imwrite(os.path.join(output, img_name), img)

                index += 1

    def viz_result(self, rgb_img=None, masks=None, pred=None):
        masks = masks.detach().data.cpu().numpy().astype(np.float32)
        rgb_image = rgb_img.detach().data.cpu().numpy().astype(np.float32)
        img_decoded = pred.detach().data.cpu().numpy().astype(np.float32)
        for mask, rgb, img_d in zip(masks, rgb_image, img_decoded):
            mask = mask.transpose(1,2,0)
            rgb = rgb.transpose(1,2,0)
            img_d = img_d.transpose(1,2,0)
            img_d = img_d * self.uv_mask
            cv2.imshow("mask",(mask))
            cv2.imshow("rgb_img",rgb)
            cv2.imshow("d_img",(img_d+0.5))
            cv2.waitKey()