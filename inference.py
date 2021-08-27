import glob
import os

import numpy as np
import torch
from PIL import Image
from skimage.io import imsave
from tqdm import tqdm

from model.upflow import UPFlow_net
from utils.flow_viz import flow_to_image
from utils.tools import tools

DEVICE = 'cuda'


def viz(img1, img2, fw_bw_flo, out_im_file):
    img1 = img1[0].permute(1, 2, 0).cpu().numpy()
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()

    fw_flo = fw_bw_flo[0]
    bw_flo = fw_bw_flo[1]
    fw_flo = fw_flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    fw_flo = flow_to_image(fw_flo)
    fw_out_filename = out_im_file.replace(".jpg", '_fw_flow.png')
    imsave(fw_out_filename, fw_flo)

    bw_flo = bw_flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    bw_flo = flow_to_image(bw_flo)
    bw_out_filename = out_im_file.replace(".jpg", '_bw_flow.png')
    imsave(bw_out_filename, bw_flo)

    img_flo = np.concatenate([img1, img2, fw_flo, bw_flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.tight_layout()
    plt.savefig(out_im_file)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]]/255.0)
    # cv2.waitKey()

def load_image(img_file):
    img = np.array(Image.open(img_file)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


class TestModel(tools.abs_test_model):
    def __init__(self, pretrain_path='./scripts/upflow_kitti2015.pth', use_cuda=True):
        super(TestModel, self).__init__()
        param_dict = {
            # use cost volume norm
            'if_norm_before_cost_volume': True,
            'norm_moments_across_channels': False,
            'norm_moments_across_images': False,
            'if_froze_pwc': False,
            'if_use_cor_pytorch': False,  # speed is very slow, just for debug when cuda correlation is not compiled
            'if_sgu_upsample': True,
        }
        net_conf = UPFlow_net.config()
        net_conf.update(param_dict)
        net = net_conf()
        net.load_model(pretrain_path, if_relax=True, if_print=True)
        if use_cuda:
            net = net.cuda()
        net.eval()
        self.net_work = net

    def eval_forward(self, im1, im2, *args):
        # === network output
        with torch.no_grad():
            input_dict = {'im1': im1, 'im2': im2, 'if_loss': False}
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            pred_flow = (flow_fw, flow_bw)
        return pred_flow

    def eval_save_result(self, save_name, predflow, *args, **kwargs):
        # you can save flow results here
        print(save_name)


def inference():
    pretrain_path = './scripts/upflow_kitti2015.pth'
    test_model = TestModel(pretrain_path=pretrain_path)
    data_path = '../data/ObjectTest/images'
    output_path = './results_upflow_object_test'
    # objects = ['cup']
    objects = ['cell_phone', 'pen', 'paper']
    for obj in objects:
        cur_object_output_path = os.path.join(output_path, obj)
        print(f"----> run inference on object test data, object: {obj}")
        os.makedirs(cur_object_output_path, exist_ok=True)
        images = glob.glob(os.path.join(data_path, obj, '*.png')) + glob.glob(os.path.join(data_path, obj, '*.jpg'))
        images = sorted(images)
        for im_file1, im_file2 in tqdm(zip(images[:-1], images[1:]), total=len(images[1:])):
            image1 = load_image(im_file1)
            image2 = load_image(im_file2)

            pred_flow = test_model.eval_forward(image1, image2)
            out_im_file = im_file1.replace(data_path, output_path)
            viz(image1, image2, pred_flow, out_im_file)


if __name__ == '__main__':
    inference()
