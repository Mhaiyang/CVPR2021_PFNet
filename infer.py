"""
 @Time    : 2021/7/6 14:36
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : infer.py
 @Function: Inference
 
"""
import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

from config import *
from misc import *
from PFNet import PFNet

torch.manual_seed(2021)
device_ids = [1]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
exp_name = 'PFNet'
args = {
    'scale': 416,
    'save_results': True
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
                       ('CHAMELEON', chameleon_path),
                       ('CAMO', camo_path),
                       ('COD10K', cod10k_path),
                       ('NC4K', nc4k_path)
                       ])

results = OrderedDict()

def main():
    net = PFNet(backbone_path).cuda(device_ids[0])

    net.load_state_dict(torch.load('PFNet.pth'))
    print('Load {} succeed!'.format('PFNet.pth'))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                _, _, _, prediction = net(img_var)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(os.path.join(results_path, exp_name, name, img_name + '.png'))
            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()
