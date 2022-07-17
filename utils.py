import torch
from harmonization_model.util import util as har_util
import glob, os
import shutil
import numpy as np
from torchvision import transforms
from harmonization_model.models.networks import RainNet
from harmonization_model.models.normalize import RAIN
def load_har_model():
    net = RainNet(input_nc=3, 
        output_nc=3, 
        ngf=64, 
        norm_layer=RAIN, 
        use_dropout=True)

    load_path = os.path.join("pretrained_model", 'net_G_last.pth')
    assert os.path.exists(load_path), print('%s not exists. Please check the file'%(load_path))
    print(f'loading the model from {load_path}')
    state_dict = torch.load(load_path, map_location='cpu')
    har_util.copy_state_dict(net.state_dict(), state_dict)
    # net.load_state_dict(state_dict)
    return net

har_transform_image = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

har_transform_mask = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor()
])   
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cuda")

def harmonization(images, gts, masks):
    har_model = load_har_model()
    har_model = har_model.to(device)
    batch_size = 16
    har_images = []
    start = 0
    end = min(batch_size, len(images))
    while end <= len(images) and start != end:
        batch_images = images[start:end]
        batch_masks = masks[start:end]
        batch_images_tensor = [har_transform_image(img).unsqueeze(0).to(device) for img in batch_images]
        batch_masks_tensor = [har_transform_mask(mask).unsqueeze(0).to(device) for mask in batch_masks]
        
        batch_images_tensor = torch.cat(batch_images_tensor)
        batch_masks_tensor = torch.cat(batch_masks_tensor)
        batch_har_images_tensor = har_model.processImage(batch_images_tensor, batch_masks_tensor, batch_images_tensor)
        #(batch_size, 3,512,512)
        batch_har_images = [har_util.tensor2im(batch_har_images_tensor[i].unsqueeze(0)) for i in range(batch_har_images_tensor.shape[0])]
        har_images.extend(batch_har_images)
        # del batch_images_tensor, batch_masks_tensor, batch_har_images_tensor, batch_har_images
        # torch.cuda.empty_cache()
        start = end
        end = min(end + batch_size, len(images))
    for i in range(len(gts)):
        gts[i][:,1] = gts[i][:,1]*(512/384)
    return har_images, gts

def cp_real_test(scene, real_dir, tar_dir):
    real_image_path = glob.glob(os.path.join(real_dir, 'test_data', 'images', scene))[0]
    real_gt_path = real_image_path.replace('images', 'ground_truth_txt')
    real_images = glob.glob(os.path.join(real_image_path, "*.jpg"))
    real_txts = glob.glob(os.path.join(real_gt_path, "*.txt"))
    if os.path.join(real_image_path, "scene.jpg") in real_images:
        real_images.remove(os.path.join(real_image_path, "scene.jpg"))
    if os.path.join(real_gt_path, "statistic.txt") in real_txts:
        real_txts.remove(os.path.join(real_gt_path, "statistic.txt"))
    assert len(real_images) == len(real_txts)
    
    tar_image_path = os.path.join(tar_dir, "real_test_data", "images")
    if not os.path.exists(tar_image_path):
        os.makedirs(tar_image_path)
    tar_gt_path = os.path.join(tar_dir, "real_test_data", "ground_truth_txt")
    if not os.path.exists(tar_gt_path):
        os.makedirs(tar_gt_path)
    
    image_txt_pairs = []
    for i in range(len(real_images)):
        real_image = real_images[i]
        real_txt = real_txts[i]
        image_base_name = os.path.basename(real_image)
        txt_base_name = os.path.basename(real_txt)
        image_txt_pairs.append([os.path.join(tar_image_path.replace(tar_dir + '/',""), image_base_name), \
            os.path.join(tar_gt_path.replace(tar_dir + '/',""), txt_base_name)])
        
        shutil.copy(real_image, tar_image_path)
        shutil.copy(real_txt, tar_gt_path)
    image_txt_pairs = np.array(image_txt_pairs)
    np.savetxt(os.path.join(tar_dir, "real_test_data.list"), image_txt_pairs, fmt="%s")
    
    
        
    
    