import os,glob
import numpy as np
from paste import createOneImage
import argparse
import log_utils as log_utils
from datetime import datetime
import cv2
import json
from PIL import Image
from utils import harmonization, cp_real_test
def parse_args():
    parser = argparse.ArgumentParser(description='createOneSccene')
    parser.add_argument('--base_path', default=r'/data/iopen/fyw/datasets/CrowdCount/compos_experiments/CityUHK-X/')
    parser.add_argument('--save_path', default=r'/data/iopen/fyw/datasets/CrowdCount/compos_experiments/')
    parser.add_argument('--pedestrian_dir', default=r'/data/iopen/fyw/datasets/CrowdCount/compos_experiments/PedestriansFromLstn/street/')
    parser.add_argument('--real_dir', default=r'/data/iopen/fyw/datasets/CrowdCount/compos_experiments/random_partition_real/CityUHK-X_scene_034/')
    parser.add_argument('--pedestrian_info_dir', default='/data/iopen/fyw/datasets/CrowdCount/compos_experiments/PedestriansFromLstn/info_json.json',help='the json file contains information of pedestrians')
    parser.add_argument('--scene', default='scene_034', type=str, help='selected scene')
    parser.add_argument('--real_distribution', type=str, default='/data/iopen/fyw/code/CrowdCount/DM-Count-master/iteration_intermediate_result/scene_034/iteration_train/start_with_harmonized_CityUHK-X_scene_034_20_Lstn_1500_20_uniform_uniform/300/iteration_3_epoch_900',
                        help='distribution infomation from real desity map')
    parser.add_argument('--reward_distribution', type=bool, default=False,
                        help='whether composite test data')
    parser.add_argument('--num_pattern', default='uniform', type=str, help='uniform|normal')
    parser.add_argument('--pos_pattern', default='uniform', type=str, help='uniform|perspective')
    parser.add_argument('--ped_source', default='Lstn', type=str, help='cityspace|real')
    parser.add_argument('--ped_num', type=int, default=20,
                        help='the number of the pedestrias used')
    parser.add_argument('--min', type=int, default=0,
                        help='the min number of the pedestrias')
    parser.add_argument('--max', type=int, default=200,
                        help='the max number of the pedestrias')
    parser.add_argument('--test', type=bool, default=True,
                        help='whether composite test data')
    parser.add_argument('--debug', type=bool, default=False,
                        help='whether composite test data')
    parser.add_argument('--harmonized', type=bool, default=True,
                        help='whether composite test data')
    parser.add_argument('--save_file_list', type=bool, default=True,
                        help='whether composite test data')
    parser.add_argument('--train_num', type=int, default=32,
                        help='the number for training')
    parser.add_argument('--test_num', type=int, default=20,
                        help='the number for test')
    parser.add_argument('--max_pedestrian_area', type=int, default=1163,
                        help='scene_area/max_ped_area')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    scene_base_path = os.path.join(args.base_path, '*','images', args.scene)
    if args.test:
        stage = ['train', 'test']
        dataset_name = '_'.join(['CityUHK-X', args.scene, str(args.ped_num),
            args.ped_source, str(args.train_num), str(args.test_num), str(args.max), str(args.min), args.num_pattern, args.pos_pattern])
    else:
        stage = ['train']
        dataset_name = '_'.join(['CityUHK-X', args.scene, str(args.ped_num),
            args.ped_source, str(args.train_num), str(args.max), str(args.min), args.num_pattern, args.pos_pattern])
    if args.debug:
        dataset_name = dataset_name + 'debug'
        
    if args.reward_distribution:
        dataset_name = dataset_name.replace(args.num_pattern + "_" + args.pos_pattern, "reward_distribution")
    elif args.real_distribution:
        dataset_name = dataset_name.replace(args.num_pattern + "_" + args.pos_pattern, "real_distribution")
        
    
    save_base_path = os.path.join(args.save_path, args.scene, dataset_name)
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    logger = log_utils.get_logger(os.path.join(save_base_path, 'train-{:s}.log'.format(time_str)))
    log_utils.print_config(vars(args), logger)
    
    p = glob.glob(scene_base_path)[0]
    scene_path = os.path.join(p, "scene.jpg")
    base_image = cv2.cvtColor(cv2.imread(scene_path), cv2.COLOR_BGR2RGB)
    
    with open(args.pedestrian_info_dir, "r") as f:
        pesestrians_info = json.load(f)
        
    pedestrians = {}
    pedestrians_path = glob.glob(os.path.join(args.pedestrian_dir, '*.png'))
    for pp in pedestrians_path:
        base_name = os.path.basename(pp)
        id = base_name.split(".")[0]
        pedestrians[id] = cv2.cvtColor(cv2.imread(pp), cv2.COLOR_BGR2RGB)
        
    min = args.min
    max = args.max
    
    probability_map = None
    if args.real_distribution:
        avg_density_map_path = os.path.join(args.real_distribution, "averageDensityMap.npy")
        avg_density_map = np.load(avg_density_map_path)
        rate_h = base_image.shape[0] // avg_density_map.shape[0]
        rate_w = base_image.shape[1] // avg_density_map.shape[1]
        new_h = round(rate_h * avg_density_map.shape[0])
        new_w = round(rate_w * avg_density_map.shape[1])
        avg_density_map = cv2.resize(avg_density_map, (new_w, new_h), cv2.INTER_CUBIC)
        map_sum = np.sum(avg_density_map)
        probability_map = avg_density_map/map_sum
        
    for method in stage:
        if method == 'train':
            pictures_num = args.train_num
            tmp_data = method + '_data'
        else:
            pictures_num = args.test_num
            tmp_data = 'fake_' + method + '_data'
                
        image_txt_pairs = []
        images = []
        gts = []
        masks = []
        for i in range(pictures_num):
            if args.real_distribution:
                nums_file_path = os.path.join(args.real_distribution, 'num.txt')
                nums = np.loadtxt(nums_file_path)
                mean = np.mean(nums)
                sigma = np.std(nums)
                totoal_pedestrians_num  = round(np.clip(np.random.normal(mean, sigma, 1), 0, None)[0])
                if args.reward_distribution:
                    totoal_pedestrians_num = np.random.randint(min, max+1)
            elif args.num_pattern == 'uniform':
                totoal_pedestrians_num = np.random.randint(min, max+1)
            elif args.num_pattern == 'normal':
                mean = round((args.max + args.min)/2)
                sigma = (mean-args.min)/3#根据3sigma原则
                totoal_pedestrians_num = round(np.clip(np.random.normal(mean, sigma, 1),args.min,args.max)[0])
            
            save_images_path = os.path.join(save_base_path, tmp_data, 'images')
            if not os.path.exists(save_images_path):
                os.makedirs(save_images_path)
            save_ground_truth_path = save_images_path.replace("images", "ground_truth_txt")
            if not os.path.exists(save_ground_truth_path):
                os.makedirs(save_ground_truth_path)
            save_mask_path = save_images_path.replace("images", "image_masks")
            if not os.path.exists(save_mask_path):
                os.makedirs(save_mask_path)
                
            image_file_name = os.path.join(save_images_path, str(i))
            
            image, gt, mask = createOneImage(args, base_image.copy(), totoal_pedestrians_num, pesestrians_info, pedestrians, probability_map)
            gt = np.array(gt)
            if len(gt.shape)==1:
                if gt.shape[0]==2:
                    gt = np.expand_dims(gt,0)
                elif gt.shape[0]==0:
                    gt = np.empty([0, 2])
            
            images.append(Image.fromarray(image))
            gts.append(gt)
            masks.append(Image.fromarray(mask).convert("1"))
            
            ground_truth_path = image_file_name.replace("images", "ground_truth_txt")
            image_txt_pairs.append([image_file_name + ".jpg", ground_truth_path + ".txt"])
            
        
        if args.harmonized:
            images, gts = harmonization(images, gts, masks)# rgb numpy
        
        
        for i in range(len(image_txt_pairs)):
            np.savetxt(image_txt_pairs[i][1], gts[i], fmt="%d")
            cv2.imwrite(image_txt_pairs[i][0], cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
            masks[i].save(image_txt_pairs[i][0].replace("images", "image_masks"))
        image_txt_pairs = list(map(lambda x:[x[0].replace(save_base_path + '/',""), x[1].replace(save_base_path + '/',"")], image_txt_pairs))
        image_txt_pairs = np.array(image_txt_pairs)
    
        if args.save_file_list:
            np.savetxt(os.path.join(save_base_path,tmp_data + '.list'),image_txt_pairs,fmt="%s")
    cp_real_test(args.scene, args.real_dir, save_base_path, )