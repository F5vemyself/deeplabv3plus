# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division

import multiprocessing
import os
import pickle
import torch
import pandas as pd
import cv2
from tqdm import trange
from skimage import io
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from datasets.transform import *

class COCODataset(Dataset):
    def __init__(self, dataset_name, cfg, period):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data','MSCOCO')
        self.dataset_dir = self.root_dir
        # data/MSCOCO/results/COCO2017/Segmentation
        self.rst_dir = os.path.join(self.root_dir, 'results', dataset_name, 'Segmentation')
        # data/MSCOCO/eval_result/COCO2017/Segmentation
        self.eval_dir = os.path.join(self.root_dir, 'eval_result', dataset_name, 'Segmentation')
        self.period = period
        self.year = self.__get_year()
        self.img_dir = os.path.join(self.dataset_dir, 'images','%s%s'%(self.period,self.year))
        self.ann_dir = os.path.join(self.dataset_dir, 'annotations/instances_%s%s.json'%(self.period,self.year))
        # coco-stuff数据集gt的位置
        self.seg_dir = os.path.join(self.dataset_dir, 'annotations')
        self.ids_file = os.path.join(self.dataset_dir, 'annotations/instances_%s%s_ids.mx'%(self.period,self.year))
        self.rescale = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.totensor = ToTensor()

        # 获取文件名,lsc add
        self.name_list = []

        self.voc2coco = [[0],
                         [5],
                         [2],
                         [16],
                         [9],
                         [44],#,46,86],
                         [6],
                         [3,8],
                         [17],
                         [62],
                         [21],
                         [67],
                         [18],
                         [19],#,24],
                         [4],
                         [1],
                         [64],
                         [20],
                         [63],
                         [7],
                         [72]]
        self.coco2voc = [0]*91
        for voc_idx in range(len(self.voc2coco)):
            for coco_idx in self.voc2coco[voc_idx]:
                self.coco2voc[coco_idx] = voc_idx
                
        self.coco = COCO(self.ann_dir)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
#       self.imgIds = self.coco.getImgIds()
        self.catIds = self.coco.getCatIds()
        from pycocotools import mask
        self.coco_mask = mask
        if os.path.exists(self.ids_file):
            with open(self.ids_file, 'rb') as f:
                self.imgIds = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.imgIds = self._preprocess(ids, self.ids_file)

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE)
        if self.period == 'train':        
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        self.cfg = cfg
    
    def __get_year(self):
        name = self.dataset_name
        if 'coco' in name:
            name = name.replace('coco','')
        else:
            name = name.replace('COCO','')
        year = name
        return year

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img_ann = self.coco.loadImgs(self.imgIds[idx])
        # /home/xxx/lsc/xxx/data/MSCOCO/images/val2017/000000352684.jpg
        name = os.path.join(self.img_dir, img_ann[0]['file_name'])
        # lsc add
        file_name = img_ann[0]['file_name']
        # 去掉.jpg字符
        file_name = file_name[:-4]
        # img_ann[0]['file_name']  xxx.jpg
        image = cv2.imread(name)
        # 新增一个属性用来保存文件的名字
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r,c,_ = image.shape
        # sample = {'image': image, 'name': name, 'row': r, 'col': c}
        sample = {'image': image, 'name': file_name, 'row': r, 'col': c}

        
        if self.period == 'train':

            annIds = self.coco.getAnnIds(imgIds=self.imgIds[idx])
            anns = self.coco.loadAnns(annIds)
            segmentation = np.zeros((r,c),dtype=np.uint8)
            for ann_item in anns:
                mask = self.coco.annToMask(ann_item)
                segmentation[mask>0] = self.coco2voc[ann_item['category_id']]
            if np.max(segmentation)>91:
                print(np.max(segmentation))
                raise ValueError('segmentation > 91')
            if np.max(segmentation)>20:
                print(np.max(segmentation))
                raise ValueError('segmentation > 20')
            sample['segmentation'] = segmentation

            if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)

        if self.cfg.DATA_RESCALE > 0:
            sample = self.rescale(sample)
        if 'segmentation' in sample.keys():
            sample['segmentation_onehot'] = onehot(sample['segmentation'], self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample
 
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3 | (m&64)>>1
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2 | (m&128)>>2
        cmap[:,:,2] = (m&4)<<5 | (m&32)<<1
        return cmap
    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if self.coco2voc[cat] != 0:
                c = self.coco2voc[cat]
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask


    def _preprocess(self, ids, ids_file):
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            if(mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.\
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        # print("rst_dir:",self.rst_dir)
        # data/MSCOCO/results/COCO2017/Segmentation
        # print("model_id:",model_id)
        i = 1
        folder_path = os.path.join(self.rst_dir, '%s_%s_cls' % (model_id, self.period))
        # print("folder_path:",folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s.png' % sample['name'])
            self.name_list.append(sample['name'])
            # predict_color = self.label2colormap(sample['predict'])
            # p = self.__coco2voc(sample['predict'])
            cv2.imwrite(file_path, sample['predict'])
            print('[%d/%d] %s saved' % (i, len(result_list), file_path))
            i += 1

    def do_python_eval(self, model_id):
        predict_folder = os.path.join(self.rst_dir, '%s_%s_cls' % (model_id, self.period))
        # seg_dir: data/MSCOCO/annotations
        # gt_folder：data/MSCOCO/annotations/stuff_val2017_pixelmaps
        gt_folder = os.path.join(self.seg_dir,"stuff_val2017_pixelmaps")
        TP = []
        P = []
        T = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            # 创建一个共享内存的值,'i' 表示整数 (integer) 类型。
            # 0：这是初始值，指定了共享值的初始值。
            # lock=True：这是一个布尔参数，如果设置为 True，则表示在访问共享值时要使用锁（lock），以确保多个进程不会同时修改它，从而避免竞争条件。
            TP.append(multiprocessing.Value('i', 0, lock=True))
            P.append(multiprocessing.Value('i', 0, lock=True))
            T.append(multiprocessing.Value('i', 0, lock=True))
        print("name_list:",self.name_list)
        def compare(start, step, TP, P, T):
            for idx in range(start, len(self.name_list), step):
                print('%d/%d' % (idx, len(self.name_list)))
                name = self.name_list[idx]
                # lsc add
                print("name:",name)
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                gt_file = os.path.join(gt_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
                gt = np.array(Image.open(gt_file))
                cal = gt < 255
                # predict 是一个包含模型的预测结果的 NumPy 数组。
                # gt 是一个包含真实标签的图像的 NumPy 数组。
                # predict == gt：这是一个逐元素的比较操作，返回一个布尔数组，其中对于每个像素，如果模型的预测结果等于真实标签，就为 True，否则为 False。这表示哪些像素被正确预测为相同的类别。
                # 比较模型的预测结果 (predict) 和真实标签 (gt)，并且只选择真实标签中非背景类别的像素
                mask = (predict == gt) * cal

                for i in range(self.cfg.MODEL_NUM_CLASSES):
                    P[i].acquire()
                    P[i].value += np.sum((predict == i) * cal)
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt == i) * cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt == i) * mask)
                    TP[i].release()

        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%11s:%7.3f%%' % ('backbound', IoU[i] * 100), end='\t')
            else:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100))

        miou = np.mean(np.array(IoU))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))

        # def do_python_eval(self, model_id):
