# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

def test_net():
	# 1449张
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'val')
	# 182
	dataloader = DataLoader(dataset,
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)
	
	net = generate_net(cfg)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)

	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
	net.load_state_dict(model_dict)
	
	net.eval()
	# result_list：[{'name':'2007_000033','predict':[[...],[...]]}]
	result_list = []
	with torch.no_grad():
		# sample_batched={dict:10},
		# image:(8,3,512,512),name:(8,1),row:(8,1),col:(8,1)
		# image_0.5:(8,3,256,256)
		# image_0.75:(8,3,384,384)
		# image_1.0:(8,3,512,512)
		# image_1.25:(8,3,640,640)
		# image_1.5:(8,3,768,768)
		# image_1.75:(8,3,896,896)

		for i_batch, sample_batched in enumerate(dataloader):
			name_batched = sample_batched['name']
			# 提取图像的宽row、高col
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			# （8，3，512，512）
			[batch, channel, height, width] = sample_batched['image'].size()
			# （8，21，512，512）
			# 张量的形状表示了对于当前批次中的每张图像，模型在每个像素位置上对每个类别的预测结果。
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(1)
			# cfg.TEST_MULTISCALE：[0.5,0.75,1.0,1.25,1.5,1.75]
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				# 使用神经网络 net 对 inputs_batched 进行前向传播，得到预测结果 predicts。
				# 这里的 .to(1) 是将预测结果移动到 GPU 上进行计算。
				predicts = net(inputs_batched).to(1)
				# 使用 clone() 创建 predicts_batched，这样可以保留原始预测结果 predicts 的副本，以便稍后的操作。
				predicts_batched = predicts.clone()
				del predicts
				# 它会进行水平翻转数据增强。
				if cfg.TEST_FLIP:
					# rate=1.75时
					# (8,3,896,896)
					# 水平翻转
					inputs_batched_flip = torch.flip(inputs_batched,[3]) 
					# 前向传播
					predicts_flip = torch.flip(net(inputs_batched_flip),[3]).to(1)
					# （8，21，896，896）
					predicts_batched_flip = predicts_flip.clone()
					del predicts_flip
					# 进行平均操作
					predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0
				# 使用 F.interpolate 对 multi_avg 进行上采样，以将预测结果还原到原始图像尺寸，并将结果累积到 multi_avg 中。
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched

			# 这是因为之前的操作将不同尺度的预测结果累加在一起，现在需要对其进行平均以得到最终的多尺度平均预测结果。
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			# 使用 torch.argmax 函数在通道维度上获取每个像素位置的类别预测。
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)

			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
			#	max_edge = max(row,col)
			#	rate = cfg.DATA_RESCALE / max_edge
			#	new_row = row*rate
			#	new_col = col*rate
			#	s_row = (cfg.DATA_RESCALE-new_row)//2
			#	s_col = (cfg.DATA_RESCALE-new_col)//2
	 
			#	p = predicts_batched[i, s_row:s_row+new_row, s_col:s_col+new_col]
				p = result[i,:,:]
				p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				result_list.append({'predict':p, 'name':name_batched[i]})

			print('%d/%d'%(i_batch,len(dataloader)))
	dataset.save_result(result_list, cfg.MODEL_NAME)
	dataset.do_python_eval(cfg.MODEL_NAME)
	print('Test finished')

if __name__ == '__main__':
	test_net()


