from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

# import ipdb
import matplotlib
import collections
from tqdm import tqdm
import torch
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
# from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667


# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
#         gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
#         if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16(n_fg_class=2)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    
    if opt.istrain:
        
        if opt.load_path:
            pretrained_dict = torch.load(opt.load_path)
            pretrained_dict = pretrained_dict['model']
            model_dict = trainer.state_dict()
            load_dict = collections.OrderedDict()
    #         lastlayer = [ 'head.cls_loc.weight', 'head.cls_loc.bias',
    #                      'head.score.weight', 'head.score.bias']
            lastlayer = ['head.classifier.2.weight', 'head.classifier.2.bias', 'head.cls_loc.weight', 'head.cls_loc.bias',
                         'head.score.weight', 'head.score.bias']

            for k, v in pretrained_dict.items():
                key = 'faster_rcnn.' + k
                if key in model_dict and k not in lastlayer:
                    load_dict[key] = v
            model_dict.update(load_dict)

            trainer.load_state_dict(model_dict)
            print('load pretrained model from %s' % opt.load_path)
            lr_ = opt.lr
        
        
        
        for epoch in range(opt.epoch):
            trainer.reset_meters()
            for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                trainer.train_step(img, bbox, label, scale)
            lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
            best_path = trainer.save(best_map='ep'+str(epoch))
            eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
            print(eval_result)
    else:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        print(eval_result)
        
#     if eval_result['map'] > best_map:
#         best_map = eval_result['map']
#         best_path = trainer.save(best_map=best_map)
#     if epoch == 9:
#         trainer.load(best_path)
#         trainer.faster_rcnn.scale_lr(opt.lr_decay)
#         lr_ = lr_ * opt.lr_decay

#     if epoch == 13:
#         return


if __name__ == '__main__':
    import fire

    fire.Fire()
