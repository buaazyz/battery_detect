import os
import xml.etree.ElementTree as ET

import numpy as np

from util import read_image


class BatBboxDataset:
    """Bounding box dataset for X Ray of battery

    The index corresponds to each image.

    When queried by an index, this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.


    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='total',
                 ):

        id_list_file = os.path.join(data_dir, '{0}.txt'.format(split))

        
        self.ids = [id_.split(' ')[1] for id_ in open(id_list_file)]
        # print(self.ids)
        self.data_dir = data_dir
        # self.label_names = VOC_BBOX_LABEL_NAMES
        self.label = Battery_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        annodir =  os.path.join(self.data_dir, 'Dataset/Annotation', id_[:-4] + '.txt')
        bbox = list()
        label = list()

        with open(annodir,encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                bbox.append(line.split(' ')[2:6])
                try:
                    label.append(self.label.index(line.split(' ')[1]))
                except:
                    # print(line.split(' ')[1])
                    label.append(2)
                    

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)


        # Load a image
        img_file = os.path.join(self.data_dir, 'Dataset/Image', id_)
        img = read_image(img_file, color=True)


        return img, bbox, label

    __getitem__ = get_example



Battery_BBOX_LABEL_NAMES = (
    '带电芯充电宝',
    '不带电芯充电宝',
    '未指定类别'
)

# bb = BatBboxDataset(data_dir='D:/PycharmProjects/slim-faster-torch/data/')
# for i in range(5500):
#     bb.get_example(i)
