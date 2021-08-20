"""
pip install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; python setup.py install
cd mmdetection; mkdir checkpoints
wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
"""

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import matplotlib.pyplot as plt

# file path
config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# image
img = '/content/mmdetection/demo/demo.jpg'
img_arr  = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(12, 12))
# plt.imshow(img_arr)

# detect
results = inference_detector(model, img)
show_result_pyplot(model, img, results)
