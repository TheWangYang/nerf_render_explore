import cv2
import os

before_dir = "/ssd3/wyy/projects/_competition/nerf-pytorch/giga_need_results/20230413_4downsample_results/ScienceSquare/"
after_dir = "/ssd3/wyy/projects/_competition/nerf-pytorch/giga_submissions/20230419_2and4downsample_submissions/ScienceSquare/"
os.makedirs(after_dir, exist_ok=True)
image_names = sorted(os.listdir(before_dir))

for name in image_names:
    before_path = os.path.join(before_dir, name)
    after_path = os.path.join(after_dir, name)
    img = cv2.imread(before_path)  # 读取数据
    print("before image_name: {}, shape: {}".format(name, img.shape))
    img = cv2.resize(img, (9568, 6376), interpolation=cv2.INTER_CUBIC)
    print("after image_name: {}, shape: {}".format(name, img.shape))
    cv2.imwrite(after_path, img)
