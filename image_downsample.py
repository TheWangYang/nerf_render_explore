import cv2
import os
from PIL import Image


# 设置的对图片进行下采样的脚本，加快训练速度和防止显存溢出
images_path = '/ssd3/wyy/projects/_competition/nerf-pytorch/data/gigaAI_data/Museum/images/'  # 原图路径
output_dir = '/ssd3/wyy/projects/_competition/nerf-pytorch/data/gigaAI_data/Museum/images_4/'  # resize后路径

factor = 4  # 降采样倍数

images_list = os.listdir(images_path)
img = Image.open(images_path + images_list[0])
(W, H) = (img.width, img.height)  # [W,H] for normal subdata set
# (W, H) = (img.height, img.width)  # [W, H] for the old gate
print("image_size : ", (W, H))

for image_name in images_list:
    img = cv2.imread(images_path+image_name)
    img_resize = cv2.resize(img, (int(W/factor), int(H/factor)))
    cv2.imwrite(output_dir + image_name, img_resize)
    print(image_name, " done")
print("all images done")
