import os
import numpy as np
import torch


# 根据测试数据集得到需要生成的位姿txt文件
def get_render_pose_matrix_from_gigiadata(cams_txt_path, images_path):
    image_names = os.listdir(images_path)
    cams_names = os.listdir(cams_txt_path)
    image_names_without_type = []

    for name in image_names:
        image_names_without_type.append(name.split(".")[0])

    identity_pose_names = []
    used_to_name_result_images = []

    # 得到准备生成的目标位姿
    for name in cams_names:
        if name.split("_")[0] not in image_names_without_type:
            identity_pose_names.append(name)
            used_to_name_result_images.append(name.split("_")[0])

    print("get identity_pose_names size: ", len(identity_pose_names))

    # 用于保存所有目标位姿的列表
    total_identity_poses_list = []

    # 循环目标位姿list得到对应的外参矩阵
    for pname in identity_pose_names:
        # 得到每个cams.txt文件对应的路径
        cam_path = os.path.join(cams_txt_path, pname)
        # print("cam_path: ", cam_path)
        # 打开文件并读取2-5行
        with open(cam_path) as file:
            lines = file.readlines()[1:5]
            # 从行中提取数字，保存为二维矩阵
            matrix = [list(map(float, line.strip().split())) for line in lines]
        # print("matrix: ", matrix)
        total_identity_poses_list.append(matrix)

    # print("torch.tensor(total_identity_poses_list):",
    #       torch.tensor(total_identity_poses_list))
    identity_pose_tensor = torch.tensor(total_identity_poses_list)
    print("tensor shape: ", identity_pose_tensor.shape)
    return identity_pose_tensor, used_to_name_result_images


if __name__ == "__main__":
    cams_path = "/ssd3/wyy/projects/_competition/nerf-pytorch/giga_test_data/PeonyGarden/cams"
    images_path = "/ssd3/wyy/projects/_competition/nerf-pytorch/giga_test_data/PeonyGarden/images"
    get_render_pose_matrix_from_gigiadata(cams_path, images_path)
