import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from Config import *
from tqdm import tqdm

class Texture:
    def __init__(self, rgbPath, disparityPath, kinectPath, resultPath, encoderPath) -> None:
        self.rgbPath = rgbPath
        self.disparityPath = disparityPath
        self.version = disparityPath[-2:]
        

        self.__load_kinect(kinectPath)
        self.__load_result(resultPath)
        self.__load_encoder(encoderPath)
        

    def __load_kinect(self, kinectPath):
        with np.load(kinectPath) as data:
            self.disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
            self.rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    def __load_result(self, resultPath):
        with np.load(resultPath) as data:
            self.occupancy_map = data['oc_map']
            self.poses = data['poses']
            self.map_ranges = data['map_ranges']
            self.trajectory = data['trajectory']
            self.grid_scale = data['grid_scale']

    def __load_encoder(self, encoderPath):
        with np.load(encoderPath) as data:
            self.position_timestamps = data['time_stamps']

    def __load_disparity_and_rgb(self, ind_dis, ind_rgb):
        dis_file = os.path.join(self.disparityPath, "disparity%s_%d.png"%(self.version, ind_dis))
        rgb_file = os.path.join(self.rgbPath, "rgb%s_%d.png"%(self.version, ind_rgb))

        return np.array(Image.open(dis_file)), np.array(Image.open(rgb_file))
    
    @staticmethod
    def cal_pixel_loc(disparity):
        shape = disparity.shape
        dd = -0.00304*disparity + 3.31
        depth = 1.03/dd
        # depth = depth.reshape([shape[None]])
        i_indexes = np.arange(0, shape[0])[:, None]
        i_indexes = np.tile(i_indexes, [1, shape[1]])
        
        rgbi = (526.37 * i_indexes + (-4.5 * 1750.46) * dd + 19276.0)/585.051
        j_indexes = np.arange(1, shape[1]+1)[None, :]
        j_indexes = np.tile(j_indexes, [shape[0], 1])
        # print(i_indexes)
        rgbj = (526.37 * j_indexes + 16662)/585.051

        # rgbi = rgbi.reshape([shape[None]])
        # rgbj = rgbj.reshape([shape[None]])

        rgb_depth = np.concatenate([rgbi[None], rgbj[None], depth[None]], axis=0)
        rgb_depth = rgb_depth.reshape([3, -1])
        rgb_depth = rgb_depth[:, np.where((rgb_depth[0] < shape[0]-0.5) & (rgb_depth[0] >= 0))[0]]
        rgb_depth = rgb_depth[:, np.where((rgb_depth[1] < shape[1]-0.5) & (rgb_depth[1] >= 0))[0]]
        # rgb_depth = rgb_depth[:, np.where(())]

        return rgb_depth
    
    @staticmethod
    def cam_to_world(rgb_depth):
        pic_coor = rgb_depth[[0, 1]]
        z_line = np.ones([1, pic_coor.shape[-1]])
        pic_coor = np.concatenate([pic_coor, z_line], axis=0)
        
        cam_coor = K_inv.dot(pic_coor)*rgb_depth[-1][None]
        regular_coor = rRo.dot(cam_coor)
        body_coor = bRr.dot(regular_coor)
        # print(cam_coor[2], rgb_depth[2])
        # print(rgb_depth[2])
        # print(np.min(body_coor[0]),np.min(body_coor[1]),np.min(body_coor[2]))
        # print(np.max(body_coor[0]),np.max(body_coor[1]),np.max(body_coor[2]))
        # print(np.min(regular_coor[0]),np.min(regular_coor[1]),np.min(regular_coor[2]))
        # print(np.max(regular_coor[0]),np.max(regular_coor[1]),np.max(regular_coor[2]))
        # print(body_coor.shape)
        valid_range = np.where((body_coor[2] < 0.015) & (body_coor[2] >-0.015))[0]
        # print(valid_range.shape)
        valid_coor = body_coor[:, valid_range]
        pixel_coor = np.round(rgb_depth[:2, valid_range]).astype(np.int16)

        # print()

        return valid_coor, pixel_coor

    def getTexture(self):
        position_index = 0
        rgb_index = 0
        self.textureMap = np.zeros([self.occupancy_map.shape[0], self.occupancy_map.shape[1], 3], dtype=np.uint8)

        T_position = self.position_timestamps.size
        T_rgb = self.rgb_stamps.size
        T_disp = self.disp_stamps.size
        for i in tqdm(range(T_disp)):
            while(position_index + 1 < T_position and self.position_timestamps[position_index+1] <= self.disp_stamps[i]):
                position_index += 1

            while(rgb_index + 1 < T_rgb and self.rgb_stamps[rgb_index+1] <= self.disp_stamps[i]):
                rgb_index += 1

            disparity, rgb = self.__load_disparity_and_rgb(i+1, rgb_index+1)
            rgb_depth = self.cal_pixel_loc(disparity)
            valid_coor, pixel_coor = self.cam_to_world(rgb_depth)
            coor_world = self.poses[:, :, position_index].dot(valid_coor) + self.trajectory[:, position_index][:, None]
            # print(np.max(valid_coor[0]), np.max(valid_coor[1]))
            # print(self.trajectory[:, position_index], position_index, self.map_ranges)
            # print('trajectory range', np.min(self.trajectory[:2]), np.max(self.trajectory[:2]), self.map_ranges)
            # print(np.max(coor_world[0]), np.max(coor_world[1]))
            map_coor = np.round(coor_world[:2]/self.grid_scale).astype(np.int16) \
                    - np.array([self.map_ranges[0, 0], self.map_ranges[1, 0]])[:, None]

            map_coor = map_coor[:, np.where((map_coor[0] >= 0) 
                                            & (map_coor[0] <= self.textureMap.shape[0]-1)
                                            & (map_coor[1] >= 0)
                                            & (map_coor[1] <= self.textureMap.shape[1]-1))[0]]
            pixel_coor = pixel_coor[:, np.where((map_coor[0] >= 0) 
                                            & (map_coor[0] <= self.textureMap.shape[0]-1)
                                            & (map_coor[1] >= 0)
                                            & (map_coor[1] <= self.textureMap.shape[1]-1))[0]]
            
            self.textureMap[map_coor[0], map_coor[1]] = rgb[pixel_coor[0], pixel_coor[1]]


# def getTexture(textureMap, position_timestamps, rgb_stamps, disp_stamps):

#     T_position = position_timestamps.size
#     T_rgb = rgb_stamps.size
#     T_disp = disp_stamps.size
#     position_index = 0
#     rgb_index = 0
#     for i in tqdm(range(T_disp)):
#         while(position_index + 1 < T_position and position_timestamps[position_index+1] <= disp_stamps[i]):
#             position_index += 1

#         while(rgb_index + 1 < T_rgb and rgb_stamps[rgb_index+1] <= disp_stamps[i]):
#             rgb_index += 1

#         disparity, rgb = self.__load_disparity_and_rgb(i+1, rgb_index+1)
#         rgb_depth = self.cal_pixel_loc(disparity)
#         valid_coor, pixel_coor = self.cam_to_world(rgb_depth)
#         coor_world = self.poses[:, :, position_index].dot(valid_coor) + self.trajectory[:, position_index][:, None]
#         # print(np.max(valid_coor[0]), np.max(valid_coor[1]))
#         # print(self.trajectory[:, position_index], position_index, self.map_ranges)
#         # print('trajectory range', np.min(self.trajectory[:2]), np.max(self.trajectory[:2]), self.map_ranges)
#         # print(np.max(coor_world[0]), np.max(coor_world[1]))
#         map_coor = np.round(coor_world[:2]/self.grid_scale).astype(np.int16) \
#                 - np.array([self.map_ranges[0, 0], self.map_ranges[1, 0]])[:, None]

#         map_coor = map_coor[:, np.where((map_coor[0] >= 0) 
#                                         & (map_coor[0] <= self.textureMap.shape[0]-1)
#                                         & (map_coor[1] >= 0)
#                                         & (map_coor[1] <= self.textureMap.shape[1]-1))[0]]
#         pixel_coor = pixel_coor[:, np.where((map_coor[0] >= 0) 
#                                         & (map_coor[0] <= self.textureMap.shape[0]-1)
#                                         & (map_coor[1] >= 0)
#                                         & (map_coor[1] <= self.textureMap.shape[1]-1))[0]]
        
#         self.textureMap[map_coor[0], map_coor[1]] = rgb[pixel_coor[0], pixel_coor[1]]


if __name__ == "__main__":
    filePath = 'F:\grad\quarter2\ECE 276A\projects\ECE276A_PR2\data\dataRGBD\RGB20\\rgb20_1500.png'
    
    filePath = 'F:\grad\quarter2\ECE 276A\projects\ECE276A_PR2\data\dataRGBD\Disparity20\\disparity20_1500.png'
    img = Image.open(filePath)
    img_arr = np.array(img)
    print(img_arr.shape)
    # img_arr[:30, :40, :] = 0
    # plt.imshow(img_arr)
    # plt.show()
    # input()
    rgb_depth = Texture.cal_pixel_loc(img_arr)
    print(rgb_depth.shape)
    Texture.cam_to_world(rgb_depth)


    dataset = 20
    n_particles = 50
    rgbPath = './data/dataRGBD/RGB%d'%(dataset)
    disparityPath = './data/dataRGBD/Disparity%d'%(dataset)
    kinectPath = "./data/Kinect%d.npz" % dataset
    resultPath = "./results/d%d_N%d.npz"%(dataset, n_particles)
    encoderPath = "./data/Encoders%d.npz" % dataset

    texture = Texture(rgbPath, disparityPath, kinectPath, resultPath, encoderPath)
    texture.getTexture()

    plt.imshow(texture.textureMap)
    plt.show()
    input()
