import os
import sys
import numpy as np
import h5py
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider
import bps_util


# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/test_files.txt'))


def convert_to_bps_h5(bps_nums, point_cloud_nums, bps_type='rect_grid', encode_method='sub'):
    # print(TRAIN_FILES)
    dir_name = os.path.join(BASE_DIR,
                            "../data/modelnet40_ply_hdf5_{}_{}_{}_{}".
                            format(str(point_cloud_nums), bps_type, bps_nums, encode_method))
    if os.path.exists(dir_name):
        return
    os.mkdir(dir_name)

    shutil.copyfile(os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/shape_names.txt'),
                    os.path.join(dir_name, 'shape_names.txt'))

    with open(os.path.join(dir_name, 'train_files.txt'), 'w') as train_files_list:
        for train_file in TRAIN_FILES:
            train_files_list.write(train_file.replace('modelnet40_ply_hdf5_2048', os.path.basename(dir_name)))
            train_files_list.write('\n')

    if bps_type == 'rect_grid':
        bps_data = bps_util.bps_rect_grid(bps_nums)
    elif bps_type == 'ball_grid':
        bps_data = bps_util.bps_rect_grid(bps_nums)
    elif bps_type == 'random_uniform_ball':
        bps_data = bps_util.bps_random_uniform_ball(bps_nums)
    elif bps_type == 'hcp':
        bps_data = bps_util.bps_hcp(bps_nums)
    else:
        assert 0 == 1, 'no such type yet.'

    for fn in range(len(TRAIN_FILES)):
        train_file = os.path.join(BASE_DIR, '..', TRAIN_FILES[fn])
        h5f = h5py.File(os.path.join(dir_name, os.path.basename(train_file)), 'w')
        current_data, current_label = provider.loadDataFile(train_file)
        current_data = current_data[:, :point_cloud_nums, :]
        # current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        encoded_data = []

        for idx in range(file_size):
            if encode_method == 'sub':
                bps_encode = bps_util.bps_kdt_sub_encode(bps_data, current_data[idx, :, :])
            elif encode_method == 'dis':
                bps_encode = bps_util.bps_kdt_dis_encode(bps_data, current_data[idx, :, :])
            else:
                assert 0 == 1, 'no such method yet.'
            encoded_data.append(bps_encode)

        encoded_data = np.array(encoded_data)
        # print(encoded_data.shape, current_label.shape)
        h5f['data'] = encoded_data
        h5f['label'] = current_label
        h5f.close()
        print("train file ", fn)

    with open(os.path.join(dir_name, 'test_files.txt'), 'w') as test_files_list:
        for test_file in TEST_FILES:
            test_files_list.write(test_file.replace('modelnet40_ply_hdf5_2048', os.path.basename(dir_name)))
            test_files_list.write('\n')

    for fn in range(len(TEST_FILES)):
        test_file = os.path.join(BASE_DIR, '..', TEST_FILES[fn])
        h5f = h5py.File(os.path.join(dir_name, os.path.basename(test_file)), 'w')
        current_data, current_label = provider.loadDataFile(test_file)
        current_data = current_data[:, :point_cloud_nums, :]
        # current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        encoded_data = []

        for idx in range(file_size):
            if encode_method == 'sub':
                bps_encode = bps_util.bps_kdt_sub_encode(bps_data, current_data[idx, :, :])
            elif encode_method == 'dis':
                bps_encode = bps_util.bps_kdt_dis_encode(bps_data, current_data[idx, :, :])
            else:
                assert 0 == 1, 'no such method yet.'
            encoded_data.append(bps_encode)

        encoded_data = np.array(encoded_data)
        # print(encoded_data.shape, current_label.shape)
        h5f['data'] = encoded_data
        h5f['label'] = current_label
        h5f.close()
        print("test file ", fn)



if __name__ == "__main__":
    convert_to_bps_h5(1024, 2048)

