import pdb
import logging
import torchvision.transforms as transforms
#from transforms import VisualTransform, get_augmentation_transforms
import torch
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from hdf5_dataset import WMCA


def parse_data_list(data_list_path):
    csv = pd.read_csv(data_list_path, header=None)
    #data_type=csv.get(3) #get the tag whether data is train/dev/eval
    data_list = csv.get(0)
    face_labels = csv.get(1) 
    return data_list, face_labels


def get_dataset_from_list(data_list_path, dataset_cls, transform, num_frames=1000, rgb_dir='',cdit_dir=''):

    data_file_list, face_labels = parse_data_list(data_list_path)
    num_file = data_file_list.size
    dataset_list = []
    WMCA_file_ext = ".hdf5"
    for i in range(num_file):
        # 0 means real face and non-zero represents spoof
        face_label = int(face_labels.get(i))
        print("Accessing data :{}".format(i))
        file_path = data_file_list.get(i)
        rgb_path = os.path.join(rgb_dir, file_path+WMCA_file_ext)
        cdit_path = os.path.join(cdit_dir,file_path+WMCA_file_ext)
        #print("Accessing RGB path :{}".format(rgb_path))
        #print("Accessing CDIT path :{}".format(cdit_path))
        if not (os.path.exists(rgb_path) and os.path.exists(cdit_path)):
            logging.warning("Skip {} and {} (not exists)".format(rgb_path,cdit_path))
            continue
        else:
            dataset = WMCA(rgb_path,cdit_path, face_label,
                                  transform=transform, num_frames=num_frames)
            if len(dataset) == 0:
                logging.warning("Skip {} and {} (zero elements)".format(rgb_path,cdit_path))
                continue
            else:
                dataset_list.append(dataset)
        

    final_dataset = torch.utils.data.ConcatDataset(dataset_list)
    return final_dataset


def get_data_loader(config):
    batch_size = config.DATA.BATCH_SIZE
    num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS
    dataset_cls = zip_dataset.__dict__[config.DATA.DATASET]
    dataset_root_dir = config.DATA.ROOT_DIR
    dataset_subdir = config.DATA.SUB_DIR  # 'EXT0.2'
    face_dataset_dir = os.path.join(dataset_root_dir, dataset_subdir)
    num_frames = config.DATA.NUM_FRAMES

    assert config.DATA.TRAIN or config.DATA.TEST, "Please provide at least a data_list"

    test_data_transform = VisualTransform(config)
    if config.DATA.TEST:

        test_dataset = get_dataset_from_list(
            config.DATA.TEST, dataset_cls, test_data_transform, num_frames, root_dir=face_dataset_dir)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                       shuffle=False, pin_memory=True, drop_last=True)
        return test_data_loader
    else:
        assert config.DATA.TRAIN, "CONFIG.DATA.TRAIN should be provided"

        aug_transform = get_augmentation_transforms(config)
        train_data_transform = VisualTransform(config, aug_transform)

        train_dataset = get_dataset_from_list(
            config.DATA.TRAIN, dataset_cls, train_data_transform, num_frames, root_dir=face_dataset_dir)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, num_workers=num_workers,
                                                        shuffle=True, pin_memory=True, drop_last=True)

        assert config.DATA.VAL, "CONFIG.DATA.VAL should be provided"
        val_dataset = get_dataset_from_list(
            config.DATA.VAL, dataset_cls, test_data_transform, num_frames=1000, root_dir=face_dataset_dir)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=num_workers,
                                                      shuffle=False, pin_memory=True, drop_last=True)

        return train_data_loader, val_data_loader


if __name__ == '__main__':
    # batch_size = 4
    # num_workers = 2
    import importlib
    # face_dataset_dir = '/home/rizhao/data/FAS/all_public_datasets_zip/EXT0.0/'
    train_data_transform =  transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()])
    cdit_path = '/home/Dataset/FaceAntiSpoofing/WMCA/WMCA_preprocessed_CDIT/WMCA/face-station/12.02.18/100_03_015_2_10.hdf5'
    rgb_path = '/home/Dataset/FaceAntiSpoofing/WMCA/WMCA_preprocessed_RGB/WMCA/face-station/12.02.18/100_03_015_2_10.hdf5'
    dataset = importlib.import_module('hdf5_dataset').__dict__['WMCA']
    dataset = WMCA(rgb_path,cdit_path, 0,train_data_transform, None)
    final_dataset,dataset_list = get_dataset_from_list(
       "/home/hazeeq/FYP-Hazeeq/data/data_list/wmca-protocols-csv/PROTOCOL-grandtest_train.csv",dataset,train_data_transform,1000,"/home/Dataset/FaceAntiSpoofing/WMCA/WMCA_preprocessed_RGB/WMCA/","/home/Dataset/FaceAntiSpoofing/WMCA/WMCA_preprocessed_CDIT/WMCA/")
    import pdb; pdb.set_trace()

    # transform = transforms.Compose(
    #     [
    #         transforms.ToPILImage(),
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor()
    #     ]
    # )

    # dataset_cls = zip_dataset.ZipDatasetPixelFPN

    # test_dataset = get_dataset_from_list(
    #     'data_list/debug.csv', dataset_cls, transform, root_dir=face_dataset_dir)
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
    #                                                shuffle=False, pin_memory=True, drop_last=True)

    # data_iterator = iter(test_data_loader)

    # data = data_iterator.next()
    # import pdb
    # pdb.set_trace()
