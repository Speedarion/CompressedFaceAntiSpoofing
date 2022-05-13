import torch
import numpy as np
import h5py
from torchvision import transforms

""""
class WMCA(torch.utils.data.Dataset):
    
        #Also work for CASIA-SURF
    
    def __init__(self, rgb_path,cdit_path, face_label, transform, num_frames=1000):
        self.rgb_path = rgb_path
        self.cdit_path = cdit_path 
        self.rgb_data = None
        self.cdit_data = None
        self.transform = transform
        self.face_label = face_label
    
        #self.rgb_dataset = h5py.File(self.rgb_path, 'r')
        #self.cdit_dataset=h5py.File(self.cdit_path, 'r')

        #frame = self.hdf5_dataset['Frame_0']['array'][1,:,:]
        with h5py.File(self.rgb_path,'r') as file:
            self.len = len(file.keys())
            #print([key for key in file.keys()])
            #print(len(file.keys()))
        #with h5py.File(self.cdit_path,'r') as file:
            #self.len = len(file.keys())
            #print([key for key in file.keys()])
            #print(len(file.keys()))
        #if len(self.cdit_dataset.keys())==len(self.rgb_dataset.keys()):
        #    self.len = len(self.rgb_dataset.keys())

        #print(self.hdf5_dataset['Frame_0']['array'].shape)
        #print(frame.shape)



    def __getitem__(self, index):
        key = "Frame_{}".format(index)
        if self.rgb_data is None:
            self.rgb_dataset = h5py.File(self.rgb_path, 'r')
        if self.cdit_data is None:
            self.cdit_dataset=h5py.File(self.cdit_path, 'r')
        #print(self.rgb_path)
        try:
            rgb_frame = self.rgb_dataset[key]['array'][:] #get rgb frame - 0,1,2 (3 channels)
            cdit_frame = self.cdit_dataset[key]['array'][1,:,:] #get depth frame -  1 (channel number 1)
        except KeyError: #some frames are missing 
            return None
        #print("rgb frame shape is {}".format(rgb_frame.shape))
        #print("cdit frame shape is {}".format(cdit_frame.shape))
        #apply transformations
        rgb_tensor = self.transform(rgb_frame)
        d_tensor = self.transform(cdit_frame)
        #print("rgb tensor size is {}".format(rgb_tensor.size()))
        #print("depth tensor size is {}".format(d_tensor.size()))
        rgbd_frame = torch.cat((rgb_tensor,d_tensor),dim=0) #join rgb and depth frame to form (4,224,224) frame
        target = self.face_label
        
        #print("rgbd tensor shape is {}".format(rgbd_frame.size()))

        return index, rgbd_frame, target

    def __len__(self):
        return self.len

    """

class WMCA(torch.utils.data.Dataset):
    """
        Also work for CASIA-SURF
    """
    def __init__(self, rgb_path,cdit_path, face_label, transform, num_frames=1000):
        self.rgb_path = rgb_path
        self.cdit_path = cdit_path 
        self.rgb_data = None
        self.cdit_data = None
        self.transform = transform
        self.face_label = face_label
    
        with h5py.File(self.rgb_path,'r') as file:
            self.rgb_dataset_keys = [key for key in file.keys()]
        with h5py.File(self.cdit_path,'r') as file:
            self.cdit_dataset_keys = [key for key in file.keys()]
        
        # Match keys; Filter out keys of missing frame        
        self.mutual_key_list = [i for i in self.rgb_dataset_keys and self.cdit_dataset_keys if i in self.rgb_dataset_keys and self.cdit_dataset_keys]
        self.len=len(self.mutual_key_list)


    def __getitem__(self, index):
        key = self.mutual_key_list[index]
        if self.rgb_data is None:
            self.rgb_dataset = h5py.File(self.rgb_path, 'r')
        if self.cdit_data is None:
            self.cdit_dataset=h5py.File(self.cdit_path, 'r')
        #How to view ->self.rgb_dataset['Frame_0']['array'][and so on]
        # in cdit_dataset index 0 , it has grayscale which has 1 channel o
        rgb_frame = self.rgb_dataset[key]['array'][:] #get rgb frame - 0,1,2 (3 channels)
        d_frame = self.cdit_dataset[key]['array'][1,:,:] #get depth frame -  1 (channel number 1)
        i_frame = self.cdit_dataset[key]['array'][2,:,:] #get infrared frame -  2 (channel number 2)
        t_frame = self.cdit_dataset[key]['array'][3,:,:] #get thermal frame -  3 (channel number 3)


        #apply transformations
        rgb_tensor = self.transform(rgb_frame)
        d_tensor = self.transform(d_frame)
        i_tensor = self.transform(i_frame)
        t_tensor = self.transform(t_frame)        

        rgbdit_frame = torch.cat((rgb_tensor,d_tensor,i_tensor,t_tensor),dim=0) #join rgb and depth frame to form (4,224,224) frame
        target = self.face_label
   
        #print("rgb frame shape is {}".format(rgb_frame.shape))
        #print("cdit frame shape is {}".format(cdit_frame.shape))
        #print("rgb tensor size is {}".format(rgb_tensor.size()))
        #print("depth tensor size is {}".format(d_tensor.size()))
        #print("rgbd tensor shape is {}".format(rgbd_frame.size()))

        return index, rgbdit_frame, target

    def __len__(self):
        return self.len

class _3DMAD(torch.utils.data.Dataset):
    def __init__(self, h5_path, face_label, transform, num_frames=1000):
        self.h5_path = h5_path
        self.transform = transform
        self.face_label = face_label

        # Keys: ['Color_Data', 'Depth_Data', 'Eye_Pos']>
        self.h5_dataset = h5py.File(self.h5_path, 'r')

        len_dataset = self.h5_dataset['Eye_Pos'].shape[0]
        self.index_list = list(range(len_dataset))
        if len(self.index_list) > num_frames:
            sample_indices = np.linspace(0, len(self.index_list) - 1, num=num_frames, dtype=int)
            self.index_list = [self.index_list[id] for id in sample_indices]

        self.len = len(self.index_list)


    def __getitem__(self, index):

        index = self.index_list[index]

        color_data = self.h5_dataset['Color_Data'][index] # (3,480,640), uint8
        # depth_data = self.h5_dataset['Depth_Data'][index] # (1,480,640), uint8

        im = color_data.transpose((2,0,1)) # (480,640,3)
        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'face_label': self.face_label
        }


        return index, tensor, target, self.zip_file_path

    def __len__(self):
        return self.len
 

class CSMAD(torch.utils.data.Dataset):
    def __init__(self, h5_path, face_label, transform, num_frames=1000):
        self.h5_path = h5_path
        self.transform = transform
        self.face_label = face_label

        # Keys: ['Color_Data', 'Depth_Data', 'Eye_Pos']>
        self.h5_dataset = h5py.File(self.h5_path, 'r')

        self.key_list = list(self.h5_dataset['data']['sr300']['infrared'].keys())


        if len(self.key_list) > num_frames:
            sample_indices = np.linspace(0, len(self.key_list) - 1, num=num_frames, dtype=int)
            self.key_list = [self.key_list[id] for id in sample_indices]

        self.len = len(self.key_list)


    def __getitem__(self, index):

        key = self.key_list[index]

        # ir_data = self.h5_dataseta['data']['seek_compact']['infrared']['09_35_58_990']
        # depth_data = self.h5_dataset['Depth_Data'][index] # (1,480,640), uint8
        #ir_data = self.h5_dataset['data']['sr300']['infrared'][key][:] # (640,830) uint16
        #depth_data = self.h5_dataset['data']['sr300']['depth'][key][:] # (640,830) uint16
        color_data = self.h5_dataset['data']['sr300']['color'][key][:] # (640,830, 3) unit8
        im = color_data

        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'face_label': self.face_label
        }
        return index, tensor, target, self.zip_file_path

    def __len__(self):
        return self.len
if __name__ == '__main__':
    train_data_transform =  transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()])
    cdit_path = '/home/Dataset/FaceAntiSpoofing/WMCA/WMCA_preprocessed_CDIT/WMCA/face-station/12.02.18/100_03_015_2_10.hdf5'
    rgb_path = '/home/Dataset/FaceAntiSpoofing/WMCA/WMCA_preprocessed_RGB/WMCA/face-station/12.02.18/100_03_015_2_10.hdf5'
    dataset = WMCA(rgb_path,cdit_path, 1,train_data_transform, None)
    #dataset_3dmad = _3DMAD('/home/Dataset/FaceAntiSpoofing/3DMAD/session01/Data/04_01_05.hdf5', 0, None)
    #dataset_csmad = CSMAD('/home/Dataset/FaceAntiSpoofing/CSMAD/attack/STAND/A/Mask_atk_A1_i0_001.h5', 0, None)
    import pdb; pdb.set_trace()

    
