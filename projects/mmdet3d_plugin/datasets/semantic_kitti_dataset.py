import os
from os import path as osp
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module()
class SemanticKittiDataset(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        img_size=[370, 1220],
        temporal = [],
        eval_range = 51.2,
        depthmodel="msnet3d",
        labels_tag = 'labels',
        color_jitter=None,
        scale=2
    ):
        super().__init__()
        
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.depth_query = "msnet3d"
        self.depthmodel = depthmodel
        self.eval_range = eval_range
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split 
        self.sequences = splits[split]
        
        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m
        self.scale = scale

        self.img_W = img_size[1]
        self.img_H = img_size[0]

        self.poses=self.load_poses()
        self.target_frames = temporal
        self.load_scans()
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.beta_dist = torch.distributions.beta.Beta(0.75, 0.75)
        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )

            for voxel_path in glob.glob(glob_path):

                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path
                    }
                )

    def set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        example = self.get_data_info(index)
        #print(example.keys())

        #######################

        calib = example["img_metas"]["cam_intrinsic"][0]
        fu = calib[0, 0]
        fv = calib[1, 1]
        cu = calib[0, 2]
        cv = calib[1, 2]

        random_mix_flag = False
        random = np.random.random()
        # print("random:", random)
        if random < 0.3:
            random_mix_flag = True
        if random_mix_flag == True and self.test_mode == False:
            #print("succeed")
            count_num = 0
            while count_num < 50:
                count_num += 1
                random_index = np.random.randint(len(self.scans))
                temp = self.get_data_info(random_index)
                calib_temp = temp["img_metas"]["cam_intrinsic"][0]
                #print(temp_p)
                temp_fu = calib_temp[0, 0]
                temp_fv = calib_temp[1, 1]
                temp_cu = calib_temp[0, 2]
                temp_cv = calib_temp[1, 2]
                if temp_fu  == fu and temp_fv == fv and temp_cu == cu and temp_cv == cv:
                    # target = example["target"]
                    # target_temp = temp["target"]
                    # alpha = self.beta_dist.sample().item()
                    # alpha = max(alpha, 1 - alpha)
                    alpha = 0.5
                    blended_tensor = example["img"] * alpha + temp["img"] * (1 - alpha)
                    example["img"] = blended_tensor
                    example_target = example["target"]
                    example['img_metas']['target_1_1_label'] = example_target.copy()
                    #print(example_target.shape)

                    #
                    # percentage = (num_ones / total_elements) * 100
                    # print("percentage",percentage)
                    example_target[example_target == 255] = 20
                    #print(example_target.shape)
                    example_target = torch.nn.functional.one_hot(torch.tensor(example_target, dtype=torch.long), num_classes=21)
                    #print(example_target.shape)
                    # example_target=example_target.permute(3, 0, 1, 2).float()
                    example_target=example_target.float()
                    #example_target = example_target * (~ignored_mask)
                    example_target = example_target[:, :,:,:20]

                    temp_target = temp["target"]
                    example['img_metas']['target_1_1_temp_label'] = temp_target.copy()
                    temp_target[temp_target == 255] = 20
                    temp_target = torch.nn.functional.one_hot(torch.tensor(temp_target, dtype=torch.long), num_classes=21)
                    # temp_target=temp_target.permute(3, 0, 1, 2).float()
                    # temp_mask =  torch.tensor(temp_mask).unsqueeze(0)  # [1,256,256,32]
                    temp_target=temp_target.float()
                    #temp_mask =  torch.tensor(temp_mask).unsqueeze(-1)  # [1,256,256,32]
                    #temp_target = temp_target * (~ignored_mask)
                    temp_target = temp_target[:, :,:,:20]
                    blended_target = example_target * alpha + temp_target * (1 - alpha)

                    #combined_mask = example_mask | temp_mask
                    #blended_target = blended_target * (~combined_mask)
                    example["target"] = blended_target

                    example_target_2 = example['img_metas']['target_1_2'].copy()
                    example_target_2[example_target_2 == 255] = 20
                    # print(example_target.shape)
                    example_target_2 = torch.nn.functional.one_hot(torch.tensor(example_target_2, dtype=torch.long),
                                                                 num_classes=21)
                    # print(example_target.shape)
                    # example_target=example_target.permute(3, 0, 1, 2).float()
                    example_target_2 = example_target_2.float()
                    # example_target = example_target * (~ignored_mask)
                    example_target_2 = example_target_2[:, :, :, :20]

                    temp_target_2 = temp['img_metas']['target_1_2'].copy()
                    example['img_metas']['target_1_2_temp_label'] = temp_target_2.copy()
                    temp_target_2[temp_target_2 == 255] = 20
                    temp_target_2 = torch.nn.functional.one_hot(torch.tensor(temp_target_2, dtype=torch.long),
                                                              num_classes=21)
                    # temp_target=temp_target.permute(3, 0, 1, 2).float()
                    # temp_mask =  torch.tensor(temp_mask).unsqueeze(0)  # [1,256,256,32]
                    temp_target_2 = temp_target_2.float()
                    #temp_mask_2 = torch.tensor(temp_mask_2).unsqueeze(-1)  # [1,256,256,32]
                    # temp_target = temp_target * (~ignored_mask)
                    temp_target_2 = temp_target_2[:, :, :, :20]
                    blended_target_2 = example_target_2 * alpha + temp_target_2 * (1 - alpha)

                    #combined_mask_2 = example_mask_2 | temp_mask_2
                    #blended_target_2 = blended_target_2 * (~combined_mask_2)
                    example['img_metas']['target_1_2'] = blended_target_2
                    #####################################################
                    lidar = example['img_metas']['lidar'].copy()
                    lidar_temp = temp['img_metas']['lidar'].copy()
                    example['img_metas']['lidar'] = np.concatenate((lidar, lidar_temp), axis=0)
                    #####################################################
                    projected_pix = example['img_metas']['projected_pix'].copy()
                    projected_pix_temp = temp['img_metas']['projected_pix'].copy()

                    #(projected_pix_temp[0].shape)
                    example['img_metas']['projected_pix'] = projected_pix+projected_pix_temp

                    fov_mask = example['img_metas']['fov_mask'].copy()
                    fov_mask_temp = temp['img_metas']['fov_mask'].copy()

                    # (projected_pix_temp[0].shape)
                    example['img_metas']['fov_mask'] = fov_mask + fov_mask_temp
                    #print("fov_mask shape:", fov_mask_temp.shape)


                    #print(example["target"].shape)
                    break
        #if len(example["target"].shape) != 4 and self.test_mode == False:
        if len(example["target"].shape) != 4:
            example_target = example["target"]
            example['img_metas']['target_1_1_label'] = example_target.copy()
            # print(example_target.shape)
            example_target[example_target == 255] = 20
            # print(example_target.shape)
            example_target = torch.nn.functional.one_hot(torch.tensor(example_target, dtype=torch.long), num_classes=21)
            # print(example_target.shape)
            # example_target = example_target.permute(3, 0, 1, 2).float()
            example_target = example_target.float()

            # example_target = example_target * (~ignored_mask)
            example_target = example_target[:, :, :, :20]

            #example_target = example_target * (~example_mask)
            example["target"] = example_target
                    #[1,3,170,1220]
                    # print(example["img"].shape)
                    # print(temp["img"].shape)
                    #[256,256,32]
                    #[256,256,32]
            example_target_2 = example['img_metas']['target_1_2'].copy()
            #example['img_metas']['target_1_2_label'] = example_target_2
            example_target_2[example_target_2 == 255] = 20
            # print(example_target.shape)
            example_target_2 = torch.nn.functional.one_hot(torch.tensor(example_target_2, dtype=torch.long),
                                                           num_classes=21)
            # print(example_target.shape)
            # example_target=example_target.permute(3, 0, 1, 2).float()
            example_target_2 = example_target_2.float()
            # example_target = example_target * (~ignored_mask)
            example_target_2 = example_target_2[:, :, :, :20]
            #example_target_2 = example_target_2 * (~example_mask_2)
            example['img_metas']['target_1_2'] = example_target_2



        ############################
        data_queue.insert(0, example)

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'] for each in queue]
        metas_map = {}
        
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        scan = self.scans[index]

        voxel_path = scan["voxel_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]
        #print(frame_id)
        meta_dict = self.get_meta_info(scan, sequence, frame_id)
        img,masks = self.get_input_info(sequence, frame_id)
        target = self.get_gt_info(sequence, frame_id)
        #print(target.type)

        data_info = dict(
            img_metas = meta_dict,
            img = img,
            masks = masks,
            target = target
        )
        return data_info

    def vox2pix(self, cam_E, cam_k, img_W=None, img_H=None, voxel_size=None, vox_origin=None, scene_size=None):
        """
        compute the 2D projection of voxels centroids
        
        Parameters:
        ----------
        cam_E: 4x4
        =camera pose in case of NYUv2 dataset
        =Transformation from camera to lidar coordinate in case of SemKITTI
        cam_k: 3x3
            camera intrinsics
        vox_origin: (3,)
            lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
        img_W: int
            image width
        img_H: int
            image height
        scene_size: (3,)
            scene size in meter: (51.2, 51.2, 6.4) for SemKITTI
        
        Returns
        -------
        projected_pix: (N, 2)
            Projected 2D positions of voxels
        fov_mask: (N,)
            Voxels mask indice voxels inside image's FOV 
        pix_z: (N,)
            Voxels'distance to the sensor in meter
        """
        img_W, img_H, voxel_size, vox_origin, scene_size = self.img_W, self.img_H, \
            self.scale*self.voxel_size, self.vox_origin, self.scene_size
        # Compute the x, y, z bounding of the scene in meter
        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels centroids in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        #print('vol_dim:',vol_dim)
        #(128,128,16)
        xv, yv, zv = np.meshgrid(
                range(vol_dim[0]),
                range(vol_dim[1]),
                range(vol_dim[2]),
                indexing='ij'
            )
        vox_coords = np.concatenate([
                xv.reshape(1,-1),
                yv.reshape(1,-1),
                zv.reshape(1,-1)
            ], axis=0).astype(int).T

        # Project voxels'centroid from lidar coordinates to camera coordinates
        cam_pts = vox2world(vox_origin, vox_coords, voxel_size)
        cam_pts = rigid_transform(cam_pts, cam_E)

        # Project camera coordinates to pixel positions
        projected_pix = cam2pix(cam_pts, cam_k)
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

        # Eliminate pixels outside view frustum
        pix_z = cam_pts[:, 2]
        fov_mask = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < img_W,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < img_H,
                    pix_z > 0))))


        return projected_pix, fov_mask

    def get_meta_info(self, scan, sequence, frame_id):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )

        # for multiple images
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        image_paths = []

        projected_pixs = []
        fov_masks = []

        pts_list = []

        # transform points from lidar to camera coordinate
        lidar2cam_rt = scan["T_velo_2_cam"]
        # camera intrisic
        P = scan["P"]
        cam_k = P[0:3, 0:3]
        #print(cam_k)
        intrinsic = cam_k
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = (viewpad @ lidar2cam_rt)

        pts_filename = os.path.join(
                self.data_root, "dataset", f"sequences_{self.depthmodel}_lidar/sequences", sequence, frame_id + ".bin"
            )
        pts = np.fromfile(pts_filename, dtype=np.float32)
        pts = pts.reshape((-1, 4))
        pts = pts[:, :3]
        #print("pts shape:", pts.shape)
        lidar2img_rts.append(lidar2img_rt)
        lidar2cam_rts.append(lidar2cam_rt)
        cam_intrinsics.append(intrinsic)
        image_paths.append(rgb_path)

        pix, mask = self.vox2pix(lidar2cam_rt, intrinsic)
        #print("PTS shape:", pts.shape)
        #(453620,3) not constant
        #print("pix shape:",pix.shape)
        #(262144,2) 128x128x16
        #print("mask shape:", mask.shape)
        #(262144,)
        projected_pixs.append(pix)
        fov_masks.append(mask)

        pts_list.append(pts)

        # for reference img
        seq_len = len(self.poses[sequence])
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )

            pose_list = self.poses[sequence]

            ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            target = pose_list[int(target_id)]
            ref2target = np.matmul(inv(target), ref) # both for lidar

            target2cam = scan["T_velo_2_cam"] # lidar to camera
            ref2cam = target2cam @ ref2target

            lidar2cam_rt  = ref2cam
            lidar2img_rt = (viewpad @ lidar2cam_rt)

            pts_filename = os.path.join(
                self.data_root, "dataset", f"sequences_{self.depthmodel}_lidar/sequences", sequence, target_id + ".bin"
            )
            pts = np.fromfile(pts_filename, dtype=np.float32)
            pts = pts.reshape((-1, 4))
            pts = np.matmul(np.matmul(inv(ref), target), pts.T).T
            pts = pts[:, :3]

            lidar2img_rts.append(lidar2img_rt)
            lidar2cam_rts.append(lidar2cam_rt)
            cam_intrinsics.append(intrinsic)
            image_paths.append(rgb_path)

            pix, mask = self.vox2pix(lidar2cam_rt, intrinsic)

            projected_pixs.append(pix)
            fov_masks.append(mask)

            pts_list.append(pts)

        # load ground truth
        if self.split == 'train' or self.split == 'val':
            target_1_2_path = os.path.join(self.label_root, sequence, frame_id + "_1_2.npy")
            target_1_2 = np.load(target_1_2_path)
            target_1_2 = target_1_2.reshape(-1)
            target_1_2 = target_1_2.reshape(128, 128, 16)
            target_1_2 = target_1_2.astype(np.float32)
        else:
            target_1_2 = None

        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            lidar=np.concatenate(pts_list),
            target_1_2=target_1_2,
            target_1_2_label=target_1_2,
            target_1_2_temp_label=target_1_2,
            target_1_1_label=None,
            target_1_1_temp_label=None,
            projected_pix=projected_pixs,
            fov_mask=fov_masks, 
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
            img_shape = [(self.img_H,self.img_W)],
        )

        return meta_dict

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.poses[sequence])
        image_list = []
        masks_list = []
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )


        img = Image.open(rgb_path).convert("RGB")
        masks = []
        for class_id in range(20):
            mask_path = os.path.join(
                self.data_root, "dataset", "sequences_ray_marching_masks", sequence, frame_id, f"{class_id}.npy"
            )
            #print(mask_path)
            if os.path.exists(mask_path):
                mask_array = np.load(mask_path)
                mask_array = mask_array[:self.img_H, :self.img_W]
                masks.append(mask_array)
        #print(masks)
        if len(masks) > 0:
            masks = np.stack(masks, axis=0)
            masks = torch.tensor(masks, dtype=torch.float32)
            masks_list.append(masks)
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        image_list.append(self.normalize_rgb(img))
        ########################
        #######################
        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]  # crop image

            image_list.append(self.normalize_rgb(img))

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]
        if len(masks_list)>0:
            masks_tensor = torch.stack(masks_list,dim=0).unsqueeze(2)  #[1,19,370,1220]
            #print(masks_tensor.shape)
        else:
            masks_tensor = torch.ones(1,20,1,1,370,1220)
        #print("masks_tensor shape:",masks_tensor.shape)
        return image_tensor, masks_tensor

    def get_gt_info(self, sequence, frame_id):
        """Get the ground truth.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            array: target. 
        """
        if self.split == "train" or self.split == "val":
            # load full-range groundtruth
            target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target = np.load(target_1_path)
            #print("ddddddddddd")
            # short-range groundtruth
            if self.eval_range == 25.6:
                target[128:, :, :] = 255
                target[:, :64, :] = 255
                target[:, 192:, :] = 255

            elif self.eval_range == 12.8:
                target[64:, :, :] = 255
                target[:, :96, :] = 255
                target[:, 160:, :] = 255
        else:
            target = np.ones((256,256,32))
        #print('target',target.shape)
        return target
    ###################
    # def semantic_ray_marching(self):
    #     scan = self.scans[0]
    #     P = scan["P"]
    #     cam_e = scan["T_velo_2_cam"]
    #
    #     cam_k = P[0:3, 0:3]
    #     intrinsic = cam_k
    #     image_tensor, _ = self.get_input_info('00000',0)
    #     height = image_tensor[2]
    #     width = image_tensor[3]
    #     N = image_tensor[0]
    #     y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    #     pixel_coords = np.stack([y_coords, x_coords], axis=-1)  # [height, width, 2]
    #     pixel_coords = pixel_coords.repeat(N, 1, 1, 1)
    #     image_coords = pixel_coords.view(1, N, height * width, 2)
    #     ray_directions = pix2cam(image_coords,intrinsic,cam_e)
    #     voxel_indices = []
    #
    #     for ray in np.nditer(ray_directions, flags=['external_loop'], order='F'):
    #         while True:
    #             voxel_index = np.floor((pos - vox_origin) / vox_size).astype(int)
    #             if np.any(voxel_index < 0) or np.any(voxel_index >= vol_dims):
    #             voxel_indices.append(voxel_index)
    #     voxel_indices = np.array(voxel_indices)
        ####################

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        if results is None:
            logger.info('Skip Evaluation')

        if 'ssc_scores' in results:
            # for single-GPU inference
            ssc_scores = results['ssc_scores']
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            # for multi-GPU inference
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])

            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])

            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)

            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }

        for name, iou in zip(self.class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou

        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)

        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']

        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)

        return eval_results


def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        offsets = np.array(offsets).astype(np.float32)
        cam_pts = vol_origin[None] + vox_size*(vox_coords + offsets[None])

        return cam_pts


def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates."""
    intr = intr.astype(np.float32)
    #intr_shape = intr.shape
    #print("intr_shape",intr_shape)
    #intr_shape"(3,3)
    pix = np.dot(intr, cam_pts.T).T
    #cam_pts_shape= cam_pts.shape
    #(262144,3) 128x128x16
    #print("cam_pts_shape:",cam_pts_shape)

    pix = pix[:, :2] / np.maximum(pix[:, 2:3], np.ones_like(pix[:, 2:3])*1e-6)

    #print("pix:", np.round(pix).astype(np.int64))
    #np.savetxt('projec.txt', np.round(pix).astype(np.int64), fmt='%f')
    return np.round(pix).astype(np.int64)
#######################################################
def pix2cam(pixel_coords, intr, cam):
    """Convert camera coordinates to pixel coordinates."""
    pixel_coords = pixel_coords.squeeze()
    ones = np.ones((pixel_coords.shape[0], 1))
    homogeneous_pixel_coords = np.concatenate([pixel_coords, ones], axis=-1)

    #print("homogeneous_pixel_coords:",homogeneous_pixel_coords) #homogeneous_pixel_coords: (451400, 3)
    intr_inv = np.linalg.inv(intr)
    #ray_directions = np.einsum('bijm,bijm->bijm', intr_inv, homogeneous_pixel_coords)
    ray_directions = np.dot(intr_inv, homogeneous_pixel_coords.T).T

    #homogeneous_ray_directions = np.hstack((ray_directions, np.ones((ray_directions.shape[0], 1))))
    cam_inv = np.linalg.inv(cam)[:3, :3]
    #cam_cord = cam_inv[:3, 3]
    # Transform camera coordinates to world coordinates
    #ray_directions_world = np.dot(homogeneous_ray_directions, cam_inv.T)
    ray_directions_world = np.dot(ray_directions, cam_inv.T)
    #ray_directions_world=ray_directions_world[:, :3]
    #ray_directions_world = ray_directions_world - cam_cord
    ray_directions_world /= np.linalg.norm(ray_directions_world, axis=1, keepdims=True)  # Normalize the directions

    # Get the camera position in the world coordinates
    #cam_position_world = cam_inv[:3, 3]
    #print("cam ",cam_position_world)
    # Replicate the camera position for all rays
    #ray_directions_world = cam_position_world + ray_directions_world
    #print('ray_direction_world shape:',ray_directions_world.shape)
    #print('ray_direction_world', ray_directions_world)

    return ray_directions_world
#######################################################

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]

def check_occupaied(voxel_index, voxel_grid):
    x, y, z = voxel_index
    if 0 <= x < voxel_grid.shape[0] and 0 <= y < voxel_grid.shape[1] and 0 <= z < voxel_grid.shape[2]:
        return voxel_grid[x, y, z]
    else:
        return 0
