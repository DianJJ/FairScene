# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jianbiao Mei
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, builder
from mmdet.models import build_detector
from projects.mmdet3d_plugin.fairescene.utils.header import Header, SparseHeader, SegNet
from projects.mmdet3d_plugin.fairescene.modules.sgb import SGB
from projects.mmdet3d_plugin.fairescene.modules.sdb import SDB
from projects.mmdet3d_plugin.fairescene.modules.flosp import FLoSP
from projects.mmdet3d_plugin.fairescene.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.fairescene.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss
from mmdet.apis import init_detector#, inference_detector
from mmdet3d.models import build_model
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import time
from PIL import Image

def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        #cfg.test_pipeline[0].type = 'LoadImageFromWebcam'

    #cfg.test_pipeline = replace_ImageToTensor(cfg.test_pipeline)
    test_pipeline = Compose(cfg.test_pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results

@HEADS.register_module()
class FairSceneHead(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
        scale_2d_list,
        pts_header_dict,
        # seg_header_dict,
        depth=3,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = True,
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.embed_dims = embed_dims

        if kwargs.get('dataset', 'semantickitti') == 'semantickitti':
            self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                                "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
            self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        elif kwargs.get('dataset', 'semantickitti') == 'kitti360':
            self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object']
            self.class_weights = torch.from_numpy(np.array([0.464, 0.595, 0.865, 0.871, 0.717, 0.657, 0.852, 0.541, 0.602, 0.567, 0.607, 0.540, 0.636, 0.513, 0.564, 0.701, 0.774, 0.580, 0.690]))
        self.n_classes = len(self.class_names)

        ##########################
        self.intra_block_a = nn.Sequential(
            nn.Conv2d(128, embed_dims * 2, kernel_size=3,  padding = 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        # self.intra_block_b = nn.Sequential(
        #     nn.Conv2d(128, embed_dims * 2, kernel_size=3, padding=1, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True),
        # )
        # self.intra_block_c = nn.Sequential(
        #     nn.Conv2d(128, embed_dims * 2, kernel_size=3, padding=1, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True),
        # )
        # self.intra_block_d = nn.Sequential(
        #     nn.Conv2d(128, embed_dims * 2, kernel_size=3, padding=1, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True),
        # )

        # self.intra_block_2 = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims * 2, kernel_size=3, padding = 1, groups=20),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=1, groups = 20),
        #     nn.ReLU(inplace=True),
        # )
        # self.seg_inter = SegNet(in_channels=320, num_classes=20, groups=20)
        # # # self.seg_inter = nn.Sequential(
        # # #     nn.Conv2d(embed_dims, embed_dims * 2, kernel_size=3, padding=1, groups=20),
        # # #     nn.ReLU(inplace=True),
        # # #     nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1, groups=20),
        # # #     nn.ReLU(inplace=True),
        # # #     nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1, groups=20),
        # # #     nn.ReLU(inplace=True),
        # # #     nn.Conv2d(embed_dims // 2, 20, kernel_size=1, groups=20),
        # # #     nn.Sigmoid()
        # # # )
        # self.classifiers = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1, groups=20),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims // 2, 20, kernel_size=1, groups=20),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Sigmoid()
        # )
        #self.num_classes = 20
        self.num_classes = 19
        ###########################

        self.flosp = FLoSP(scale_2d_list)
        ################group#####################
        #self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)
        self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, groups=self.num_classes)
        #################group####################
        self.sgb = SGB(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims)
        #self.sgb = SGB(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims, groups = 20)
        ###############group#################################
        self.mlp_prior = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.LayerNorm(self.embed_dims//2),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dims//2, self.embed_dims)
        )
        self.mask_embed = nn.Embedding(1, self.embed_dims)
        occ_channel = 16 if pts_header_dict.get('guidance', True) else 0  #8 channel occ guidance
        #occ_channel = 12 if pts_header_dict.get('guidance', True) else 0  # 8 channel occ guidance
        #self.sdb = SDB(channel=self.embed_dims + occ_channel, out_channel=self.embed_dims // 2, depth=depth)
        self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=(self.embed_dims+occ_channel)//2, depth=depth, groups=self.num_classes+1)
        
        # self.occ_header = nn.Sequential(
        #     SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=1),
        #     nn.Conv3d(self.embed_dims//2, 1, kernel_size=3, padding=1)
        # )
        self.sem_header = SparseHeader(self.n_classes, feature=self.embed_dims)
        self.ssc_header = Header(self.n_classes, feature=((self.embed_dims+occ_channel) // 2))
        #self.ssc_header = Header(self.n_classes, feature=((self.embed_dims)//2))
        #self.ssc_header = Header(self.n_classes, feature=16)
        self.pts_header = builder.build_head(pts_header_dict)
        ##############################modified 6/13 add segmentation head mask rcnn
        # seg_cfg = Config.fromfile(seg_header_dict['config_path'])
        # self.seg_header = build_detector(seg_cfg.model,
        #                   test_cfg = seg_cfg.get('test_cfg'))  # or device='cuda:0'
        # self.seg_header = init_detector(seg_cfg, checkpoint=seg_header_dict['checkpoint_path'], device='cuda:0')
        #self.seg_header = builder.build_head(seg_cfg.model)  # or device='cuda:0'
        # checkpoint = load_checkpoint(self.seg_header, seg_header_dict['checkpoint_path'], map_location='cuda:0')
        # for param in self.seg_header.parameters():
        #     param.requires_grad = False
        ##############################
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
        
    def forward(self, img,  masks, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """

        B = masks.shape[0]
        N = masks.shape[1]
        H = masks.shape[4]
        W = masks.shape[5]
        masks = masks.view(1,B * N, H, W)
        masks = F.interpolate(masks, size=(93, 305), mode='bilinear', align_corners=False)
        masks = torch.flip(masks, dims=[3])
        #masks shape [19,1,370,1220]
        #img shape [1,3,370,1220] B * N, C, H, W
        dtype = mlvl_feats[0].dtype
        #################################
        # with torch.no_grad():
        #     img = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
        #     #pdb.set_trace()
        #     img_seg = inference_detector(self.seg_header, img)
        #     #img_seg = self.seg_header(return_loss=False, rescale=False, img=[img])[0]
        ########################################
        #img_seg = self.seg_header(img,img_metas)
        out = {}
        ##########################
        # # start_time = time.time()
        # # start_time_intra_block = time.time()
        intra_a = self.intra_block_a(mlvl_feats[0].flatten(0, 1))
        intra_b = self.intra_block_a(mlvl_feats[1].flatten(0, 1))
        intra_c = self.intra_block_a(mlvl_feats[2].flatten(0, 1))
        intra_d = self.intra_block_a(mlvl_feats[3].flatten(0, 1))
        intra = [intra_a, intra_b, intra_c, intra_d]

        #trainable_params = sum(p.numel() for p in self.intra_block.parameters())
        #self.intra_block : 246720
        #sum(p.numel() for p in self.mlp_prior.parameters())
        #411200
        #trainable_params = sum(p.numel() for p in self.sgb.parameters())
        #54289280
        ##pdb.set_trace()
        # ####################
        # target_size = intra_a.shape[-2:]
        #
        # #
        # intra_b_resized = F.interpolate(intra_b, size=target_size, mode='bilinear', align_corners=False)
        # intra_c_resized = F.interpolate(intra_c, size=target_size, mode='bilinear', align_corners=False)
        # intra_d_resized = F.interpolate(intra_d, size=target_size, mode='bilinear', align_corners=False)
        #
        # #
        # intra_combined = intra_a + intra_b_resized + intra_c_resized + intra_d_resized
        # #################################
        # seg_out = self.seg_inter(intra_combined)
        # #92820
        #
        # # end_time_seg_inter = time.time()
        # # elapsed_time_seg_inter = end_time_seg_inter- start_time_seg_inter
        # #(1,20,93,305)
        # # start_time_classifiers = time.time()
        # class_out = self.classifiers(intra_combined)
        # #trainable_params = sum(p.numel() for p in self.classifiers.parameters())
        # #92820
        # #pdb.set_trace()
        # # end_time_classifiers = time.time()
        # # elapsed_time_classifiers = end_time_classifiers- start_time_classifiers
        #
        # #intra = self.intra_block_2(intra)
        # class_out = class_out.view(intra_combined.size(0), self.num_classes)
        # #(1,20,20)
        # class_gt = (masks.sum(dim=(2, 3)) > 0).float()
        # #pdb.set_trace()
        #############################
        # seg_mid = seg_out.squeeze(0).cpu().detach()
        #
        # #
        # seg_mid = (seg_mid.numpy()*255).astype(np.uint8)
        # mask_mid = masks.squeeze(0).cpu().detach()
        #
        # #
        # mask_mid = (mask_mid.numpy()*255).astype(np.uint8)

        # def denormalize(tensor, mean, std):
        #     for t, m, s in zip(tensor, mean, std):
        #         t.mul_(s).add_(m)  # t = t * s + m
        #     return tensor
        #
        #
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        #
        # denormalized_tensor = denormalize(img[0].cpu().clone(), mean, std)
        #
        # denormalized_array = denormalized_tensor.permute(1, 2, 0).numpy()
        # denormalized_array = (denormalized_array * 255).astype(np.uint8)
        # img_mid = Image.fromarray(denormalized_array)
        # for i in range(seg_mid.shape[0]):
        #     seg = Image.fromarray(seg)
        #     seg.save(f'seg_class_{i}.jpg')
        #
        #     msk_mid = Image.fromarray(msk_mid)
        #     msk_mid.save(f'mask_class_{i}.jpg')
        #
        #     img_mid.save(f'img.jpg')





        # transposed_array = np.transpose(img[0].cpu().numpy(), (1, 2, 0))
        # #transposed_array = transposed_array[..., ::-1]
        # # Convert to uint8 if necessary (assuming the array is in the range [0, 255])
        # transposed_array = (transposed_array*255).astype(np.uint8)
        # img_mid = Image.fromarray(transposed_array)

        ###########################
        # out["seg_out"] = seg_out
        # out["masks"] = masks
        # out["class_out"] = class_out
        # out["class_gt"] = class_gt
        ##############################################
        #x3d = self.flosp(mlvl_feats, img_metas) # bs, c, nq
        bs, cam, _, _, _ = mlvl_feats[0].shape
        x3d = self.flosp([mlvl_feat.view(bs, cam, *mlvl_feat.shape[1:]) for mlvl_feat in intra], img_metas)

        bs, c, _ = x3d.shape
        x3d = self.bottleneck(x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z))
        #trainable_params = sum(p.numel() for p in self.bottleneck.parameters())
        #553600
        #torch.Size([1, 1280, 128, 128, 16])

        #geometry guidance
        # occ = self.occ_header(x3d).squeeze(1)
        # out["occ"] = occ
        # ###################

        x3d = x3d.reshape(bs, c, -1).contiguous()
        # torch.Size([1, 1280, 128x128x16])
        # Load proposals#######proposal mlvl_feats are not used
        pts_out = self.pts_header(mlvl_feats, img_metas, target)
        # trainable_params = sum(p.numel() for p in self.pts_header.parameters())
        #348546
        pts_occ = pts_out['occ_logit'].squeeze(1)
        proposal =  (pts_occ > 0).float().detach().cpu().numpy()
        out['pts_occ'] = pts_occ
        # #################
        if proposal.sum() < 2:
            proposal = np.ones_like(proposal)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1)>0)).astype(np.int32)
        out['unmasked_idx'] = unmasked_idx
        #[1, 197829]
        masked_idx = np.asarray(np.where(proposal.reshape(-1)==0)).astype(np.int32)
        #[1, 64315]
        vox_coords = self.get_voxel_indices()

        # Compute seed features
        seed_feats = x3d[0, :, vox_coords[unmasked_idx[0], 3]].permute(1, 0)
        #[197829,320]
        seed_coords = vox_coords[unmasked_idx[0], :3]
        coords_torch = torch.from_numpy(np.concatenate(
            [np.zeros_like(seed_coords[:, :1]), seed_coords], axis=1)).to(seed_feats.device)
        ############semantic guidance#########
        seed_feats_desc = self.sgb(seed_feats, coords_torch)
        sem = self.sem_header(seed_feats_desc)
        out["sem_logit"] = sem
        #seed_feats_desc = seed_feats
        # ##############################

        out["coords"] = seed_coords

        # Complete voxel features

        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=x3d.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims).contiguous()
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats_desc
        #### geometry prior to the unmask voxel
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mlp_prior(x3d[0, :, vox_coords[masked_idx[0], 3]].permute(1, 0))
        # vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(
        #     masked_idx.shape[1], self.embed_dims).to(dtype)
        # ###############################################
        vox_feats_diff = vox_feats_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        #
        if self.pts_header.guidance:
            vox_feats_diff = torch.cat([vox_feats_diff, pts_out['occ_x']], dim=1)


            # vox_feats_split = vox_feats_diff.reshape(1, 20, self.embed_dims // 20, self.bev_h, self.bev_w, self.bev_z)
            #
            # pts_out_expanded = pts_out_expanded.expand(-1, 20, -1, -1, -1, -1)
            # #vox_feats_diff = torch.cat([vox_feats_split, pts_out_expanded], dim=2)
            # vox_feats_diff = vox_feats_split + pts_out_expanded
            # vox_feats_diff = vox_feats_diff.reshape(1, self.embed_dims, self.bev_h, self.bev_w, self.bev_z)
            ##################################################


        vox_feats_diff = self.sdb(vox_feats_diff) # 1, C,H,W,Z
        #vox_feats_diff = vox_feats_diff[:,:320,:,:,:]
        #trainable_params = sum(p.numel() for p in self.sdb.parameters() if p.requires_grad)
        #trainable_params = 627840
        #（[1, 640, 128, 128, 16])


           #ssc_dict = self.ssc_header(vox_feats_diff)
#########################################
        # res = []
        # # start_time_ssc_header = time.time()
        # # ######################
        #
        #     #x_group = vox_feats_diff[:, i * ((self.embed_dims//2)//self.num_classes):(i + 1) * ((self.embed_dims//2)//self.num_classes), :, :, :]
        #     #print(i)
        # header_result = self.ssc_header(vox_feats_diff)
        # # up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # # header_result = up_scale_2(vox_feats_diff)[:,:20,:,:,:]
        # # trainable_params = sum(p.numel() for p in self.ssc_header.parameters())
        # # #9080
        # # pdb.set_trace()
        # # trainable_params = 627840
        #     #res.append(header_result["ssc_logit"])
        # res.append(header_result)
        # # #######################
        # # for i in range(20):
        # #     x_group = vox_feats_diff[:, i * ((self.embed_dims // 2) // self.num_classes):(i + 1) * (
        # #                 (self.embed_dims // 2) // self.num_classes), :, :, :]
        # #     # print(i)
        # #     header_result = self.ssc_header(x_group)
        # #     # res.append(header_result["ssc_logit"])
        # #     res.append(header_result)
        # #############################
        # # end_time_ssc_header = time.time()
        # # elapsed_time_ssc_header = end_time_ssc_header - start_time_ssc_header
        # # end_time = time.time()
        # # elapsed_time = end_time - start_time
        # #pdb.set_trace()
        # # Concatenate results along the feature dimension
        # ssc_dict = {}
        # ssc_dict["ssc_logit"] = torch.cat(res, dim=1)
        # #pdb.set_trace()
        # out.update(ssc_dict)
############################################################
        ssc_dict = self.ssc_header(vox_feats_diff)
        out.update(ssc_dict)
        return out

    def step(self, out_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """

        ssc_pred = out_dict["ssc_logit"]
        if step_type== "train":
            ####################semantic guidance
            sem_pred_2 = out_dict["sem_logit"]
            ############################
            target_2 = img_metas[0]['target_1_2'].unsqueeze(0).to(target.device)
            target_2_label = torch.from_numpy(img_metas[0]['target_1_2_label']).unsqueeze(0).to(target.device)
            target_1_label = torch.from_numpy(img_metas[0]['target_1_1_label']).unsqueeze(0).to(target.device)
            #target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
            coords = out_dict['coords']
            sp_target_2 = target_2.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2], : ]
            sp_target_2_label = target_2_label.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2]]
            ###############################
            loss_dict = dict()

            class_weight = self.class_weights.type_as(target)
            if img_metas[0]['target_1_1_temp_label'] is not None:

                valid_mask = (img_metas[0]['target_1_1_label'] != 255) | (img_metas[0]['target_1_1_temp_label'] != 255)
            else:
                valid_mask = (img_metas[0]['target_1_1_label'] != 255)
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight, valid_mask)
                loss_dict['loss_ssc'] = loss_ssc
            #
            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target, valid_mask)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target,valid_mask)
                loss_dict['loss_geo_scal'] = loss_geo_scal
            # if self.CE_ssc_loss:
            #     loss_ssc = CE_ssc_loss(ssc_pred, target_1_label, class_weight)
            #     loss_dict['loss_ssc'] = loss_ssc
            #
            # if self.sem_scal_loss:
            #     loss_sem_scal = sem_scal_loss(ssc_pred, target_1_label)
            #     loss_dict['loss_sem_scal'] = loss_sem_scal
            #
            # if self.geo_scal_loss:
            #     loss_geo_scal = geo_scal_loss(ssc_pred, target_1_label)
            #     loss_dict['loss_geo_scal'] = loss_geo_scal
            ######################semantic guidance
            # loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=1), sp_target_2_label, ignore=255)
            # #0.9492
            # #pdb.set_trace()
            # loss_sem += F.cross_entropy(sem_pred_2, sp_target_2_label.long(), ignore_index=255)
            #3.8475


            if img_metas[0]['target_1_2_temp_label'] is not None:
                valid_mask_2 = torch.from_numpy(((img_metas[0]['target_1_2_label'] != 255) | (img_metas[0]['target_1_2_temp_label'] != 255))).unsqueeze(0)
            else:
                valid_mask_2 = torch.from_numpy((img_metas[0]['target_1_2_label'] != 255)).unsqueeze(0)
                #[1,128,128,16]
            unmasked_indx = out_dict['unmasked_idx']

            valid_mask_2_sem = valid_mask_2.view(1,-1)
            #[1,262144]

            valid_mask_2_sem = valid_mask_2_sem[:,unmasked_indx[0,:]]
            ##########################
            loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=-1), sp_target_2, 'present', valid_mask_2_sem)
            #0.9492  [197829,20]
            log_prob_sem = F.log_softmax(sem_pred_2, dim=-1)
            loss_softmax_sem= -torch.sum(sp_target_2 * log_prob_sem, dim=-1).squeeze()
            loss_softmax_sem = loss_softmax_sem[valid_mask_2_sem[0]]
            loss_softmax_sem = torch.mean(loss_softmax_sem)
            loss_sem+=loss_softmax_sem
            loss_dict['loss_sem'] = loss_sem
            ########################################
            ones = torch.ones_like(target_2_label).to(target_2_label.device)
            zeros = torch.zeros_like(target_2_label).to(target_2_label.device)
            target_2_binary = torch.where(target_2[:,:,:,:,0] != 0, zeros, ones)

            # ones = torch.ones_like(target_2_label).to(target_2_label.device)
            # target_2_binary = torch.where(torch.logical_or(target_2_label == 255, target_2_label == 0), target_2_label, ones)
            ########### geometry guidance############

            # loss_occ = F.binary_cross_entropy(out_dict['occ'].sigmoid()[valid_mask_2], target_2_binary[valid_mask_2].to(target.device).float())
            # loss_dict['loss_occ'] = loss_occ
            ########################
            # loss_dict['loss_pts'] = F.binary_cross_entropy(out_dict['pts_occ'].sigmoid()[target_2_binary != 255],
            #                                                target_2_binary[target_2_binary != 255].float())


            loss_dict['loss_pts'] = F.binary_cross_entropy(out_dict['pts_occ'].sigmoid()[valid_mask_2], target_2_binary[valid_mask_2].to(target.device).float())
            ############
            # class_out = out_dict["class_out"]
            # class_gt = out_dict["class_gt"]
            # loss_dict['loss_class'] = F.binary_cross_entropy(class_out, class_gt)
            # seg_out = out_dict["seg_out"]
            # masks = out_dict["masks"]
            # #pdb.set_trace()
            # seg_target = masks.argmax(dim=1).long()
            # loss_dict['loss_mask'] = F.cross_entropy(seg_out, seg_target)           # class_out = out_dict["class_out"]
            # class_gt = out_dict["class_gt"]
            # loss_dict['loss_class'] = F.binary_cross_entropy(class_out, class_gt)
            # seg_out = out_dict["seg_out"]
            # masks = out_dict["masks"]
            #loss_dict['loss_mask'] = F.binary_cross_entropy(seg_out, masks.float()) + dice_loss(seg_out, masks.float())
            #############
            return loss_dict
#####################################
        elif step_type== "val" or "test":
        #elif step_type == "train":
            result = dict()
            result['output_voxels'] = ssc_pred.permute(0, 4, 1, 2, 3)
            #result['output_voxels'] = ssc_pred
            #[1,20,256,256,32]
            if step_type == "val":
                result['target_voxels'] = torch.tensor(img_metas[0]['target_1_1_label']).unsqueeze(0).to(target.device)
            else:
                result['target_voxels'] = torch.ones(1,256,256,32).to(ssc_pred.device)
            #[1,256,256,32]

            if self.save_flag:
                y_pred = ssc_pred
                y_pred = y_pred.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
                #y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                self.save_pred(img_metas, y_pred)
            return result

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        #return self.step(out_dict, target, img_metas, "test")
        return self.step(out_dict, target, img_metas, "val")

    def get_voxel_indices(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h
        #represent for bonderay
        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        return vox_coords

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        Note: only for semantickitti
        """

        # y_pred[y_pred==10] = 44
        # y_pred[y_pred==11] = 48
        # y_pred[y_pred==12] = 49
        # y_pred[y_pred==13] = 50
        # y_pred[y_pred==14] = 51
        # y_pred[y_pred==15] = 70
        # y_pred[y_pred==16] = 71
        # y_pred[y_pred==17] = 72
        # y_pred[y_pred==18] = 80
        # y_pred[y_pred==19] = 81
        # y_pred[y_pred==1] = 10
        # y_pred[y_pred==2] = 11
        # y_pred[y_pred==3] = 15
        # y_pred[y_pred==4] = 18
        # y_pred[y_pred==5] = 20
        # y_pred[y_pred==6] = 30
        # y_pred[y_pred==7] = 31
        # y_pred[y_pred==8] = 32
        # y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./kitti_5_16", "sequences", img_metas[0]['sequence_id'], "predictions")
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))


def dice_loss(pred, target, smooth=1.):
    # Apply sigmoid activation to the predictions
    pred = torch.sigmoid(pred)

    # Calculate the intersection between predictions and targets
    intersection = (pred * target).sum(dim=(2, 3))

    # Calculate the Dice coefficient
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)

    # Return the Dice loss
    return 1 - dice.mean()
