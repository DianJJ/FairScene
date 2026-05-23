work_dir = 'work_dirs/fairescene-S-Efficientnet_b7-semantickitti'
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_dim_ = 320
#_dim_ = 128
_labels_tag_ = 'labels'
_temporal_ = []
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]

_sem_scal_loss_ = True
_geo_scal_loss_ = True
_depthmodel_= 'msnet3d'
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
pretrained='ckpts/efficientnet-b7_3rdparty-ra-noisystudent_in1k_20221103-a82894bc.pth'
model = dict(
   type='FairScene',
   #pretrained=dict(img='ckpts/efficientnet-b7_3rdparty-ra-noisystudent_in1k_20221103-a82894bc.pth'),
   img_backbone=dict(
        # _delete_=True,
        type='mmcls.EfficientNet',
        arch='b7',
        out_indices=(2, 3, 4, 6),
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.'),
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        # style='pytorch'
   ),
    img_neck=dict(
        type='FPN',
        # in_channels=[1024],
        in_channels=[48, 80, 224, 2560],
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        # num_outs=1,
        num_outs=4,
        relu_before_extra_convs=True),
   pts_bbox_head=dict(
       type='FairSceneHead',
       bev_h=128,
       bev_w=128,
       bev_z=16,
       embed_dims=_dim_,
       pts_header_dict=dict(
           type='FairSceneOccHead',
           point_cloud_range=point_cloud_range,
           spatial_shape=[256,256,32],
           guidance=True,
           nbr_classes=1),
       CE_ssc_loss=True,
       geo_scal_loss=_geo_scal_loss_,
       sem_scal_loss=_sem_scal_loss_,
       #scale_2d_list=[16]
       scale_2d_list=[4,8,16,32]
       #scale_2d_list=[4]
       ),
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4)))


dataset_type = 'SemanticKittiDataset'
data_root = './kitti/'
file_client_args = dict(backend='disk')

data = dict(
   samples_per_gpu=1,
   workers_per_gpu=4,
   train=dict(
       type=dataset_type,
       split = "train",
       test_mode=False,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       temporal = _temporal_,
       labels_tag = _labels_tag_),
   val=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       temporal = _temporal_,
       labels_tag = _labels_tag_),
   test=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       temporal = _temporal_,
       labels_tag = _labels_tag_),
   shuffler_sampler=dict(type='DistributedGroupSampler'),
   nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
   type='AdamW',
   lr=2e-4,
   weight_decay=0.01)
# optimizer = dict(
#     type='SGD',
#     lr=1e-2,
#     momentum=0.9,  # Set the momentum value
#     weight_decay=0.005
# )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 3,
   min_lr_ratio=1e-3)
total_epochs = 48
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
   interval=50,
   hooks=[
       dict(type='TextLoggerHook'),
       dict(type='TensorboardLoggerHook')
   ])

# checkpoint_config = None
checkpoint_config = dict(interval=1)
