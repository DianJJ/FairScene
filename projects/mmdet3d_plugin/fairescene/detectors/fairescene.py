from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
@DETECTORS.register_module()
class FairScene(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 occupancy=False,
                 ):

        super(FairScene,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.only_occ = occupancy

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            img_feats = self.img_backbone(img)
            #trainable_params = sum(p.numel() for p in self.img_backbone.parameters())
            #23508032
            #########################mask dino
            #img_feats = img_feats['feats']
            ###########################
            shape0 = img_feats[0].shape
            # original torch.Size([1, 256, 93, 305])
            # mask dino torch.Size([1, 128, 93, 305])
            # efficientnet [1,48,93,305]
            shape1 = img_feats[1].shape
            # efficientnet [1,80,93,305]
            # original torch.Size([1, 512, 47, 153])
            #mask dino torch.Size([1, 128, 47, 153])
            shape2 = img_feats[2].shape
            # efficientnet [1,224,24,77]
            # original torch.Size([1, 1024, 24, 77])
            # mask dino torch.Size([1, 128, 24, 77])
            shape3 = img_feats[3].shape
            # efficientnet [1,2560,12,39]
            # original torch.Size([1, 2048, 12, 39])
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            #trainable_params = sum(p.numel() for p in self.img_neck.parameters())
            #1082368
            #shape0 = img_feats[0].shape
            #torch.Size([1, 128, 93, 305])
            #shape1 = img_feats[1].shape
            #torch.Size([1, 128, 47, 153])
            #shape2 = img_feats[2].shape
            #torch.Size([1, 128, 24, 77])
            #shape3 = img_feats[3].shape
            #torch.Size([1, 128, 12, 39])

        img_feats_reshaped = []
        #print(len(img_feats))

        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped,img

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats,img = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats, img

    def forward_pts_train(self,
                          img,
                          masks,
                          img_feats, 
                          img_metas,
                          target):
        """Forward function'
        """
        outs = self.pts_bbox_head(img, masks, img_feats, img_metas, target)
        #trainable_params = sum(p.numel() for p in self.pts_bbox_head.parameters())
        #57323127
        losses = self.pts_bbox_head.training_step(outs, target, img_metas)
        return losses

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      masks = None,
                      target=None):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Losses of different branches.
        """
        #print("masks shape:",masks.shape)
        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        masks = masks[:, -1, ...]
        if self.only_occ:
            img_feats = None
        else:
            img_feats, img = self.extract_feat(img=img)

        losses = dict()
        losses_pts = self.forward_pts_train(img, masks, img_feats, img_metas, target)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     img_metas=None,
                     img=None,
                     masks = None,
                     target=None,
                      **kwargs):
        """Forward testing function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Completion result.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        masks = masks[:, -1, ...]
        if self.only_occ:
            img_feats = None
        else:
            img_feats,img = self.extract_feat(img=img)
        outs = self.pts_bbox_head(img, masks, img_feats, img_metas, target)

        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        return completion_results
