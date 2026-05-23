import torch
from torchmetrics.metric import Metric

class SSCMetrics(Metric):
    def __init__(self, n_classes=20, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        
        self.n_classes = n_classes
        
        self.add_state('tps', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        self.add_state('fps', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        self.add_state('fns', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        
        self.add_state('completion_tp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fn', default=torch.zeros(1), dist_reduce_fx='sum')
    
    def compute_single(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        # # evaluate semantic completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        
        ret = (tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy(), tp_sum.cpu().numpy(), fp_sum.cpu().numpy(), fn_sum.cpu().numpy())
        
        return ret
        
    def update(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn
        
        # # evaluate semantic completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum
    
    def compute(self):
        precision = self.completion_tp / (self.completion_tp + self.completion_fp)
        recall = self.completion_tp / (self.completion_tp + self.completion_fn)
        iou = self.completion_tp / \
                (self.completion_tp + self.completion_fp + self.completion_fn)
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
       
        per_prec = self.tps / (self.tps + self.fps + eps)
        per_rec  = self.tps / (self.tps + self.fns + eps)
        per_f1   = 2 * per_prec * per_rec / (per_prec + per_rec + eps)
        
        
        output = {
            "precision": precision,
            "recall": recall,
            "iou": iou.item(),
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": iou_ssc[1:].mean().item(),
        }

        for cls_idx in range(self.n_classes):
            output[f"precision_cls{cls_idx}"] = per_prec[cls_idx].item()
            output[f"recall_cls{cls_idx}"]    = per_rec[cls_idx].item()
            output[f"f1_cls{cls_idx}"]        = per_f1[cls_idx].item()

        
        return output

    def get_score_completion(self, predict, target, nonempty=None):
        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = torch.zeros_like(predict)
        b_true = torch.zeros_like(target)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = torch.sum((y_true == 1) & (y_pred == 1))
            fp = torch.sum((y_true != 1) & (y_pred == 1))
            fn = torch.sum((y_true == 1) & (y_pred != 1))
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        tp_sum = torch.zeros(_C).type_as(predict)
        fp_sum = torch.zeros(_C).type_as(predict)
        fn_sum = torch.zeros(_C).type_as(predict)

        for idx in range(_bs):
            y_true = target[idx]  # GT
            y_pred = predict[idx]
            
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                valid_mask = (nonempty_idx == 1) & (y_true != 255)
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]
            
            for j in range(_C):  # for each class
                tp = torch.sum((y_true == j) & (y_pred == j))
                fp = torch.sum((y_true != j) & (y_pred == j))
                fn = torch.sum((y_true == j) & (y_pred != j))
                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum
    

class Metric_PointPRF:
    """Point‑level per‑class Accuracy / Completeness / F1 (SSC geometry metric)."""

    def __init__(self,
                 num_classes: int = 18,
                 leaf_size: int = 10,
                 th_acc: float = 1,
                 th_comp: float = 1,
                 voxel_size=(0.4, 0.4, 0.4),
                 area_range=(-40, -40, -1, 40, 40, 5.4),
                 void=(17, 255),
                 use_lidar_mask=False,
                 use_image_mask=False,
                 class_names=None):
        self.num_classes = num_classes
        self.leaf_size = leaf_size
        self.th_acc = th_acc
        self.th_comp = th_comp
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.area_range = np.array(area_range, dtype=np.float32)
        self.void = set(void)
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.class_names = class_names or []

        self.sum_acc = np.zeros(self.num_classes, dtype=np.float64)
        self.sum_comp = np.zeros(self.num_classes, dtype=np.float64)
        self.sum_f1 = np.zeros(self.num_classes, dtype=np.float64)
        self.count_samples = np.zeros(self.num_classes, dtype=np.int64)
        self.eps = 1e-8

    def _vox2pts(self, idx):
        pts = idx.astype(np.float32)
        pts *= self.voxel_size
        pts += self.voxel_size / 2
        pts[:, 0] += self.area_range[0]
        pts[:, 1] += self.area_range[1]
        pts[:, 2] += self.area_range[2]
        return pts

    def add_batch(self, pred, gt, mask_lidar=None, mask_camera=None):
        if self.use_image_mask and mask_camera is not None:
            gt = gt.copy(); pred = pred.copy()
            gt[~mask_camera] = 255; pred[~mask_camera] = 255
        elif self.use_lidar_mask and mask_lidar is not None:
            gt = gt.copy(); pred = pred.copy()
            gt[~mask_lidar] = 255; pred[~mask_lidar] = 255

        for c in range(self.num_classes):
            if c in self.void:
                continue
            mask_pred_c = pred == c
            mask_gt_c = gt == c
            if not mask_pred_c.any() and not mask_gt_c.any():
                continue

            self.count_samples[c] += 1
            pts_pred = self._vox2pts(np.stack(np.where(mask_pred_c), axis=1)) if mask_pred_c.any() else np.empty((0,3))
            pts_gt = self._vox2pts(np.stack(np.where(mask_gt_c), axis=1)) if mask_gt_c.any() else np.empty((0,3))

            if pts_pred.shape[0] == 0:
                acc = 0.0
            else:
                tree_gt = KDTree(pts_gt if pts_gt.shape[0] else np.array([[1e9,0,0]]), leaf_size=self.leaf_size)
                d_acc, _ = tree_gt.query(pts_pred)
                acc = (d_acc.flatten() < self.th_acc).mean()

            if pts_gt.shape[0] == 0:
                comp = 0.0
            else:
                tree_pred = KDTree(pts_pred if pts_pred.shape[0] else np.array([[1e9,0,0]]), leaf_size=self.leaf_size)
                d_comp, _ = tree_pred.query(pts_gt)
                comp = (d_comp.flatten() < self.th_comp).mean()

            f1 = 2 * acc * comp / (acc + comp + self.eps)
            self.sum_acc[c] += acc
            self.sum_comp[c] += comp
            self.sum_f1[c] += f1

    def report(self, return_dict=False):
        per_class = {}
        macro_p = macro_r = macro_f1 = 0.0
        num_valid = 0
        for c in range(self.num_classes):
            if c in self.void or self.count_samples[c] == 0:
                continue
            acc = self.sum_acc[c] / self.count_samples[c]
            comp = self.sum_comp[c] / self.count_samples[c]
            f1 = self.sum_f1[c] / self.count_samples[c]
            per_class[self.class_names[c]] = {
                'Acc': acc,
                'Comp': comp,
                'F1': f1
            }
            macro_p += acc; macro_r += comp; macro_f1 += f1; num_valid += 1
        macro = {
            'Macro_Acc': macro_p / num_valid if num_valid else 0.0,
            'Macro_Comp': macro_r / num_valid if num_valid else 0.0,
            'Macro_F1': macro_f1 / num_valid if num_valid else 0.0
        }
        if return_dict:
            return {'per_class': per_class, 'macro': macro}
        # default print
        print("######## Point‑level per‑class PRF ########")
        for cls, vals in per_class.items():
            print(f"{cls:20} Acc:{vals['Acc']*100:6.2f} | Comp:{vals['Comp']*100:6.2f} | F1:{vals['F1']*100:6.2f}")
        print("-------------------------------------------------------------------")
        print(f"Macro‑ave Accuracy      : {macro['Macro_Acc']*100:6.2f}")
        print(f"Macro‑ave Completeness  : {macro['Macro_Comp']*100:6.2f}")
        print(f"Macro‑ave F1            : {macro['Macro_F1']*100:6.2f}")