from pyexpat import model
import torch
from torch import nn
import utils.general as utils
import math
import pdb

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    
    
class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 virtual_normal_weight = 0,
                 decode_rgb_weight = 0.1,
                 point_based_weight = 0,
                 weigths_consistency_weight = 0,
                 rgb_occ_weight = 1,
                 depth_occ_weight = 0.1,
                 normal_l1_occ_weight = 0.05,
                 normal_cos_occ_weight = 0.05,
                 end_step = -1,
                 select_group = 100):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.virtual_normal_weight = virtual_normal_weight
        self.decode_rgb_weight = decode_rgb_weight
        self.point_based_weight = point_based_weight
        self.weigths_consistency_weight = weigths_consistency_weight
        self.rgb_occ_weight = rgb_occ_weight
        self.depth_occ_weight = depth_occ_weight
        self.normal_l1_occ_weight = normal_l1_occ_weight
        self.normal_cos_occ_weight = normal_cos_occ_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step
        self.select_group = select_group

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt, mask):
        # TODO remove hard-coded scaling for depth
        return self.depth_loss(depth_pred.reshape(1, 32, 32), (depth_gt * 50 + 0.5).reshape(1, 32, 32), mask.reshape(1, 32, 32))
        
    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        return l1, cos

    def get_virtual_normal_loss(self, depth_pred, depth_gt, mask, uv, intrinsics):
        inv_intrinsics = torch.inverse(intrinsics.squeeze())
        uv = uv.reshape(-1, 2)[mask.squeeze()]
        xyz = torch.cat([uv, torch.ones_like(uv[:, :1])], dim=-1).permute(1, 0) # (N, 3)
        xyz = torch.matmul(inv_intrinsics[:3, :3], xyz).permute(1, 0) # (3, N)
        xyz_pred = xyz * depth_pred.reshape(-1, 1) # (3, N)
        xyz_gt = xyz * depth_gt.reshape(-1, 1) # (3, N)

        p_pred_list = []
        p_gt_list = []
        while True:
            select_index = torch.randperm(uv.shape[0])[:self.select_group]
            pt1 = xyz_gt[select_index[0]]
            pt2 = xyz_gt[select_index[1]]
            pt3 = xyz_gt[select_index[2]]

            xyz12 = pt2 - pt1
            xyz13 = pt3 - pt1
            xyz23 = pt3 - pt2

            norm_xyz12 = torch.norm(xyz12)
            norm_xyz13 = torch.norm(xyz13)
            norm_xyz23 = torch.norm(xyz23)

            cos_12 = torch.dot(xyz12, xyz13) / (norm_xyz12 * norm_xyz13)
            cos_13 = torch.dot(xyz12, xyz23) / (norm_xyz12 * norm_xyz23)
            cos_23 = torch.dot(xyz13, xyz23) / (norm_xyz13 * norm_xyz23)

            if (cos_12 > 0.867 or cos_12 < -0.867) and (cos_13 > 0.867 or cos_13 < -0.867) and (cos_23 > 0.867 or cos_23 < -0.867):
                continue
            if norm_xyz12 < 0.007 or norm_xyz13 < 0.007 or norm_xyz23 < 0.007:
                continue
            point = torch.stack([pt1, pt2, pt3], dim=-1)

            pt1_pred = xyz_pred[select_index[0]]
            pt2_pred = xyz_pred[select_index[1]]
            pt3_pred = xyz_pred[select_index[2]]
            point_pred = torch.stack([pt1_pred, pt2_pred, pt3_pred], dim=0)
            p_gt_list.append(point)
            p_pred_list.append(point_pred)
            if len(p_gt_list) > 100:
                break

        loss = 0
        for i in range(len(p_gt_list)):
            virtual_normal_pred = torch.cross(p_pred_list[i][0] - p_pred_list[i][1], p_pred_list[i][0] - p_pred_list[i][2])
            virtual_normal_gt = torch.cross(p_gt_list[i][0] - p_gt_list[i][1], p_gt_list[i][0] - p_gt_list[i][2])
            loss += torch.mean(torch.nn.functional.normalize(virtual_normal_pred, p=2, dim=-1) - torch.nn.functional.normalize(virtual_normal_gt, p=2, dim=-1))
        
        loss /= len(p_gt_list)

        return loss

    def get_point_based_depth_loss(self, weights, depth_pred_vals, depth_gt_scaled, mask):
        indices = torch.searchsorted(depth_pred_vals, depth_gt_scaled.squeeze(0))[mask.squeeze()]
        weights = weights[mask.squeeze()]
        selected_weights_bef = weights[torch.arange(0, indices.shape[0]), indices]
        selected_weights_aft = weights[torch.arange(0, indices.shape[0]), torch.clamp_max(indices + 1, weights.shape[-1]-1)]

        point_based_depth_loss = torch.mean((torch.ones_like(selected_weights_bef) - selected_weights_bef)**2 + (torch.ones_like(selected_weights_aft) - selected_weights_aft)**2)
        return point_based_depth_loss

    def get_weights_consistency_loss(self, weights_sdf, weights_occ):
        # we hope the sdf weights can be the same with occ weights
        return torch.mean(torch.abs((weights_sdf - weights_occ.detach())**2))

        
    def forward(self, model_outputs, model_input, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        weights_sdf = model_outputs['weights'].cuda()
        weights_occ = model_outputs['weights_occ'].cuda()
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]
        normal_pred_occ = model_outputs['normal_map_occ'][None]
        
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        rgb_loss_occ = self.get_rgb_loss(model_outputs['rgb_values_occ'], rgb_gt)
        if "rgb_direct_decoder_values" in model_outputs:
            rgb_direct_loss = self.get_rgb_loss(model_outputs['rgb_direct_decoder_values'], rgb_gt)
        else:
            rgb_direct_loss = torch.tensor(0.0).cuda().float()
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        if self.point_based_weight > 0:
            scale, shift = compute_scale_and_shift(depth_pred, depth_gt, mask)
            depth_gt_scaled = (depth_gt - shift) / scale
            point_based_loss = self.get_point_based_depth_loss(model_outputs['weights'], model_outputs['depth_vals'], depth_gt_scaled, mask)
        else:
            point_based_loss = torch.tensor(0.0).cuda().float()

        if self.depth_weight > 0:
            depth_loss = self.get_depth_loss(depth_pred, depth_gt, mask)
            depth_loss_occ = self.get_depth_loss(model_outputs['depth_values_occ'], depth_gt, mask)
        else:
            depth_loss = torch.tensor(0.0).cuda().float()    
            depth_loss_occ = torch.tensor(0.0).cuda().float()

        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)
        normal_l1_occ, normal_cos_occ = self.get_normal_loss(normal_pred_occ * mask, normal_gt)
        
        smooth_loss = self.get_smooth_loss(model_outputs)

        if self.virtual_normal_weight > 0:
            virtual_normal_loss = self.get_virtual_normal_loss(depth_pred, depth_gt, mask, model_input['uv'], model_input['intrinsics'])
        else:
            virtual_normal_loss = torch.tensor(0.0).cuda().float()

        if self.weigths_consistency_weight > 0:
            weights_consistency_loss = self.get_weights_consistency_loss(weights_sdf, weights_occ)
        else:
            weights_consistency_loss = torch.tensor(0.0).cuda().float()
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        loss = rgb_loss +\
               self.rgb_occ_weight * rgb_loss_occ +\
               self.eikonal_weight * eikonal_loss +\
               self.smooth_weight * smooth_loss +\
               self.decode_rgb_weight * rgb_direct_loss +\
               decay * self.depth_weight * depth_loss +\
               decay * self.depth_occ_weight * depth_loss_occ +\
               decay * self.normal_l1_weight * normal_l1 +\
               decay * self.normal_cos_weight * normal_cos +\
               decay * self.normal_l1_occ_weight * normal_l1_occ +\
               decay * self.normal_cos_occ_weight * normal_cos_occ +\
               decay * self.virtual_normal_weight * virtual_normal_loss+\
               decay * self.weigths_consistency_weight * weights_consistency_loss
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_loss': depth_loss,
            'normal_l1': normal_l1,
            'normal_cos': normal_cos,
            'virtual_normal_loss': virtual_normal_loss,
            'rgb_direct_loss': rgb_direct_loss,
            'rgb_loss_occ': rgb_loss_occ,
            'depth_loss_occ': depth_loss_occ,
            'normal_l1_occ': normal_l1_occ,
            'normal_cos_occ': normal_cos_occ,
        }

        return output
