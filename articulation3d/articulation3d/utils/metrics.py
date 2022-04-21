import logging
import torch
import numpy as np

@torch.no_grad()
def compare_planes(pred_planes, gt_planes, ):
    """
    naively calculate 3d vector l2 distance
    """
    pred_planes = torch.tensor(np.array(pred_planes), dtype=torch.float32)
    pred_offsets = torch.norm(pred_planes, p=2, dim=1) + 1e-5
    pred_norms = pred_planes.div(pred_offsets.view(-1,1).expand_as(pred_planes))
    gt_planes = torch.tensor(np.array(gt_planes), dtype=torch.float32)
    gt_offsets = torch.norm(gt_planes, p=2, dim=1) + 1e-5
    gt_norms = gt_planes.div(gt_offsets.view(-1,1).expand_as(gt_planes))
    norm_distance_matrix = torch.clamp(torch.cdist(pred_norms, gt_norms, p=2), 0, 2)
    norm_angle_matrix = 2*torch.asin(norm_distance_matrix/2) / np.pi * 180
    offset_distance_matrix = torch.cdist(pred_offsets.view(-1,1), gt_offsets.view(-1,1), p=1)
    return {'norm': norm_angle_matrix, 'offset': offset_distance_matrix}

def compare_planes_one_to_one(pred_planes, gt_planes, ):
    pred_planes = torch.tensor(np.array(pred_planes), dtype=torch.float32)
    pred_offsets = torch.clamp(torch.norm(pred_planes, p=2, dim=1), min=1e-5)
    pred_norms = pred_planes.div(pred_offsets.view(-1,1).expand_as(pred_planes))
    gt_planes = torch.tensor(np.array(gt_planes), dtype=torch.float32)
    gt_offsets = torch.clamp(torch.norm(gt_planes, p=2, dim=1), min=1e-5)
    gt_norms = gt_planes.div(gt_offsets.view(-1,1).expand_as(gt_planes))

    l2 = torch.norm(pred_planes-gt_planes, dim=1).numpy().mean()
    norm = torch.acos(torch.clamp(torch.sum(pred_norms * gt_norms, dim=1), max=1, min=-1)).numpy().mean()
    offset = torch.abs(pred_offsets - gt_offsets).numpy().mean()
    return {'l2': l2, 'norm': norm, 'offset': offset}


@torch.no_grad()
def compare_axis(pred_axis, gt_axis, ):
    """
    naively calculate 3d vector l2 distance
    """
    if len(gt_axis) == 0 or len(pred_axis) == 0:
        return {'norm': [], 'offset': []}
    pred_norms = pred_axis[:,:2]
    gt_norms = gt_axis[:,:2]
    pred_offsets = pred_axis[:,2:]
    gt_offsets = gt_axis[:,2:]
    norm_distance_matrix = torch.clamp(torch.cdist(pred_norms, gt_norms, p=2), 0, 2)
    norm_angle_matrix = 2*torch.asin(norm_distance_matrix/2) / np.pi * 180
    offset_distance_matrix = torch.cdist(pred_offsets.view(-1,1), gt_offsets.view(-1,1), p=1)

    return {'norm': norm_angle_matrix, 'offset': offset_distance_matrix}

def sa_metric(angle_p, angle_g):
    d_angle = np.abs(angle_p - angle_g)
    d_angle = min(d_angle, np.pi - d_angle)
    d_angle = d_angle * 2 / np.pi
    return max(0, (1 - d_angle)) ** 2

def se_metric(coord_p, coord_g, size=(640, 480)):
    c_p = [(coord_p[0] + coord_p[2]) / 2, (coord_p[1] + coord_p[3]) / 2]
    c_g = [(coord_g[0] + coord_g[2]) / 2, (coord_g[1] + coord_g[3]) / 2]
    d_coord = np.abs(c_p[0] - c_g[0])**2 + np.abs(c_p[1] - c_g[1])**2
    d_coord = np.sqrt(d_coord) / max(size[0], size[1])
    return max(0, (1 - d_coord)) ** 2

def EA_metric(l_pred, l_gt, size=(640, 480)):
    se = se_metric(l_pred.coord, l_gt.coord, size=size)
    sa = sa_metric(l_pred.angle(), l_gt.angle())
    return sa * se

class Line(object):
    def __init__(self, coordinates=[0, 0, 1, 1]):
        """
        coordinates: [y0, x0, y1, x1]
        """
        assert isinstance(coordinates, list)
        assert len(coordinates) == 4
        assert coordinates[0]!=coordinates[2] or coordinates[1]!=coordinates[3]
        self.__coordinates = coordinates

    @property
    def coord(self):
        return self.__coordinates

    @property
    def length(self):
        start = np.array(self.coord[:2])
        end = np.array(self.coord[2::])
        return np.sqrt(((start - end) ** 2).sum())

    def angle(self):
        y0, x0, y1, x1 = self.coord
        if x0 == x1:
            return -np.pi / 2
        return np.arctan((y0-y1) / (x0-x1))

    def rescale(self, rh, rw):
        coor = np.array(self.__coordinates)
        r = np.array([rh, rw, rh, rw])
        self.__coordinates = np.round(coor * r).astype(np.int).tolist()

    def __repr__(self):
        return str(self.coord)