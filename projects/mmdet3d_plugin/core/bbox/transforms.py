import numpy as np


def bbox3d2result(bboxes, scores, labels, trajs=None, trajs_cls=None, attrs=None):
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if trajs is not None:
        result_dict['trajs_3d'] = trajs.cpu()
    if trajs_cls is not None:
        result_dict['trajs_cls_3d'] = trajs_cls.cpu()

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict


def map3d2result(bboxes, scores, labels):
    result_dict = dict(
        vectors=bboxes.cpu(),
        scores=scores.cpu(),
        labels=labels.cpu())

    return result_dict


def occ3d2result(occ):
    result_dict = dict(
        occ=occ['occ'].cpu().numpy().astype(np.uint8))
    if 'flow' in occ:
        result_dict['flow'] = occ['flow'].cpu().numpy()

    return result_dict


def seg_map3d2result(seg):
    result_dict = dict(
        seg=seg.cpu())

    return result_dict


def planning3d2result(plan_reg):
    result_dict = dict(
        plan_reg=plan_reg.cpu().numpy())

    return result_dict