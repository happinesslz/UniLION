from shapely.geometry import LineString, box, Polygon
from shapely import ops, strtree

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union

class NuscMapExtractor(object):
    """NuScenes map ground-truth extractor.

    Args:
        data_root (str): path to nuScenes dataset
        roi_size (tuple or list): bev range
    """
    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        self.roi_size = roi_size
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        
        # local patch in nuScenes format
        self.local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, 
                roi_size[0] / 2, roi_size[1] / 2)
    
    def _union_ped(self, ped_geoms: List[Polygon]) -> List[Polygon]:
        ''' merge close ped crossings.
        
        Args:
            ped_geoms (list): list of Polygon
        
        Returns:
            union_ped_geoms (Dict): merged ped crossings 
        '''

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = strtree.STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                # o = ped_geoms[o]
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))
        return results

    def get_map_mask(self,
                     seg_map_config,
                     location: str,
                     scale_bda: float,
                     translation: Union[List, NDArray],
                     rotation: Union[List, NDArray]):
        seg_classes = seg_map_config['seg_map_classes']
        xbound = seg_map_config['xbound']
        ybound = seg_map_config['ybound']
        patch_h = (ybound[1] - ybound[0]) / scale_bda
        patch_w = (xbound[1] - xbound[0]) / scale_bda
        canvas_h = int((ybound[1] - ybound[0]) / ybound[2])
        canvas_w = int((xbound[1] - xbound[0]) / xbound[2])
        patch_size = (patch_h, patch_w)
        canvas_size = (canvas_h, canvas_w)

        patch_box = (translation[0], translation[1], patch_size[0], patch_size[1])
        # rotation = Quaternion(rotation)
        # yaw = quaternion_yaw(rotation) / np.pi * 180
        rotation = rotation[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        yaw = yaw / np.pi * 180

        mappings = {}
        for name in seg_classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        masks = self.map_explorer[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=yaw,
            layer_names=layer_names,
            canvas_size=canvas_size,
        )

        masks = masks.astype(np.bool)

        num_classes = len(seg_classes)
        gt_bev_masks = np.zeros((num_classes, *canvas_size), dtype=np.long)
        for k, name in enumerate(seg_classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                gt_bev_masks[k, masks[index]] = 1

        return gt_bev_masks

    def get_map_geom(self,
                     location: str, 
                     translation: Union[List, NDArray],
                     rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `location` and self pose, self may be lidar or ego.
        
        Args:
            location (str): city name
            translation (array): self2global translation, shape (3,)
            rotation (array): self2global quaternion, shape (4, )
            
        Returns:
            geometries (Dict): extracted geometries by category.
        '''
        with np.errstate(invalid="ignore"):
            # (center_x, center_y, len_y, len_x) in nuscenes format
            patch_box = (translation[0], translation[1], self.roi_size[1], self.roi_size[0])
            # rotation = Quaternion(rotation)
            # yaw = quaternion_yaw(rotation) / np.pi * 180
            rotation = rotation[:3, :3]
            v = np.dot(rotation, np.array([1, 0, 0]))
            yaw = np.arctan2(v[1], v[0])
            yaw = yaw / np.pi * 180

            # get dividers
            lane_dividers = self.map_explorer[location]._get_layer_line(
                        patch_box, yaw, 'lane_divider')

            road_dividers = self.map_explorer[location]._get_layer_line(
                        patch_box, yaw, 'road_divider')

            all_dividers = []
            for line in lane_dividers + road_dividers:
                all_dividers += split_collections(line)

            # get ped crossings
            ped_crossings = []
            ped = self.map_explorer[location]._get_layer_polygon(
                        patch_box, yaw, 'ped_crossing')

            for p in ped:
                ped_crossings += split_collections(p)
            # some ped crossings are split into several small parts
            # we need to merge them
            ped_crossings = self._union_ped(ped_crossings)

            ped_crossing_lines = []
            for p in ped_crossings:
                # extract exteriors to get a closed polyline
                line = get_ped_crossing_contour(p, self.local_patch)
                if line is not None:
                    ped_crossing_lines.append(line)

            # get boundaries
            # we take the union of road segments and lanes as drivable areas
            # we don't take drivable area layer in nuScenes since its definition may be ambiguous
            road_segments = self.map_explorer[location]._get_layer_polygon(
                        patch_box, yaw, 'road_segment')
            lanes = self.map_explorer[location]._get_layer_polygon(
                        patch_box, yaw, 'lane')
            union_roads = ops.unary_union(road_segments)
            union_lanes = ops.unary_union(lanes)
            drivable_areas = ops.unary_union([union_roads, union_lanes])

            drivable_areas = split_collections(drivable_areas)

            # boundaries are defined as the contour of drivable areas
            boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

            return dict(
                divider=all_dividers, # List[LineString]
                ped_crossing=ped_crossing_lines, # List[LineString]
                boundary=boundaries, # List[LineString]
                drivable_area=drivable_areas, # List[Polygon],
            )

