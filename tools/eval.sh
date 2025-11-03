#!/usr/bin/env bash


# tools/dist_test.sh projects/configs/mm_lion_fusion_swin_384_perception_2_wd.py ckpts/mmlion_fusion_swin_perception.pth 8 --eval mAP

# tools/dist_test.sh projects/configs/mm_lion_fusion_swin_384_perception_5_wd.py ckpts/mmlion_fusion_swin_perception.pth 8 --eval mAP

# tools/dist_test.sh projects/configs/mm_lion_fusion_swin_384_perception_wd.py ckpts/mmlion_fusion_swin_perception.pth 8 --eval mAP

# tools/dist_test.sh projects/configs/mm_lion_seq_fusion_swin_384_perception_5.py ckpts/mmlion_lidar_seq_perception.pth 8 --eval mAP

tools/dist_test.sh projects/configs/mm_lion_fusion_r50_256_perception_10cm.py ckpts/mm_lion_fusion_det_occ_and_map_dw_01_lr_01x_shared_ep24_occ1_map01_resnet50/latest.pth 8 --eval mAP

tools/dist_test.sh projects/configs/mm_lion_fusion_r50_256_perception_15cm.py ckpts/mm_lion_fusion_det_occ_and_map_dw_01_lr_01x_shared_ep24_occ1_map01_resnet50/latest.pth 8 --eval mAP

tools/dist_test.sh projects/configs/mm_lion_fusion_r50_256_perception_20cm.py ckpts/mm_lion_fusion_det_occ_and_map_dw_01_lr_01x_shared_ep24_occ1_map01_resnet50/latest.pth 8 --eval mAP

tools/dist_test.sh projects/configs/mm_lion_fusion_r50_256_perception_30cm.py ckpts/mm_lion_fusion_det_occ_and_map_dw_01_lr_01x_shared_ep24_occ1_map01_resnet50/latest.pth 8 --eval mAP

tools/dist_test.sh projects/configs/mm_lion_fusion_r50_256_perception_40cm.py ckpts/mm_lion_fusion_det_occ_and_map_dw_01_lr_01x_shared_ep24_occ1_map01_resnet50/latest.pth 8 --eval mAP

tools/dist_test.sh projects/configs/mm_lion_fusion_r50_256_perception_50cm.py ckpts/mm_lion_fusion_det_occ_and_map_dw_01_lr_01x_shared_ep24_occ1_map01_resnet50/latest.pth 8 --eval mAP