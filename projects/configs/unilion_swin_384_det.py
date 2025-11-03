plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-54.0, -54.0, -3.0, 54.0, 54.0, 5.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_class_names = [
    'ped_crossing', 'divider', 'boundary',
]
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

seg_map_classes = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider'
]

out_size_factor = 2

dataset_type = 'NuScenes3DDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

voxel_size = [0.3, 0.3, 0.25]
grid_size = [360, 360, 32]
lion_dim = 128

bda_aug_conf = dict(
    rot_lim=(-22.5 * 2, 22.5 * 2),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=[0.5, 0.5, 0.5]
)

grid_config = {
    'x': [-54.0, 54.0, 0.3],
    'y': [-54.0, 54.0, 0.3],
    'z': [-3, 5, 0.25],
    'depth': [1.0, 60.0, 0.5],
}

data_config = {
    'cams': [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (384, 1056),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

tasks = dict(
    det=True,
    occ=False,
    map=False,
    motion=False,
    planning=False,
)

model = dict(
    type='UniLION',
    use_grid_mask=True,
    tasks=tasks,
    fusion=True,
    with_depth_loss=True,
    loss_depth_weight=0.1,
    voxel_size=voxel_size,
    pc_range=point_cloud_range,
    img_backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(
            type='Pretrained',
            checkpoint="ckpts/swint_nuimg_pretrained.pth",
            prefix='backbone.'),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=False,
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_map2bev=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        k=4,
        downsample=8,
        out_channels=lion_dim,
        bev_out_channels=lion_dim * 2,
        out_voxel=True,
        out_bev=False,
        with_depth_w=True,
        with_cp=False,
        with_depth_from_lidar=True),
    pts_voxel_encoder=dict(
        type='DynamicPillarVFE3D',
        with_distance=False,
        use_absolute_xyz=True,
        use_norm=True,
        num_filters=[lion_dim, lion_dim],
        num_point_features=5,
        voxel_size=voxel_size,
        grid_size=grid_size,
        point_cloud_range=point_cloud_range),
    pts_backbone=dict(
        type='Lion3DBackbone',
        grid_size=grid_size,
        layer_dim=[lion_dim, lion_dim, lion_dim, lion_dim],
        num_layers=4,
        depths=[2, 2, 2, 2],
        layer_down_scales=[[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]],
                           [[2, 2, 2], [2, 2, 2]]],
        window_shape=[[13, 13, 32], [13, 13, 16], [13, 13, 8], [13, 13, 4]],
        group_size=[4096, 2048, 1024, 512],
        direction=['x', 'y'],
        diffusion=True,
        shift=True,
        diff_scale=0.2,
        linear_operator=dict(
            type='Mamba',
            cfg=dict(
                d_state=16,
                d_conv=4,
                expand=2,
                drop_path=0.2))),
    map2bev=dict(
        type='HeightCompression',
        output_shape=[360, 360, 2],
        num_bev_feats=lion_dim * 2),
    bev_backbone=dict(
        type='ResSECOND',
        in_channels=lion_dim * 2,
        out_channels=[128, 128, 256],
        blocks_nums=[1, 2, 2],
        layer_strides=[1, 2, 2]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=False),
    det_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=sum([128, 128, 128]),
        hidden_channel=128,
        num_classes=10,
        nms_kernel_size=3,
        iou_rescore_weight=0.5,
        bn_momentum=0.1,
        num_decoder_layers=1,
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2], iou=[1, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=out_size_factor,
            voxel_size=[0.3, 0.3],
            code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss', reduction='mean', loss_weight=0.25),
        train_cfg=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            voxel_size=[0.3, 0.3],
            pc_range=[-54.0, -54.0],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)),
    train_cfg=dict(
        dynamic=False,
        loss_det_weight=1.0,
        loss_occ_weight=0.0,
        loss_map_weight=0.0,
    ))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='ToEgo'),
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(type='LoadMultiTaskAnnotations3D'),
    dict(
        type='VectorizeMap',
        roi_size=(60, 30),
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True,
    ),
    dict(
        type='UnifiedObjectSample',
        sample_2d=True,
        mixup_rate=0.5,
        db_sampler=dict(
            type='UnifiedDataBaseSampler',
            data_root=data_root,
            info_path=data_root + 'nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ))),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='MultiTaskFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points',
                                 'img',
                                 'img_inputs',
                                 'gt_depth',
                                 'gt_bboxes_3d',
                                 'gt_labels_3d',
                                 'gt_occ',
                                 'mask_camera',
                                 'gt_map_labels',
                                 'gt_map_pts',
                                 'gt_bev_masks',
                                 'ego_status',
                                 'gt_agent_fut_trajs',
                                 'gt_agent_fut_masks',
                                 'gt_ego_fut_trajs',
                                 'gt_ego_fut_masks',
                                 'gt_ego_fut_cmd'],
         meta_keys=('filename', 'lidar2img'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='ToEgo'),
    dict(
        type='PrepareImageInputs',
        is_train=False,
        data_config=data_config),
    dict(type='LoadMultiTaskAnnotations3D'),
    dict(
        type='VectorizeMap',
        roi_size=(60, 30),
        simplify=True,
        permute=False,
        normalize=False
    ),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='MultiTaskFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points',
                                         'img',
                                         'img_inputs',
                                         'gt_depth',
                                         # 'vectors',
                                         'gt_bev_masks',
                                         'ego_status',
                                         'fut_boxes',
                                         'gt_ego_fut_trajs',
                                         'gt_ego_fut_masks',
                                         'gt_ego_fut_cmd'],
                 meta_keys=('filename', 'token'))
        ])
]

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    occ_root='occ3d',
    classes=class_names,
    map_classes=map_class_names,
    seg_map_config=dict(
        seg_map_classes=seg_map_classes,
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5]
    ),
    bda_aug_conf=bda_aug_conf,
    occ_class=occ_class_names,
    modality=input_modality,
    version="v1.0-trainval",
    keep_consistent_seq_aug=True,
    sequences_split_num=-1
)
eval_config = dict(
    **data_basic_config,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    pipeline=test_pipeline,
    test_mode=True,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        **data_basic_config,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        load_interval=1,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        **data_basic_config,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        test_mode=True,
        eval_config=eval_config),
    test=dict(
        **data_basic_config,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        test_mode=True,
        eval_config=eval_config))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)
        }),
    weight_decay=0.05
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 36
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)

eval_mode = dict(
    with_det=tasks['det'],
    with_tracking=False,
    with_occ=tasks['occ'],
    with_map=tasks['map'],
    with_motion=tasks['motion'],
    with_planning=tasks['planning'],
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=total_epochs,
    eval_mode=eval_mode,
)

custom_hooks = [dict(type='Fading', fade_epoch=32)]
