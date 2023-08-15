import os
import SharedArray as SA
import joblib
import gc

SIZE = {
    'waymo_seg_with_r2_top_training.point': 23691,
    'waymo_seg_with_r2_top_training.rgb': 23691,
    'waymo_seg_with_r2_top_training.label': 23691,
    'waymo_seg_with_r2_top_training.instance': 23691,
    'waymo_seg_with_r2_top_training.box_ladn': 23691,
    'waymo_seg_with_r2_top_training.top_lidar_origin': 23691,
    'waymo_seg_with_r2_top_training.db_point_feat_label': 2863660,
    'waymo_seg_with_r2_top_toy_training.point': 237,
    'waymo_seg_with_r2_top_toy_training.rgb': 237,
    'waymo_seg_with_r2_top_toy_training.label': 237,
    'waymo_seg_with_r2_top_toy_training.instance': 237,
    'waymo_seg_with_r2_top_toy_training.box_ladn': 237,
    'waymo_seg_with_r2_top_toy_training.top_lidar_origin': 237,
    'waymo_seg_with_r2_top_toy_training.db_point_feat_label': 28637,
    'waymo_seg_with_r2_top_validation.point': 5976,
    'waymo_seg_with_r2_top_validation.rgb': 5976,
    'waymo_seg_with_r2_top_validation.label': 5976,
    'waymo_seg_with_r2_top_validation.instance': 5976,
    'waymo_seg_with_r2_top_validation.box_ladn': 5976,
    'waymo_seg_with_r2_top_validation.top_lidar_origin': 5976,
    'waymo_seg_with_r2_top_toy_validation.point': 60,
    'waymo_seg_with_r2_top_toy_validation.rgb': 60,
    'waymo_seg_with_r2_top_toy_validation.label': 60,
    'waymo_seg_with_r2_top_toy_validation.instance': 60,
    'waymo_seg_with_r2_top_toy_validation.box_ladn': 60,
    'waymo_seg_with_r2_top_toy_validation.top_lidar_origin': 60,
}

ALIAS = {
    'waymo_seg_with_r2_top_training': 'waymo',
    'waymo_seg_with_r2_top_validation': 'waymo',
    'waymo_seg_with_r2_top_toy_training': 'waymo',
    'waymo_seg_with_r2_top_toy_validation': 'waymo',
    'waymo_seg_with_r2_top_training.point': 'waymo.point',
    'waymo_seg_with_r2_top_training.rgb': 'waymo.rgb',
    'waymo_seg_with_r2_top_training.label': 'waymo.label',
    'waymo_seg_with_r2_top_training.instance': 'waymo.instance',
    'waymo_seg_with_r2_top_training.box_ladn': 'waymo.box_ladn',
    'waymo_seg_with_r2_top_training.top_lidar_origin': 'waymo.top_lidar_origin',
    'waymo_seg_with_r2_top_training.db_point_feat_label': 'waymo.db_point_feat_label',
    'waymo_seg_with_r2_top_toy_training.point': 'waymo.point',
    'waymo_seg_with_r2_top_toy_training.rgb': 'waymo.rgb',
    'waymo_seg_with_r2_top_toy_training.label': 'waymo.label',
    'waymo_seg_with_r2_top_toy_training.instance': 'waymo.instance',
    'waymo_seg_with_r2_top_toy_training.box_ladn': 'waymo.box_ladn',
    'waymo_seg_with_r2_top_toy_training.top_lidar_origin': 'waymo.top_lidar_origin',
    'waymo_seg_with_r2_top_toy_training.db_point_feat_label': 'waymo.db_point_feat_label',
    'waymo_seg_with_r2_top_validation.point': 'waymo.point',
    'waymo_seg_with_r2_top_validation.rgb': 'waymo.rgb',
    'waymo_seg_with_r2_top_validation.label': 'waymo.label',
    'waymo_seg_with_r2_top_validation.instance': 'waymo.instance',
    'waymo_seg_with_r2_top_validation.top_lidar_origin': 'waymo.top_lidar_origin',
    'waymo_seg_with_r2_top_validation.box_ladn': 'waymo.box_ladn',
    'waymo_seg_with_r2_top_toy_validation.point': 'waymo.point',
    'waymo_seg_with_r2_top_toy_validation.rgb': 'waymo.rgb',
    'waymo_seg_with_r2_top_toy_validation.label': 'waymo.label',
    'waymo_seg_with_r2_top_toy_validation.instance': 'waymo.instance',
    'waymo_seg_with_r2_top_toy_validation.top_lidar_origin': 'waymo.top_lidar_origin',
    'waymo_seg_with_r2_top_toy_validation.box_ladn': 'waymo.box_ladn',
}

OFFSET = {
    'waymo_seg_with_r2_top_training.point': 0,
    'waymo_seg_with_r2_top_training.rgb': 0,
    'waymo_seg_with_r2_top_training.label': 0,
    'waymo_seg_with_r2_top_training.instance': 0,
    'waymo_seg_with_r2_top_training.box_ladn': 0,
    'waymo_seg_with_r2_top_training.top_lidar_origin': 0,
    'waymo_seg_with_r2_top_training.db_point_feat_label': 0,
    'waymo_seg_with_r2_top_toy_training.point': 0,
    'waymo_seg_with_r2_top_toy_training.rgb': 0,
    'waymo_seg_with_r2_top_toy_training.label': 0,
    'waymo_seg_with_r2_top_toy_training.instance': 0,
    'waymo_seg_with_r2_top_toy_training.box_ladn': 0,
    'waymo_seg_with_r2_top_toy_training.top_lidar_origin': 0,
    'waymo_seg_with_r2_top_toy_training.db_point_feat_label': 0,
    'waymo_seg_with_r2_top_validation.point': 23691,
    'waymo_seg_with_r2_top_validation.rgb': 23691,
    'waymo_seg_with_r2_top_validation.label': 23691,
    'waymo_seg_with_r2_top_validation.instance': 23691,
    'waymo_seg_with_r2_top_validation.top_lidar_origin': 23691,
    'waymo_seg_with_r2_top_validation.box_ladn': 23691,
    'waymo_seg_with_r2_top_toy_validation.point': 237,
    'waymo_seg_with_r2_top_toy_validation.rgb': 237,
    'waymo_seg_with_r2_top_toy_validation.label': 237,
    'waymo_seg_with_r2_top_toy_validation.instance': 237,
    'waymo_seg_with_r2_top_toy_validation.top_lidar_origin': 237,
    'waymo_seg_with_r2_top_toy_validation.box_ladn': 237,
}

def _allocate_data(data_tag, split, data_type, root_path):
    data_name = f'{data_tag}_{split}.{data_type}'
    if data_name in OFFSET:
        offset = OFFSET[data_name]
    else:
        offset = 0
    original_data_name = data_name
    num_samples = SIZE[data_name]
    if data_name in ALIAS:
        data_name = ALIAS[data_name]
        # allocate data to shm:///
        path_template = data_name.replace('.', '_{}.')
    else:
        path_template = f'{data_tag}_{split}_{{}}.{data_type}'

    max_sample_id = num_samples + offset - 1
    #if len(glob.glob(f'/dev/shm/{data_tag}_{split}_*.{data_type}')) < num_samples:
    if not os.path.exists('/dev/shm/'+path_template.format(max_sample_id)):
        #print(data_name, original_data_name, max_sample_id)
        #if not (data_name in ['waymo.bbox', 'waymo.db_point_feat_label', 'waymo.instance', 'waymo.top_lidar_origin']):
        #    assert False
        filename = root_path / original_data_name
        data_list = joblib.load(filename)
        for idx, data in enumerate(data_list):
            if not os.path.exists('/dev/shm/'+path_template.format(idx+offset)):
                x = SA.create("shm://"+path_template.format(idx+offset), data.shape, dtype=data.dtype)
                x[...] = data[...]
                x.flags.writeable = False
        del data_list
        gc.collect()
