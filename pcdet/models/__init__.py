from collections import namedtuple

import numpy as np
import torch
import re

from .detectors import build_detector
from .registration import build_registration

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

def build_network(model_cfg, cfg, dataset):
    import pcdet.models.detectors as detectors
    import pcdet.models.registration as registration

    builder_dict = {}
    for name in dir(detectors):
        if name[:1].isupper():
            builder_dict[name] = build_detector

    for name in dir(registration):
        if name[:1].isupper():
            builder_dict[name] = build_registration

    builder = builder_dict[model_cfg.NAME]

    model = builder(model_cfg=model_cfg, runtime_cfg=cfg, dataset=dataset)

    freezed_modules = cfg.MODEL.get('FREEZED_MODULES', None)
    if freezed_modules is not None:
        for name, param in model.named_parameters():
            for module_regex in freezed_modules:
                if re.match(module_regex, name) is not None:
                    print(f"FREEZING {name}")
                    param.requires_grad = False

    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'obj_ids']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
