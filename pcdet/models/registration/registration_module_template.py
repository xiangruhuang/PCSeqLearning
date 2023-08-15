import os

import torch
import torch.nn as nn

from ...utils.spconv_utils import find_all_spconv_keys
from pcdet.models import registration
import pcdet.models.registration.preprocessors as preprocessor
from .. import visualizers
 
class RegistrationTemplate(nn.Module):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.runtime_cfg = runtime_cfg
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        
        self.scale = 1 if 'SCALE' not in model_cfg else model_cfg.pop('SCALE')

        self.module_topology = [
            'preprocessors', 'registration', 'visualizer',
        ]

    def update_ema(self):
        pass

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_point_features': self.dataset.num_point_features,
            'max_num_points': self.dataset.max_num_points*self.dataset.num_sweeps*2,
            'scale': self.scale,
        }
        if self.model_cfg.get('VISUALIZER', None) is not None:
            model_info_dict['visualize'] = True

        model_info_dict.update(self.dataset.runtime_cfg)
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict,
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_visualizer(self, model_info_dict):
        if self.model_cfg.get('VISUALIZER', None) is None:
            return None, model_info_dict

        visualizer_module = visualizers.__all__[self.model_cfg.VISUALIZER.NAME](
            model_cfg=self.model_cfg.VISUALIZER,
            runtime_cfg=model_info_dict,
        )
        model_info_dict['module_list'].append(visualizer_module)
        return visualizer_module, model_info_dict

    def build_preprocessors(self, model_info_dict):
        if self.model_cfg.get('PREPROCESSORS', None) is None:
            return None, model_info_dict
        
        preprocessors = nn.ModuleList()
        for PREPROCESSOR in self.model_cfg.PREPROCESSORS:
            preprocessor_module = preprocessor.__all__[PREPROCESSOR.NAME](
                runtime_cfg=model_info_dict,
                model_cfg=PREPROCESSOR,
            )
            preprocessors.append(preprocessor_module)
        model_info_dict['module_list'].append(preprocessors)
        return preprocessors, model_info_dict
    
    def build_registration(self, model_info_dict):
        if self.model_cfg.get('REGISTRATION', None) is None:
            return None, model_info_dict

        reg_module = registration.__all__[self.model_cfg.REGISTRATION.NAME](
            runtime_cfg=model_info_dict,
            model_cfg=self.model_cfg.REGISTRATION,
        )
        model_info_dict['module_list'].append(reg_module)
        return reg_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state
    
    def _load_and_ema_state_dict(self, model_state_disk, momentum, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val * (1-momentum) + state_dict[key] * momentum
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, ema=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        
        if ema:
            model_state_ema = model_state_disk.pop('ema')
            model_state_disk.update(model_state_ema)

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s in state dict: %s, no key matching' % (key, str(state_dict[key].shape)))
            else:
                logger.info('Updated weight %s in state dict from file: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))
    
    def load_ema_params_from_files(self, filenames, logger, momentum, to_cpu=False):
        for i, filename in enumerate(filenames):
            if not os.path.isfile(filename):
                raise FileNotFoundError

            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
            loc_type = torch.device('cpu') if to_cpu else None
            checkpoint = torch.load(filename, map_location=loc_type)
            model_state_disk = checkpoint['model_state']

            version = checkpoint.get("version", None)
            if version is not None:
                logger.info('==> Checkpoint trained from version: %s' % version)

            if i == 0:
                state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
            else:
                state_dict, update_model_state = self._load_and_ema_state_dict(model_state_disk, momentum, strict=False)

            for key in state_dict:
                if key not in update_model_state:
                    logger.info('Not updated weight %s in state dict: %s, no key matching' % (key, str(state_dict[key].shape)))
                else:
                    logger.info('Updated weight %s in state dict from file: %s' % (key, str(state_dict[key].shape)))

            logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
