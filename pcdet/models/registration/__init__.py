#from .next_frame_registration import NextFrameRegistration
from .simple_reg import SimpleReg

__all__ = dict(
    #NextFrameRegistration=NextFrameRegistration,
    SimpleReg=SimpleReg,
)

def build_registration(model_cfg, runtime_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset
    )

    return model
