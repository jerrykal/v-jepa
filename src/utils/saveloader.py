import sys
import torch
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# -- Load Function
def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def load_component(checkpoint, name, model, use_ddp):
    if model is not None and name in checkpoint:
        try:
            ckpt= checkpoint[name] if use_ddp else _strip_module_prefix(checkpoint[name])
            msg = model.load_state_dict(ckpt)
            logger.info(f'Loaded {name} with msg: {msg}')
        except Exception as e:
            logger.warning(f'Failed to load {name}: {e}')
    else:
        logger.warning(f'No "{name}" found in checkpoint.')
    return model