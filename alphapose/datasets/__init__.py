from .wfcoco_17 import Wfcoco17
from .wfcoco_17_det import Wfcoco17_det
from .coco_det import Mscoco_det
from .concat_dataset import ConcatDataset
from .custom import CustomDataset
from .mscoco import Mscoco
from .mpii import Mpii

__all__ = ['CustomDataset', 'Mscoco', 'Mscoco_det', 'Wfcoco17', 'Wfcoco17_det', 'Mpii', 'ConcatDataset']
