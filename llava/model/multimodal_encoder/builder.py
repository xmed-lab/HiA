import os
from .clip_encoder import CLIPVisionTower

from .clip_encoder import CLIPVisionTower
from .steam import SpatialPriorModule

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    print(is_absolute_path_exists, vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_steam_tower(resnet_path='..', **kwargs):
    return SpatialPriorModule(resnet_path)