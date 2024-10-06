"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re


from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BlipImageBaseProcessor:
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    
    def __call__(self, item):
        return self.transform(item)

# class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
#     def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):

#         super().__init__(mean=mean, std=std)

#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     image_size,
#                     scale=(min_scale, max_scale),
#                     interpolation=InterpolationMode.BICUBIC,
#                 ),
#                 transforms.ToTensor(),
#                 self.normalize,
#             ]
#         )

#     def __call__(self, item):
#         return self.transform(item)

#     @classmethod
#     def from_config(cls, hr_crop_size):
       

#         image_size =hr_crop_size
       

#         min_scale = cfg.get("min_scale", 0.5)
#         max_scale = cfg.get("max_scale", 1.0)

#         return cls(
#             image_size=image_size,
#             mean=mean,
#             std=std,
#             min_scale=min_scale,
#             max_scale=max_scale,
#         )


