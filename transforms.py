
import torchvision.transforms as transforms

from clip.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from timm.data.transforms import RandomResizedCropAndInterpolation, ToTensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import io
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_transforms_dinov2(input_size, min_crop):

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                input_size, scale=(min_crop, 1.0), \
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            # transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        ])

    return train_transform


def get_transforms_eva(input_size, min_crop):

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                input_size, scale=(min_crop, 1.0), \
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        ])

    return train_transform


def get_transforms_deit(input_size, min_crop=0.08):
    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # train_transform = transforms.Compose([
    #         RandomResizedCropAndInterpolation(
    #             input_size, scale=(min_crop, 1.0), interpolation='bicubic'),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomChoice([transforms.RandomGrayscale(p=1.0),
    #                                 transforms.RandomSolarize(threshold=128, p=1.0),
    #                                 GaussianBlur(p=1.0)]),
    #         transforms.ColorJitter(0.3, 0.3, 0.3),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean, std=std),
    #     ])

    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )

    
    return train_transform


def get_transforms_swin(input_size, min_crop):

    transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
    )

    return transform
    
def get_transforms_internVIT(config, is_train=True):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.Resize(config.data.input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.data.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    return transform

import numpy as np
from timm.data.mixup import cutmix_bbox_and_lam

class Mixup:
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, \
            prob=1.0, switch_prob=0.5, correct_lam=True):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)


    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix


    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam


    def __call__(self, x):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        self._mix_batch(x)
        return x
