import torch
from torch import nn

from timm.models.deit import VisionTransformerDistilled
from timm.models.swin_transformer import SwinTransformer
from clip.eva_clip.eva_vit_model import EVAVisionTransformer
from intern_vit.model import InternViT6B
from clip.open_clip.transformer import VisionTransformer as openclip_vit
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    
    def forward(self, x):
        if isinstance(self.model, VisionTransformerDistilled):
            feat = self.model.forward_features(x)
            output = self.model.forward_head(feat)

            return output, feat[:, 2:]
        elif isinstance(self.model, SwinTransformer):
            feat = self.model.forward_features(x)
            output = self.model.forward_head(feat)

            return output, feat
        elif isinstance(self.model, EVAVisionTransformer):
            feat = self.model.forward_features(x, return_all_features=True)
            feat = self.model.norm(feat)
            output = self.model.head(feat[:, 0])

            return output, feat[:, 1:]
        elif isinstance(self.model, InternViT6B):
            x = self.model.forward_features(x)
            output = self.model.forward_head(x)
            return output, x[:, 1:]
        elif isinstance(self.model, openclip_vit):
            feat, output = self.model(x, output_tokens=True)
            return feat, output
        else:
            RuntimeError(f'Unrecognize model: {self.model}')


class ModelHeadWrapper1(torch.nn.Module):
    def __init__(self, model, expand_ratio=1):
        super().__init__()
        self.model = model

        if isinstance(self.model, VisionTransformerDistilled):
            embed_dim = self.model.head.in_features
            dtype = self.model.head.weight.dtype
            self.model.head = nn.Linear(embed_dim, expand_ratio * embed_dim, dtype=dtype)
            self.model.head_dist = nn.Identity()
        elif isinstance(self.model, SwinTransformer):
            embed_dim = self.model.head.fc.in_features
            dtype = self.model.head.fc.weight.dtype
            self.model.head = nn.Linear(embed_dim, expand_ratio * embed_dim, dtype=dtype)
        elif isinstance(self.model, EVAVisionTransformer):
            embed_dim = self.model.head.in_features
            dtype = self.model.head.weight.dtype
            self.model.head = nn.Linear(embed_dim, expand_ratio * embed_dim, dtype=dtype)
    
    
    def forward(self, x):
        if isinstance(self.model, VisionTransformerDistilled):
            feat = self.model.forward_features(x)
            out_feat = self.model.head(feat)
            return out_feat[:, :2], out_feat[:, 2:]

        elif isinstance(self.model, SwinTransformer):
            feat = self.model.forward_features(x)
            output = self.model.head(feat)
            return output

        elif isinstance(self.model, EVAVisionTransformer):
            feat = self.model.forward_features(x, return_all_features=True)
            feat = self.model.norm(feat)
            output = self.model.head(feat)
            return output[:, 0], output[:, 1:]

        else:
            raise RuntimeError(f'Unrecognize model: {self.model}')


class ModelHeadWrapper(torch.nn.Module):
    def __init__(self, model, expand_ratio=1):
        super().__init__()
        self.model = model

        if isinstance(self.model, VisionTransformerDistilled):
            self.model.head = nn.Identity()
            self.model.head_dist = nn.Identity()
        elif isinstance(self.model, SwinTransformer) or \
                isinstance(self.model, EVAVisionTransformer):
            self.model.head = nn.Identity()
    
    def forward(self, x):
        if isinstance(self.model, VisionTransformerDistilled):
            feat = self.model.forward_features(x)
            return feat[:, :2], feat[:, 2:]

        elif isinstance(self.model, SwinTransformer):
            feat = self.model.forward_features(x)
            return feat

        elif isinstance(self.model, EVAVisionTransformer):
            feat = self.model.forward_features(x, return_all_features=True)
            return feat[:, 0], feat[:, 1:]

        else:
            raise RuntimeError(f'Unrecognize model: {self.model}')


def update_head_momentum(teacher, student, m):

    for param_t, param_s in zip(teacher.head.parameters(), student.head.parameters()):
        param_t.data.mul_(m).add_(param_s.data.mul_(1. - m))

    return

def intialize_student_head(teacher, student):
    for param_t, param_s in zip(teacher.head.parameters(), student.head.parameters()):
        param_s.data.copy_(param_t.data)
        # param_t.requires_grad = False

    return 