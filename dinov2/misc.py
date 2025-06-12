import os
import torch

def test_linear_head():
    dir_ckpt = "D:\\Code\\SelfSupervisedLearning\\dinov2-new\\dinov2-main\\weight\\Pretrained heads-Image classification"
    # fname_ckpt = "dinov2_vitg14_linear4_head.pth"
    fname_ckpt = "dinov2_vitg14_linear_head.pth"

    state_dict = torch.load(os.path.join(dir_ckpt, fname_ckpt), map_location="cpu")

    for k, v in state_dict.items():
        print(k, v.shape)

    return


if __name__ == '__main__':
    test_linear_head()