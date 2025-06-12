import sys
from pathlib import Path

wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from eva_clip import create_model_and_transforms, get_tokenizer


def load_eva_clip(model_name: str, pretrained: str, \
                  cache_dir: str = None, device='cpu'):
    
    model, _, transform = create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir, force_custom_clip=True)
    model = model.half()
    model = model.to(device)

    tokenizer = get_tokenizer(model_name)

    return model, transform, tokenizer

