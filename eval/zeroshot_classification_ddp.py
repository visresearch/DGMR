"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from contextlib import suppress

import torch
import torch.nn.functional as F
from torch.distributed import get_rank, all_gather, get_world_size

from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm

from clip.eva_clip.model import CLIP, CustomCLIP
from intern_vit.clip_benchmark.clip_benchmark.models.internvl_c_pytorch.internvl_c import InternVL_C
def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True, cupl=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.


    model:
        CLIP-like model with `encode_text`

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    classnames: list of str
        name of classes

    templates: list of str
        templates to use.

    Returns
    -------

    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if cupl:
                texts = templates[classname]
            else:
                texts = [template.format(c=classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.

    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.

    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies

    Returns
    -------

    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def all_gather_tensor(tensor_rank):
    world_size = get_world_size()

    tensor_all_ranks = torch.empty(
        world_size,
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )

    tensor_list = list(tensor_all_ranks.unbind(0))

    all_gather(tensor_list, tensor_rank.contiguous())

    return tensor_all_ranks.flatten(end_dim=1)


@torch.inference_mode()
def run_classification(model, classifier, dataloader, device, amp=True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`

    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`

    dataloader: torch.utils.data.Dataloader

    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []

    niter_per_epoch = len(dataloader)
    dataloader_iter = iter(dataloader)

    images, target = next(dataloader_iter)

    if amp:
        images, target = images.half(), target.half()

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # with torch.no_grad():
    # for images, target in tqdm(dataloader):
    # for images, target in dataloader:
    for i in tqdm(range(niter_per_epoch)):

        # images = images.to(device)
        # target = target.to(device)
        if i < niter_per_epoch - 1:
            _images, _target = next(dataloader_iter)
            _images, _target = _images.half(), _target.half()
            _images = _images.to(device, non_blocking=True)
            _target = _target.to(device, non_blocking=True)

        with autocast():
            if hasattr(model, 'visual') or isinstance(model, InternVL_C):
                image_features = model.encode_image(images)
            else:
                image_features = model(images)
                
            image_features = F.normalize(image_features, dim=-1)
            logits = 100. * image_features @ classifier

        # true.append(target.cpu())
        # pred.append(logits.float().cpu())
        true.append(target)
        pred.append(logits)

        if i < niter_per_epoch - 1:
            images, target = _images, _target

    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)

    pred = all_gather_tensor(pred)
    true = all_gather_tensor(true)

    return pred, true


def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes

    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes

    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is
    the number of classes.

    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, cupl=False,
             save_clf=None, load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names

    templates: list of str
        templates to use for zero-shot classification

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    rank = get_rank()

    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, cupl=cupl)
        # del model.text
        if hasattr(model, "text"):
            model.text = torch.nn.Identity()
        elif hasattr(model, "transformer"):
            model.transformer = torch.nn.Identity()

    if rank == 0 and save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
    is_multilabel = (len(target.shape) == 2)

    if rank == 0:
        logits, target = logits.float().cpu(), target.cpu()
        if is_multilabel:
            if verbose:
                print('Detected a multi-label classification dataset')
            # Multiple labels per image, multiple classes on the dataset
            ap_per_class = average_precision_per_class(logits, target)
            if verbose:
                for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                    print(f'Class: {class_name}, AveragePrecision: {ap}')
            return {'mean_average_precision': ap_per_class.mean().item()}
        else:
            # Single label per image, multiple classes on the dataset
            # just compute accuracy and mean_per_class_recall

            pred = logits.argmax(axis=1)
            # measure accuracy
            if len(dataloader.dataset.classes) >= 5:
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            else:
                acc1, = accuracy(logits, target, topk=(1,))
                acc5 = float('nan')
            mean_per_class_recall = balanced_accuracy_score(target, pred)
            if verbose:
                print(classification_report(target, pred, digits=3))
            return {'acc1': acc1, 'acc5': acc5, 'mean_per_class_recall': float(mean_per_class_recall)}
