# ------------------------------------- 32-39_768 with training -------------------------------------
python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_768/dinov2_vitg14_pretrain_prune.pth \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1024

# kNN: 84.27
# {"best_classifier": {"name": "classifier_1_blocks_avgpool_True_lr_0_06400", "accuracy": 0.8642799854278564}}

# ------------------------------------- original only test -------------------------------------
python dinov2/run/eval/linear_eval_only_test.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/dinov2_vitg14_pretrain.pth \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --fnames-ckpt-classifiers /public/scccse/model_weight/dinov2/dinov2_vitg14_linear_head.pth /public/scccse/model_weight/dinov2/dinov2_vitg14_linear4_head.pth \
    --batch-size 1024

# kNN: 83.5
# {"best_classifier": {"name": "classifier_4_blocks_avgpool_True", "accuracy": 0.8656799793243408}}


python dinov2/run/eval/linear_eval_only_test.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+24-31_1024/dinov2_vitg14_pretrain_prune.pth \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --fnames-ckpt-classifiers /public/scccse/model_weight/dinov2/dinov2_vitg14_linear_head.pth /public/scccse/model_weight/dinov2/dinov2_vitg14_linear4_head.pth \
    --batch-size 1024

# kNN: 81.26
# {"best_classifier": {"name": "classifier_4_blocks_avgpool_True", "accuracy": 0.6551799774169922}}

python dinov2/run/eval/linear_eval_only_test.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_768/dinov2_vitg14_pretrain_prune.pth \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --fnames-ckpt-classifiers /public/scccse/model_weight/dinov2/dinov2_vitg14_linear_head.pth /public/scccse/model_weight/dinov2/dinov2_vitg14_linear4_head.pth \
    --batch-size 1024

# kNN: 84.27
# {"best_classifier": {"name": "classifier_4_blocks_avgpool_True", "accuracy": 0.8226400017738342}}


# ------------------------------------- svd prune -------------------------------------
python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2/[83.37]dinov2_2024-08-24_22-18-49/ckpt/student_vit_giant2_10.pth \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --output-dir ./out/dinov2/[83.37]dinov2_2024-08-24_22-18-49/ckpt/linear \
    --epoch-length 10000 \
    --batch-size 128

