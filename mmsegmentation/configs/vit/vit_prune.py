_base_ = ["vit_deit-s16_upernet_8xb2-80k_ade20k-512x512.py"]

load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_80k_ade20k/upernet_deit-s16_512x512_80k_ade20k_20210624_095228-afc93ec2.pth"
dist_mult = 0.5  # Only using 4 instead of 8 GPUs
lr_ratio = 1 * dist_mult
model = dict(pretrained=load_from)
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW", lr=0.00006 * lr_ratio, betas=(0.9, 0.999), weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

prune_cfg = "./vit_b16_small.json"
custom_hooks = [
    dict(
        type="PruningHook",
        # In pruning process, you need set priority
        # as 'LOWEST' to insure the pruning_hook is excused
        # after optimizer_hook, in fintune process, you
        # should set it as 'HIGHEST' to insure it excused
        # before checkpoint_hook
        pruning=True,
        interval=50,
        priority="LOWEST",
        prune_cfg=prune_cfg,
        save_sparsity_thr=[0.4],
        continue_finetune=True,
    )
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=2000)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        # save_best="mIoU",
    ),
)
