# Comb, Prune, Distill: Towards Unified Pruning for Vision Model Compression

## Introduction

We propose a novel unified pruning framework Comb, Prune, Distill (CPD), which allows for model-agnostic and task-agnostic pruning. 
Our framework employs a combing step to resolve hierarchical layer-wise dependency issues, enabling architecture independence. The pruning framework adaptively removes parameters on the basis of Hessian-based importance scoring.

## Repository Structure
The directory structure is as follows:
```
.
└── mmsegmentation
    ├── configs
        └── vit
            └── vit_prune.py         # Example config for pruning ViT-DeiT-S16
    ├── mmseg
        ├── engine
            ├── hooks
                └── pruning_hook.py  # The pruning hook
    ├── prune                        # Parent directory containing all other pruning related files
        ├── pruners                  # Custom pruning functions for more complex network structures
        ├── tools                    # Additional pruning utils and tools
            └── gen_prune_cfg.py     # Script to generate the dependency structure for a given model
    └── vit_b16_small.json           # Dependency structure for pruning ViT-DeiT-S16, generated by gen_prune_cfg.py
```

## Generating the dependency structure
To generate the dependency structure needed for pruning, the user may run ```mmsegmentation/prune/tools/gen_prune_cfg.py``` as follows:
``` sh
python ./mmsegmentation/prune/tools/gen_prune_cfg.py configs/<your_architecture>/<your_config>.py --out <pruning_config>.json
```

## Pruning a model
By using the previously generated dependency structure, our pruning hook performs pruning & fine-tuning.

For this, the pruning hook needs to be attached to the training process.
An example for this can be found in ```mmsegmentation/configs/vit/vit_prune.py```

Essentially, the pruning hook must be added to the custom hooks:
``` python
custom_hooks = [
    dict(
        type="PruningHook",
        priority="LOWEST",
        interval=50,                # Specify the interval of training iterations between which the model is pruned
        prune_cfg="./mmsegmentation/vit_b16_small.json",        # Specify the dependency structure config file
        '''
        Specify sparsity thresholds after which the model should be saved in the corresponding work_dir.

        Note that the threshold should be in ascending order. 
        The last entry determines the threshold used for fine-tuning.
        '''
        save_sparsity_thr=[0.4], 
        continue_finetune=True,     # Controls if the model should be fine-tuned after pruning. Recommended.
    )
]
```
Now the model can be trained with the usual mmsegmentation training process, i.e.
``` sh
./mmsegmentation/tools/dist_train.sh configs/<your_architecture>/<your_pruning_config>.py <NUM_GPUS>
```