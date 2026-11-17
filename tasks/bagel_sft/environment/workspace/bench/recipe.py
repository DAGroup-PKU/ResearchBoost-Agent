"""The frozen BAGEL-7B-MoT fine-tuning recipe: upstream's fine-tuning command (TRAIN.md of ByteDance-Seed/Bagel @
a2fa77d) as the arguments of `train/pretrain_unified_navit.py`. The judge launches every run with exactly these
arguments; the tree under test cannot change them.

Upstream's command on 4 GPUs instead of 8: the per-GPU load is upstream's (10240 expected packed tokens per GPU, at
most 11520, samples up to 10240), so a step trains on 4 x 10240 tokens instead of 8 x 10240; FSDP shards over the 4
GPUs (`--num_shard 4`; upstream's default 8 is its GPU count). The released checkpoint initializes both the model and
its EMA (`--finetune_from_ema`), learning rate 2e-5 with upstream's default warmup (2000 steps) and constant schedule,
AdamW betas 0.9/0.95, eps 1e-15, EMA 0.9999, gradient clipping 1.0, global seed 4396, data seed 42, one data worker,
upstream's data configuration (data/configs/example.yaml, frozen as bench/data_config.yaml). The judge's own settings:
wandb offline (the evaluation has no network), a fresh results and checkpoint directory per run (no checkpoint is
reached within a judged run)."""
ENTRY = 'train/pretrain_unified_navit.py'
WORLD_SIZE = 4
EXPECTED_TOKENS = 10240


def argv(model_path, data_config, results_dir):
    return [
        '--dataset_config_file', data_config, '--model_path', model_path, '--layer_module', 'Qwen2MoTDecoderLayer',
        '--max_latent_size', '64', '--resume_from', model_path, '--finetune_from_hf', 'True', '--auto_resume', 'True',
        '--resume_model_only', 'True', '--finetune_from_ema', 'True', '--log_every', '1', '--lr', '2e-5',
        '--num_workers', '1', '--expected_num_tokens', str(EXPECTED_TOKENS), '--max_num_tokens', '11520',
        '--max_num_tokens_per_sample', '10240', '--num_shard', str(WORLD_SIZE),
        '--results_dir', results_dir, '--checkpoint_dir', results_dir + '/checkpoints', '--wandb_offline', 'True',
    ]
