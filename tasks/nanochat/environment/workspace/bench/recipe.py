"""The frozen nanochat D24 pretraining recipe (upstream karpathy/nanochat @ 92d63d4), as the arguments of
`python -m scripts.base_train`. The judge launches every run with exactly these arguments; the tree under test cannot
change them. Depth 24 (model dim 1536, 12 heads of 128, ~1.38B parameters), sliding-window pattern SSSL, sequence
2048, 16 sequences per GPU as upstream, 262,144 tokens per optimizer update (4 accumulation steps on 2 GPUs: upstream's
1,048,576 on 8 GPUs scaled by 2/8), tensorwise FP8
linear layers, data:param ratio 8 (the training horizon, which sets the learning-rate warmdown, the Muon momentum and
the weight-decay schedules), every evaluation, sampling and checkpoint disabled."""
ENTRY = 'scripts.base_train'
WORLD_SIZE = 2
DEVICE_BATCH = 16
SEQ_LEN = 2048
TOTAL_BATCH = DEVICE_BATCH * SEQ_LEN * 4 * WORLD_SIZE      # 262,144: 4 accumulation steps


def argv():
    return [
        '--depth=24', '--aspect-ratio=64', '--head-dim=128', '--max-seq-len=%d' % SEQ_LEN, '--window-pattern=SSSL',
        '--device-batch-size=%d' % DEVICE_BATCH, '--total-batch-size=%d' % TOTAL_BATCH,
        '--target-param-data-ratio=8', '--fp8', '--fp8-recipe=tensorwise', '--num-iterations=-1', '--run=dummy',
        '--eval-every=-1', '--core-metric-every=-1', '--sample-every=-1', '--save-every=-1', '--model-tag=d24',
    ]
