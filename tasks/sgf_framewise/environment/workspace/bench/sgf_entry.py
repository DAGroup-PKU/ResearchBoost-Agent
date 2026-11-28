"""Run the pinned SGF Trainer.train under torchrun with judge-owned observers."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import fingerprint, read_json, require


def tensor_meta(value):
    return [list(value.shape), str(value.dtype).removeprefix("torch.")]


def tensor_hash(value):
    raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(raw).hexdigest()


def named_metadata(module):
    return [[name, tensor_meta(value), value.requires_grad] for name, value in module.named_parameters()]


def state_hash(module):
    return fingerprint([[name, tensor_meta(value), tensor_hash(value)]
                        for name, value in (*module.named_parameters(), *module.named_buffers())])


def optimizer_config(optimizer):
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def optimizer_metadata(optimizer):
    return fingerprint([
        {key: tensor_meta(value) if isinstance(value, torch.Tensor) else type(value).__name__
         for key, value in optimizer.state.get(param, {}).items()}
        for group in optimizer.param_groups for param in group["params"]
    ])


def check_generator_context(generator, config):
    fields = ("local_attn_size", "sink_size")
    require(all(getattr(generator, field, None) == config.model_kwargs[field] for field in fields),
            "Generator context does not match the frozen recipe")


def with_random_trace(operation, target, on_sample=None):
    originals = {}

    def traced(name, function):
        def sample(*args, **kwargs):
            value = function(*args, **kwargs)
            target.append([name, tensor_meta(value), tensor_hash(value)])
            if on_sample is not None:
                on_sample(name, value)
            return value
        return sample

    try:
        for name in ("rand", "rand_like", "randn", "randn_like", "randint", "randint_like"):
            originals[name] = getattr(torch, name)
            setattr(torch, name, traced(name, originals[name]))
        return operation()
    finally:
        for name, function in originals.items():
            setattr(torch, name, function)


class Rollout:
    def __init__(self, pipeline, exact, config, noise, require_grad):
        self.pipeline = pipeline
        self.exact = exact
        self.config = config
        self.noise = noise
        self.require_grad = require_grad
        self.blocks = exact["pass1_blocks_per_rollout"]
        self.stages = exact["pass1_denoising_stages_per_block"]
        self.calls, self.outputs, self.context, self.inputs, self.transitions = [], [], [], [], []
        self.exit = None
        self.block_noise = None
        require(list(noise.shape) == list(config.image_or_video_shape), "Rollout latent shape mismatch")
        require(all(getattr(pipeline, key) == config[key] for key in (
            "self_gradient_forcing_cache_mode", "per_rank_exit_step", "self_gradient_forcing_match_context",
            "num_frame_per_block", "context_noise",
        )), "Runtime forcing recipe mismatch")

    def random_sample(self, name, value):
        if self.exit is None and name == "randint":
            require(value.numel() == 1, "Expected one rank-local sampled exit")
            self.exit = value.item()
            require(0 <= self.exit < self.stages, "Invalid sampled exit")

    def timestep(self, tensor, value):
        require(bool((tensor == value).all()), "Unexpected rollout timestep")
        return [tensor.reshape(-1)[0].item(), tensor_meta(tensor)]

    def same(self, actual, expected):
        return actual.shape == expected.shape and actual.dtype == expected.dtype and (
            actual.data_ptr() == expected.data_ptr() and actual.stride() == expected.stride()
            or torch.equal(actual, expected)
        )

    def forward(self, module, args, kwargs, output):
        require(not args and self.exit is not None, "Missing rank-local rollout sampling evidence")
        stride = self.stages + 1
        block, stage = divmod(len(self.calls), stride)
        latent = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        if block == self.blocks:
            clean, aug = kwargs["clean_x"], kwargs["aug_t"]
            require(stage == 0 and not kwargs.get("kv_cache") and not clean.requires_grad,
                    "Pass 2 cache or detach boundary mismatch")
            require(torch.is_grad_enabled() == self.require_grad and output[1].requires_grad == self.require_grad,
                    "Pass 2 gradient boundary mismatch")
            require(self.same(clean, torch.cat(self.context, dim=1))
                    and self.same(latent, torch.cat(self.inputs, dim=1)), "Pass 2 context or exit input mismatch")
            self.calls.append(["pass2", self.timestep(timestep, self.pipeline.denoising_step_list[self.exit]),
                               self.timestep(aug, self.config.context_noise), tensor_meta(latent), tensor_meta(clean),
                               torch.is_grad_enabled()])
            return
        require(block < self.blocks and not torch.is_grad_enabled() and not output[1].requires_grad,
                "Pass 1 gradient boundary mismatch")
        require(kwargs["current_start"] == block * self.pipeline.frame_seq_length
                and kwargs["kv_cache"] is self.pipeline.kv_cache1
                and kwargs["crossattn_cache"] is self.pipeline.crossattn_cache,
                "Pass 1 cache order mismatch")
        start = block * self.pipeline.num_frame_per_block
        expected = self.noise[:, start:start + self.pipeline.num_frame_per_block] if stage == 0 else self.block_noise
        require(expected is not None and self.same(latent, expected), "Pass 1 latent transition mismatch")
        value = self.config.context_noise if stage == self.stages else self.pipeline.denoising_step_list[stage]
        self.calls.append([block, stage, kwargs["current_start"], self.timestep(timestep, value),
                           tensor_meta(latent), tensor_meta(output[1])])
        if stage == self.stages:
            self.context.append(latent)
            self.outputs.clear()
        else:
            self.outputs.append(output[1])
            if stage == self.exit:
                self.inputs.append(latent)
        self.block_noise = None

    def observe(self, function, *args, **kwargs):
        scheduler = self.pipeline.scheduler
        add_noise = scheduler.add_noise

        def traced_noise(original, noise, timestep):
            require(self.outputs and self.block_noise is None, "Unexpected rollout noise operation")
            context = len(self.outputs) == self.stages
            selected = self.exit if context else len(self.outputs) - 1
            expected = self.outputs[selected]
            require(self.same(original, expected.flatten(0, 1)), "Incorrect denoised cache state selection")
            time_value = self.config.context_noise if context else self.pipeline.denoising_step_list[selected + 1]
            self.transitions.append([len(self.context), selected, context, self.timestep(timestep, time_value),
                                     tensor_hash(noise)])
            result = add_noise(original, noise, timestep)
            self.block_noise = result.unflatten(0, expected.shape[:2])
            return result

        observer = self.pipeline.generator.register_forward_hook(self.forward, with_kwargs=True)
        scheduler.add_noise = traced_noise
        try:
            result = function(*args, **kwargs)
        finally:
            observer.remove()
            scheduler.add_noise = add_noise
        require(len(self.calls) == self.blocks * (self.stages + 1) + 1 and self.calls[-1][0] == "pass2"
                and len(self.context) == self.blocks and len(self.transitions) == self.blocks * self.stages,
                "Incomplete dual-pass rollout")
        return result, {"exit": self.exit, "shape": tensor_meta(self.noise), "noise": tensor_hash(self.noise),
                        "calls": self.calls, "context_trace": self.transitions}


class Observer:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.runtime = spec["runtime"]
        self.steps, self.updates, self.rollouts, self.stamps = [], [], [], {}
        self.update = None
        self.active_rollout = None
        self.model_calls = []
        self.prompts = []
        self.text_batches = 0
        self.flash_calls = 0
        self.boundaries = {step for window in spec["windows"] for step in window}

    def install_attention(self):
        from importlib import import_module

        attention = import_module("wan.modules.attention")

        if self.spec["reference"]:
            require(attention.FLASH_ATTN_2_AVAILABLE and not attention.FLASH_ATTN_3_AVAILABLE,
                    "Pinned upstream FlashAttention-2 dispatch is unavailable", "UNAVAILABLE")
        if attention.FLASH_ATTN_2_AVAILABLE:
            kernel = attention.flash_attn.flash_attn_varlen_func

            def flash(*args, **kwargs):
                result = kernel(*args, **kwargs)
                require(result.is_cuda, "FlashAttention-2 did not return a CUDA tensor")
                self.flash_calls += 1
                return result

            attention.flash_attn.flash_attn_varlen_func = flash

    def install_pipeline(self):
        from pipeline.self_gradient_forcing_training import SelfGradientForcingTrainingPipeline

        original = SelfGradientForcingTrainingPipeline._self_gradient_forcing_inference

        def rollout(pipeline, noise, *args, **kwargs):
            require(self.update is not None and self.active_rollout is None, "Unexpected nested or untracked rollout")
            probe = Rollout(pipeline, self.runtime, self.config, noise, self.update == "generator")
            self.active_rollout = probe
            try:
                result, trace = probe.observe(original, pipeline, noise, *args, **kwargs)
            finally:
                self.active_rollout = None
            self.rollouts.append(fingerprint(trace))
            return result

        SelfGradientForcingTrainingPipeline._self_gradient_forcing_inference = rollout

    def attach(self, trainer):
        from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

        self.trainer = trainer
        require(trainer.step == 0 and trainer.world_size == self.spec["world_size"]
                and dist.get_world_size() == self.spec["world_size"]
                and int(os.environ["LOCAL_WORLD_SIZE"]) == trainer.world_size
                and self.runtime["nnodes"] == 1 and dist.get_backend() == "nccl",
                "Distributed runtime mismatch")
        accumulation = self.config.total_batch_size // (trainer.world_size * self.config.batch_size)
        require(accumulation == self.runtime["gradient_accumulation_steps"], "Unexpected gradient accumulation")
        require(trainer.dtype == torch.bfloat16, "Trainer mixed precision is inactive")
        require(torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32,
                "Trainer TF32 is inactive")
        models = {name: getattr(trainer.model, name)
                  for name in ("generator", "real_score", "fake_score", "text_encoder")}
        require(isinstance(models["generator"], FullyShardedDataParallel), "Expected FSDP generator")
        check_generator_context(models["generator"].module.model, self.config)
        precision = [self.runtime[name] for name in (
            "fsdp_forward_parameter_dtype", "fsdp_reduce_dtype", "fsdp_buffer_dtype")]
        policies = {}
        for role, model in models.items():
            policies[role] = []
            require(isinstance(model, FullyShardedDataParallel), "Expected FSDP model")
            for name, module in model.named_modules():
                if isinstance(module, FullyShardedDataParallel):
                    policy = module.mixed_precision
                    dtypes = [str(dtype).removeprefix("torch.") for dtype in (
                        policy.param_dtype, policy.reduce_dtype, policy.buffer_dtype)]
                    require(dtypes == precision and module.sharding_strategy == ShardingStrategy.HYBRID_SHARD,
                            "FSDP precision or sharding mismatch")
                    policies[role].append([name, dtypes])
            if role in {"generator", "fake_score"} and self.spec["reference"]:
                require(any(getattr(module, "gradient_checkpointing", False) for module in model.modules()),
                        "Reference gradient checkpointing is inactive")

        optimizers = {name: getattr(trainer, f"{name}_optimizer") for name in ("generator", "critic")}
        require(all(isinstance(opt, torch.optim.AdamW) and not opt.state for opt in optimizers.values())
                and trainer.generator_ema is None, "Unexpected optimizer or EMA starting state")
        self.initial = {
            "optimizer": fingerprint({name: optimizer_config(opt) for name, opt in optimizers.items()}),
            "starting_weights": fingerprint({name: state_hash(model) for name, model in models.items()}),
            "tensor_metadata": fingerprint({"parameters": {name: named_metadata(model) for name, model in models.items()},
                                            "policies": policies}),
            "grad_clip": [trainer.max_grad_norm_generator, trainer.max_grad_norm_critic],
        }
        self.pending = None
        self.before = {}
        for name, opt in optimizers.items():
            opt.register_step_pre_hook(self.optimizer_pre(name))
            opt.register_step_post_hook(self.optimizer_post(name))
        models["text_encoder"].register_forward_pre_hook(self.text_input, with_kwargs=True)
        for role in ("real_score", "fake_score"):
            models[role].register_forward_hook(self.score_call(role), with_kwargs=True)

    def text_input(self, module, args, kwargs):
        require(self.update is not None and not args and not torch.is_grad_enabled(), "Invalid text encoder invocation")
        expected = self.prompts if self.text_batches == 0 else [self.config.negative_prompt] * len(self.prompts)
        require(list(kwargs["text_prompts"]) == expected, "Text encoder consumed a different batch")
        self.text_batches += 1

    def score_call(self, role):
        def capture(module, args, kwargs, output):
            latent = kwargs["noisy_image_or_video"]
            require(self.update is not None and not args and latent.is_cuda and output[1].is_cuda,
                    "Invalid CUDA score invocation")
            self.model_calls.append([role, torch.is_grad_enabled(), tensor_meta(latent),
                                     tensor_meta(kwargs["timestep"]), tensor_meta(output[1])])
        return capture

    def fwdbwd(self, function, batch, train_generator, *args, **kwargs):
        require(self.update is None and self.pending is None, "Overlapping or skipped optimizer update")
        self.update = "generator" if train_generator else "critic"
        self.rollouts = []
        self.model_calls = []
        self.prompts = list(batch["prompts"])
        self.text_batches = 0
        samples = batch["idx"].tolist()
        require(len(samples) == len(batch["prompts"]) == self.config.batch_size, "Invalid rank batch size")
        start_random = [tensor_hash(torch.random.get_rng_state()), tensor_hash(torch.cuda.get_rng_state())]
        sampled = []
        flash_before = self.flash_calls
        def observe_rollout_sample(name, value):
            if self.active_rollout is not None:
                self.active_rollout.random_sample(name, value)

        result = with_random_trace(lambda: function(batch, train_generator, *args, **kwargs), sampled, observe_rollout_sample)
        require(len(self.rollouts) == 1 and sampled and self.text_batches in (1, 2),
                "Missing optimizer sampling, text encoding or rollout evidence")
        require(any(call[0] == "fake_score" for call in self.model_calls)
                and (not train_generator or any(call[0] == "real_score" for call in self.model_calls)),
                "Missing score-model work")
        if self.spec["reference"]:
            require(self.flash_calls > flash_before, "No active FlashAttention-2 dispatch in reference update")
        details = {
            "kind": self.update, "sample_ids": samples, "prompts": fingerprint([self.prompts, self.text_batches]),
            "sampling": fingerprint([start_random, sampled, tensor_hash(torch.cuda.get_rng_state())]),
            "rollout_trace": self.rollouts[0],
            "score_trace": fingerprint(self.model_calls),
        }
        endpoints = {}
        if self.spec["endpoints"]:
            role = self.update
            loss_key = "generator_dmd_loss" if train_generator else "critic_loss"
            endpoints = {loss_key: result[f"{role}_loss"].detach().mean().item(),
                         f"{role}_gradient_norm": result[f"{role}_grad_norm"].detach().mean().item()}
        self.pending = {"exact": details, "endpoints": endpoints}
        self.update = None
        return result

    def optimizer_pre(self, role):
        def capture(optimizer, args, kwargs):
            require(self.pending is not None and self.pending["exact"]["kind"] == role,
                    "Optimizer called out of order")
            params = [param for group in optimizer.param_groups for param in group["params"] if param.grad is not None]
            require(params, "No local updated parameters")
            self.pending["exact"]["gradients"] = fingerprint([tensor_meta(param.grad) for param in params])
            if self.spec["endpoints"]:
                with torch.no_grad():
                    self.before = [(param, param.detach().clone()) for param in params]
        return capture

    def optimizer_post(self, role):
        def capture(optimizer, args, kwargs):
            self.pending["exact"]["optimizer_state"] = optimizer_metadata(optimizer)
            if self.spec["endpoints"]:
                with torch.no_grad():
                    differences = [(param.detach().float() - before.float()).norm().double().square()
                                   for param, before in self.before]
                    self.pending["endpoints"][f"{role}_parameter_delta"] = torch.stack(differences).sum().item()
                self.before = []
            self.updates.append(self.pending)
            self.pending = None
        return capture

    def completed(self, step):
        if step in self.boundaries:
            torch.cuda.synchronize()
            self.stamps[str(step)] = time.perf_counter()
        expected = ["critic"]
        prior = step - 1
        if prior > 0 and prior % self.config.dfake_gen_update_ratio == 0:
            expected.insert(0, "generator")
        require(step == len(self.steps) + 1 and self.pending is None
                and [update["exact"]["kind"] for update in self.updates] == expected,
                "Incomplete or out-of-order Trainer.train iteration")
        endpoints = {key: value for update in self.updates for key, value in update["endpoints"].items()}
        require(all(math.isfinite(value) for value in endpoints.values()), "Nonfinite trajectory endpoint")
        self.steps.append({"step": step, "exact": [update["exact"] for update in self.updates], "endpoints": endpoints})
        self.updates = []


def worker(spec_path, output):
    global torch, dist
    import torch
    import torch.distributed as dist
    from importlib.metadata import version
    from omegaconf import OmegaConf

    spec = read_json(spec_path)
    require(version("diffusers") == "0.31.0", "diffusers==0.31.0 is required", "UNAVAILABLE")
    require(torch.cuda.is_available() and torch.cuda.device_count() == spec["world_size"]
            and all("H200" in torch.cuda.get_device_name(device) for device in range(spec["world_size"])),
            "Expected four visible H200s", "UNAVAILABLE")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    sys.path.insert(1, str(Path.cwd()))
    from trainer.distillation import Trainer

    config = OmegaConf.create(spec["config"])
    observer = Observer(spec, config)
    observer.install_attention()
    observer.install_pipeline()

    class ObservedTrainer(Trainer):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "step" and value > 0:
                observer.completed(value)

        def fwdbwd_one_step(self, batch, train_generator, *args, **kwargs):
            return observer.fwdbwd(super().fwdbwd_one_step, batch, train_generator, *args, **kwargs)

    trainer = ObservedTrainer(config)
    observer.attach(trainer)
    trainer.train()
    require(trainer.step == len(observer.steps) == spec["steps"] and not observer.updates,
            "Upstream Trainer.train did not reach the last completed update")
    torch.cuda.synchronize()
    dist.barrier()
    rank = dist.get_rank()
    report = {"rank": rank, "seed": spec["seed"], "steps": observer.steps, "timestamps": observer.stamps,
              "initial": observer.initial, "config": fingerprint(OmegaConf.to_container(config, resolve=True)),
              "flash_calls": observer.flash_calls}
    dist.destroy_process_group()
    (output / f"rank-{rank}.json").write_text(json.dumps(report, allow_nan=False) + "\n")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True)
    args, extra = parser.parse_known_args()
    from bench import main as bench_main

    return bench_main(["time", "--out", args.logdir, *extra])


if __name__ == "__main__":
    raise SystemExit(main())
