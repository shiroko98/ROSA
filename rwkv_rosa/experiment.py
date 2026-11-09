"""Paired RWKV-7 / original ROSA / improved ROSA SFT experiment runner."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from rosa_checkpointing import atomic_torch_save
from rosa_optim import build_training_optimizers
from rosa_runtime import RosaAddressBatch
from rosa_train_async import maybe_wrap_train_address_prefetch
from third_party.rwkv_lm_v7.sft_loss import masked_head_cross_entropy
from .backbone import RWKVConfig, UPSTREAM_COMMIT, build_backbone, infer_config, load_base_state
from .data import (ROOT, DEFAULT_TEMPLATE, EqualHistoryDataset, build_tokenizer, collate,
                   load_documents, split_documents, split_summary, validate_splits)
from .model import RWKVRosaLM, configure_train_scope
from .precision import WKV_PRECISION_POLICY
from .binidx_data import load_binidx_documents
from .tracking import ArmTracker
from .freeze_audit import check_frozen_backbone, parameter_hash


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def device_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


def make_loader(dataset, args, model, *, train, epoch=0):
    generator = torch.Generator().manual_seed(args.seed + epoch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, generator=generator,
                        collate_fn=collate, num_workers=0, pin_memory=args.device == "cuda")
    return maybe_wrap_train_address_prefetch(loader, address_engine=model.address_engine,
                                            enabled=model.has_rosa and args.address_async,
                                            max_workers=1, prefetch_batches=args.prefetch_depth)


def unpack_batch(batch, device, *, address_device=None):
    tensors = {k: batch[k].to(device, non_blocking=True) for k in ("input_ids", "targets", "loss_mask")}
    address = None
    if "rosa_precomputed_ids" in batch:
        ids = batch["rosa_precomputed_ids"]
        address = RosaAddressBatch(addr_ids=ids, raw_match_lens=batch["rosa_precomputed_raw_best_lens"],
                                   fired_match_lens=batch["rosa_precomputed_match_lens"], valid_mask=ids.ge(0),
                                   special_mask=torch.zeros_like(ids, dtype=torch.bool),
                                   source=batch["rosa_precomputed_source"]).to(address_device or device)
    return tensors, address


@torch.no_grad()
def score_hidden(hidden, head, targets, mask, chunk_size):
    active = mask.reshape(-1).gt(0)
    h = hidden.reshape(-1, hidden.shape[-1])[active]
    y = targets.reshape(-1)[active]
    weights = mask.reshape(-1)[active].float()
    loss_sum = correct = count = 0.0
    for start in range(0, len(y), chunk_size):
        logits = head(h[start:start + chunk_size]).float()
        weight = weights[start:start + chunk_size]
        target = y[start:start + chunk_size]
        loss_sum += float((F.cross_entropy(logits, target, reduction="none") * weight).sum())
        correct += float((logits.argmax(-1).eq(target) * weight).sum())
        count += float(weight.sum())
    return loss_sum, correct, count


def finish_metrics(loss_sum, correct, count):
    if count <= 0:
        raise ValueError("No supervised tokens were evaluated.")
    loss = loss_sum / count
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite loss.")
    return {"loss": loss, "ppl": math.exp(min(loss, 80)), "token_accuracy": correct / count,
            "supervised_tokens": int(count)}


@torch.no_grad()
def evaluate(model, dataset, args, device, dtype):
    model.eval()
    loss_sum = correct = count = 0.0
    for i, raw in enumerate(make_loader(dataset, args, model, train=False)):
        if args.eval_batches and i >= args.eval_batches:
            break
        batch, address = unpack_batch(raw, device, address_device=model.rosa_device)
        with autocast_context(device, dtype):
            hidden, _, _ = model.forward_features(batch["input_ids"], addresses=address)
            nll, hits, tokens = score_hidden(hidden, model.backbone.head, batch["targets"],
                                              batch["loss_mask"], args.head_chunk_size)
        loss_sum += nll
        correct += hits
        count += tokens
    return finish_metrics(loss_sum, correct, count)


@torch.no_grad()
def verify_online(model, dataset, device, dtype):
    """Separate FP32 state semantics from expected BF16 GEMM shape-rounding drift."""
    model.eval()
    ids = dataset[0]["input_ids"][:64].unsqueeze(0).to(device)
    split = max(1, ids.shape[1] // 2)

    def compare(precision):
        with autocast_context(device, precision):
            full = model(ids)["logits"].float()
            if split < ids.shape[1]:
                first = model(ids[:, :split], return_state=True)
                last = model(ids[:, split:], state=first["state"])
                streamed = torch.cat((first["logits"], last["logits"]), dim=1).float()
            else:
                streamed = model(ids, return_state=True)["logits"].float()
        diff = full - streamed
        return full, streamed, {"max_abs_diff": float(diff.abs().max()),
                               "relative_rms_error": float(diff.square().mean().sqrt() /
                                                           full.square().mean().sqrt().clamp_min(1e-12)),
                               "top1_agreement": float(full.argmax(-1).eq(streamed.argmax(-1)).float().mean()),
                               "top1_changed": int(full.argmax(-1).ne(streamed.argmax(-1)).sum()),
                               "predictions": full.shape[0] * full.shape[1]}

    low_precision = compare(dtype)[2] if dtype != torch.float32 else None
    if low_precision is not None and low_precision["top1_changed"]:
        print(f"WARNING {model.variant}: BF16 partition changed "
              f"{low_precision['top1_changed']}/{low_precision['predictions']} next-token predictions; "
              "use FP32 for accuracy comparisons.", flush=True)
    storage_dtypes = {name: p.dtype for name, p in model.named_parameters()}
    runtime = model.backbone._reference_runtime
    previous_runtime_dtype = runtime.DTYPE
    previous_scale = model.scale
    try:
        # The small verification window temporarily promotes frozen BF16 weights.
        # This increases verification memory, but leaves saved/trained dtypes intact.
        model.float()
        runtime.configure_runtime(dtype=torch.float32)
        full, streamed, diagnostics = compare(torch.float32)
        model.scale = 0.0
        zero_rosa = model(ids)["logits"]
        upstream = model.backbone(ids)
        zero_diff = float((zero_rosa.float() - upstream.float()).abs().max())
        equal = torch.allclose(full, streamed, atol=2e-3, rtol=2e-4)
        if not equal or zero_diff != 0:
            raise AssertionError(f"FP32 RWKV/ROSA state parity failed: {diagnostics}, scale_zero={zero_diff}")
    finally:
        model.scale = previous_scale
        runtime.configure_runtime(dtype=previous_runtime_dtype)
        for name, p in model.named_parameters():
            p.data = p.data.to(storage_dtypes[name])
    return {"tokens": ids.shape[1], "verification_precision": "float32", "streaming_allclose": equal,
            "streaming_max_abs_diff": diagnostics["max_abs_diff"], "float32": diagnostics,
            "streaming_atol": 2e-3, "streaming_rtol": 2e-4,
            "scale_zero_vs_upstream_max_abs_diff": zero_diff, "training_precision_streaming": low_precision,
            "bf16_generation_equivalence": "not_guaranteed" if low_precision is not None else "not_applicable"}


def save_checkpoint(path, model, optimizer, *, signature, epoch, batch_offset, step, history):
    # Frozen backbone tensors are reconstructed from the recorded, hashed base .pth.
    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    atomic_torch_save({"format": "rwkv_rosa_trainable_delta_v1", "signature": signature,
                       "trainable_state": weights, "dense_optimizer": optimizer.dense_optimizer.state_dict(),
                       "sparse_optimizer": optimizer.sparse_optimizer.state_dict() if optimizer.sparse_optimizer else None,
                       "epoch": epoch, "batch_offset": batch_offset, "global_step": step, "history": history,
                       "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, str(path))


def restore_checkpoint(path, model, optimizer, signature):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["signature"] != signature:
        raise ValueError("Resume configuration, data, upstream commit or base checkpoint changed.")
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(ckpt["trainable_state"]) != expected:
        raise ValueError("Resume trainable parameter set does not match.")
    model.load_state_dict(ckpt["trainable_state"], strict=False)
    optimizer.dense_optimizer.load_state_dict(ckpt["dense_optimizer"])
    if optimizer.sparse_optimizer is not None:
        optimizer.sparse_optimizer.load_state_dict(ckpt["sparse_optimizer"])
    torch.set_rng_state(ckpt["torch_rng"])
    random.setstate(ckpt["python_rng"])
    if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return ckpt


def clip_all_gradients(optimizer, max_norm):
    """One global L2 norm, including SparseAdam rows; reject NaN/Inf before update."""
    gradients = []
    for parameter in [*optimizer.dense_params, *optimizer.sparse_params]:
        if parameter.grad is None:
            continue
        if parameter.grad.is_sparse:
            parameter.grad = parameter.grad.coalesce()
            gradients.append(parameter.grad.values())
        else:
            gradients.append(parameter.grad)
    if not gradients:
        return 0.0
    norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for g in gradients]))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm; optimizer update aborted.")
    if max_norm > 0:
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for gradient in gradients:
            gradient.mul_(scale.to(gradient.device))
    return float(norm)


def train_variant(model, datasets, args, signature, out_dir, device, dtype, tracker=None):
    optimizer = build_training_optimizers(model, lr=args.lr, weight_decay=args.weight_decay)
    step, start_epoch, offset, history = 0, 0, 0, []
    last = out_dir / "last.pt"
    if args.resume and last.exists():
        ckpt = restore_checkpoint(last, model, optimizer, signature)
        step, start_epoch, offset, history = (ckpt[k] for k in ("global_step", "epoch", "batch_offset", "history"))
    frozen_audit = None
    if args.train_scope == 'rosa_only':
        frozen_audit = {'backbone_sha256_before': check_frozen_backbone(model),
                        'rosa_sha256_before': parameter_hash(model.rosa_value_store),
                        'all_backbone_requires_grad_false': True,
                        'trainable_devices': sorted({str(p.device) for p in model.parameters() if p.requires_grad})}
        write_json(out_dir / 'freeze_audit.json', frozen_audit)
    initial_path = out_dir / "initial_val.json"
    initial_val = (json.loads(initial_path.read_text()) if args.resume and initial_path.exists()
                   else evaluate(model, datasets["val"], args, device, dtype))
    write_json(initial_path, initial_val)
    if tracker:
        tracker.log('initial_val', initial_val, step)
        tracker.summary('initial_val', initial_val)
    print(f'{model.variant} initial_val={initial_val}', flush=True)
    supervised_tokens_seen = sum(row['supervised_tokens'] for row in history)
    device_sync(device)
    for epoch in range(start_epoch, args.epochs):
        if args.max_steps and step >= args.max_steps:
            break
        model.train()
        loader = make_loader(datasets["train"], args, model, train=True, epoch=epoch)
        batches = iter(loader)
        consumed = offset if epoch == start_epoch else 0
        for _ in range(consumed):
            next(batches)
        for_bucket = consumed
        try:
            while True:
                device_sync(device)
                started = time.perf_counter()
                bucket = list(islice(batches, args.grad_accum_steps))
                if not bucket:
                    break
                tokens = sum(float(b['loss_mask'].sum()) for b in bucket)
                batch_digest = hashlib.sha256()
                for raw in bucket:
                    for key in ('input_ids', 'targets', 'loss_mask'):
                        batch_digest.update(str(tuple(raw[key].shape)).encode())
                        batch_digest.update(raw[key].contiguous().numpy().tobytes())
                optimizer.zero_grad()
                loss_sum = correct = count = 0.0
                stats = {}
                for raw in bucket:
                    batch, address = unpack_batch(raw, device, address_device=model.rosa_device)
                    with autocast_context(device, dtype):
                        hidden, stats, _ = model.forward_features(batch["input_ids"], addresses=address)
                        loss = masked_head_cross_entropy(hidden, model.backbone.head, batch["targets"],
                                                         batch["loss_mask"], chunk_size=args.head_chunk_size)
                        weight = float(batch["loss_mask"].sum())
                        normalized = loss * (weight / tokens)
                        nll, hits, n = score_hidden(hidden, model.backbone.head, batch["targets"],
                                                    batch["loss_mask"], args.head_chunk_size)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss; no checkpoint written for this step.")
                    normalized.backward()
                    loss_sum += nll
                    correct += hits
                    count += n
                if frozen_audit is not None:
                    if any(p.grad is not None for p in model.backbone.parameters()):
                        raise AssertionError('Unexpected backbone gradient during ROSA-only training.')
                    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                    if not grads:
                        raise AssertionError('No ROSA gradient; CPU/GPU autograd chain is broken.')
                    if args.rosa_device == 'cpu' and any(g.device.type != 'cpu' for g in grads):
                        raise AssertionError('ROSA gradient unexpectedly resides on CUDA.')
                norm = clip_all_gradients(optimizer, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if frozen_audit is not None and 'backbone_sha256_after_first_update' not in frozen_audit:
                    frozen_audit['backbone_sha256_after_first_update'] = check_frozen_backbone(
                        model, frozen_audit['backbone_sha256_before'])
                    frozen_audit['rosa_sha256_after_first_update'] = parameter_hash(model.rosa_value_store)
                    frozen_audit['rosa_values_changed_after_first_update'] = (
                        frozen_audit['rosa_sha256_before'] != frozen_audit['rosa_sha256_after_first_update'])
                    frozen_audit['backbone_gradients_absent'] = True
                    frozen_audit['first_observed_update'] = step
                    if args.rosa_device == 'cpu':
                        states = [v for opt in (optimizer.dense_optimizer, optimizer.sparse_optimizer) if opt is not None
                                  for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
                        if any(v.device.type != 'cpu' for v in states):
                            raise AssertionError('CPU ROSA optimizer state unexpectedly resides on CUDA.')
                        frozen_audit['optimizer_states_all_cpu'] = True
                    write_json(out_dir / 'freeze_audit.json', frozen_audit)
                    if tracker:
                        tracker.summary('freeze_audit', frozen_audit)
                    print('Frozen-backbone first-update verification:', frozen_audit, flush=True)
                for_bucket += len(bucket)
                device_sync(device)
                metrics = {**finish_metrics(loss_sum, correct, count), **stats, "step": step,
                           "epoch": epoch + 1, "seconds": time.perf_counter() - started,
                           "gradient_norm": norm}
                supervised_tokens_seen += int(count)
                metrics['supervised_tokens_seen'] = supervised_tokens_seen
                metrics['batch_sha256'] = batch_digest.hexdigest()
                metrics['learning_rate'] = args.lr
                metrics['input_tokens'] = sum(raw['input_ids'].numel() for raw in bucket)
                metrics['supervised_tokens_per_second'] = count / metrics['seconds']
                if device.type == "cuda":
                    metrics["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
                history.append(metrics)
                if tracker:
                    tracker.log('train', metrics, step)
                if args.eval_every_steps and step % args.eval_every_steps == 0:
                    val = evaluate(model, datasets['val'], args, device, dtype)
                    model.train()
                    with (out_dir / 'val.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'step': step, **val}, allow_nan=False) + '\n')
                    if tracker:
                        tracker.log('val', val, step)
                        tracker.summary('val', val)
                    print(f'{model.variant} step={step} val={val}', flush=True)
                with (out_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                print(f"{model.variant} epoch={epoch+1} step={step} loss={metrics['loss']:.5f} "
                      f"acc={metrics['token_accuracy']:.4f} seconds={metrics['seconds']:.3f}", flush=True)
                if args.save_checkpoints and args.save_every_steps and step % args.save_every_steps == 0:
                    save_checkpoint(last, model, optimizer, signature=signature, epoch=epoch,
                                    batch_offset=for_bucket, step=step, history=history)
                if args.max_steps and step >= args.max_steps:
                    break
        finally:
            # Close async generator even on early stop to join its CPU workers.
            if hasattr(batches, "close"):
                batches.close()
        completed_epoch = for_bucket >= len(loader)
        next_epoch, next_offset = (epoch + 1, 0) if completed_epoch else (epoch, for_bucket)
        if args.save_checkpoints:
            save_checkpoint(last, model, optimizer, signature=signature, epoch=next_epoch,
                            batch_offset=next_offset, step=step, history=history)
        offset = 0
    if frozen_audit is not None:
        frozen_audit['backbone_sha256_final'] = check_frozen_backbone(model, frozen_audit['backbone_sha256_before'])
        frozen_audit['backbone_unchanged'] = True
        write_json(out_dir / 'freeze_audit.json', frozen_audit)
    final_val = evaluate(model, datasets["val"], args, device, dtype)
    test = evaluate(model, datasets["test"], args, device, dtype)
    parity = verify_online(model, datasets["val"], device, dtype) if args.verify_online and model.variant != "capacity" else None
    result = {"freeze_audit": frozen_audit, "online_verification": parity, "initial_val": initial_val, "final_val": final_val, "test": test, "global_step": step,
              "train": history, "checkpoint": str(last) if last.exists() else None,
              "evaluation_is_truncated": bool(args.eval_batches or args.eval_windows), "wkv_backend": "upstream_torch_reference"}
    if tracker:
        tracker.log('val', final_val, step)
        tracker.log('test', test, step)
        tracker.summary('val', final_val)
        tracker.summary('test', test)
        result['wandb_url'] = tracker.url
    write_json(out_dir / "summary.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data", default=str(ROOT / "examples/rwkv_rosa_toy.json"))
    parser.add_argument('--data-format', choices=('auto', 'records', 'binidx'), default='auto')
    parser.add_argument('--binidx-max-rows', type=int, default=2048,
                        help='Seed-random physical rows to unpack; 0 = entire source, potentially huge.')
    parser.add_argument('--binidx-eot-id', type=int, default=65532)
    parser.add_argument('--extra-special-ids', default='', help='Additional ROSA-excluded marker IDs, comma separated.')
    parser.add_argument('--eval-windows', type=int, default=0,
                        help='Fixed seed-random validation/test windows per split; 0 = all.')
    parser.add_argument('--eval-every-steps', type=int, default=0)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb-project', default='ROSA-RWKV7')
    parser.add_argument('--wandb-entity', default='')
    parser.add_argument('--wandb-group', default='')
    parser.add_argument('--wandb-run-name', default='')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument("--val-data", default="")
    parser.add_argument("--test-data", default="")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--tokenizer", choices=("byte", "rwkv"), default="byte")
    parser.add_argument("--vocab", default=str(ROOT / "rwkv_vocab_v20230424.txt"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/rwkv_rosa_compare"))
    parser.add_argument("--variants", default="baseline,original,improved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Total optimizer updates per arm; 0 = full epochs.")
    parser.add_argument("--eval-batches", type=int, default=0, help="0 = all selected validation/test windows; >0 truncates evaluation batches.")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-len", type=int, default=32)
    parser.add_argument("--history-tokens", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--head-chunk-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--train-scope", choices=("full", "last_block", "head", "rosa_only"), default="full")
    parser.add_argument('--freeze-backbone', action=argparse.BooleanOptionalAction, default=False,
                        help='Freeze ALL backbone parameters including embeddings, norms and head; train improved ROSA only.')
    parser.add_argument('--rosa-device', choices=('same', 'cpu'), default='same',
                        help='cpu places values, projections, norms, gates and ROSA optimizer on CPU, not just lookup.')
    parser.add_argument("--train-last-blocks", type=int, default=1)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--address-async", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--inject-layers", default="0")
    parser.add_argument("--rosa-scale", type=float, default=0.15)
    parser.add_argument("--min-match-len", type=int, default=1)
    parser.add_argument("--sparse-values", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--value-shards", type=int, default=1)
    parser.add_argument("--sam-impl", choices=("fast", "stateful", "compiled_cpu"), default="fast")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--verify-online", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--toy-dim", type=int, default=32)
    parser.add_argument("--toy-layers", type=int, default=2)
    parser.add_argument("--toy-head-size", type=int, default=16)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    if known.config:
        config = json.loads(Path(known.config).read_text(encoding="utf-8"))
        valid = {a.dest for a in parser._actions}
        if unknown := set(config) - valid:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for key in ("epochs", "batch_size", "chunk_len", "grad_accum_steps", "head_chunk_size", "prefetch_depth"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    for key in ("history_tokens", "max_steps", "eval_batches", "max_docs", "save_every_steps",
                "binidx_max_rows", "eval_windows", "eval_every_steps"):
        if getattr(args, key) < 0:
            parser.error(f"{key} must be nonnegative")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("CPU comparison uses float32; select cuda for bfloat16.")
    if args.checkpoint and args.tokenizer == "byte":
        parser.error("Real RWKV checkpoints require --tokenizer rwkv, not the toy byte tokenizer.")
    if bool(args.val_data) != bool(args.test_data):
        parser.error("Provide both --val-data and --test-data or neither.")
    if args.resume and not args.save_checkpoints:
        parser.error("--resume requires checkpoints.")
    if args.freeze_backbone:
        args.train_scope = 'rosa_only'
    if args.train_scope == 'rosa_only' and args.variants != 'improved':
        parser.error('ROSA-only/frozen-backbone training requires --variants improved.')
    if args.rosa_device == 'cpu' and args.train_scope != 'rosa_only':
        parser.error('--rosa-device cpu currently requires --freeze-backbone or --train-scope rosa_only.')
    return args


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This comparison runner is single-process; do not launch it with torchrun/DDP.")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if device.type == "cuda":
        # Prefer FP32 accumulation over BF16 partial reductions for recurrent stability.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
    variants = args.variants.split(",")
    if len(set(variants)) != len(variants) or not set(variants) <= {"baseline", "original", "improved", "capacity"}:
        raise ValueError("variants must be distinct: baseline,original,improved[,capacity].")
    layer_ids = tuple(int(s) for s in args.inject_layers.split(","))
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Output directory is not empty; use a new directory or --resume.")
    out.mkdir(parents=True, exist_ok=True)
    if not args.data:
        raise ValueError("Provide --data with a real training corpus; the SFT config intentionally has no toy fallback.")
    tokenizer = build_tokenizer(args.tokenizer, args.vocab)
    tokenizer.special_ids |= {int(i) for i in args.extra_special_ids.split(',') if i.strip()}
    data_path = Path(args.data)
    is_binidx = args.data_format == 'binidx' or (args.data_format == 'auto' and
                (data_path.is_dir() or data_path.suffix in ('.idx', '.bin') or Path(str(data_path) + '.idx').exists()))
    source_metadata = None
    if is_binidx:
        if args.val_data:
            raise ValueError('Binidx uses seeded example-disjoint splits; explicit separate binidx splits not implemented.')
        docs, source_metadata = load_binidx_documents(args.data, seed=args.seed, max_rows=args.binidx_max_rows,
                                                      eot_id=args.binidx_eot_id, max_docs=args.max_docs)
    else:
        docs = load_documents(args.data, tokenizer, template_path=args.template, max_docs=args.max_docs)
    if args.val_data:
        splits = {'train': docs,
                  'val': load_documents(args.val_data, tokenizer, template_path=args.template, max_docs=args.max_docs),
                  'test': load_documents(args.test_data, tokenizer, template_path=args.template, max_docs=args.max_docs)}
    else:
        train, val, test = split_documents(docs, args.seed)
        splits = {'train': train, 'val': val, 'test': test}
    validate_splits(splits)
    datasets = {k: EqualHistoryDataset(v, chunk_len=args.chunk_len, history_tokens=args.history_tokens)
                for k, v in splits.items()}
    data_summary = split_summary(splits, datasets)
    for index, name in enumerate(('val', 'test')):
        if args.eval_windows and args.eval_windows < len(datasets[name]):
            selected = random.Random(args.seed + 1001 + index).sample(range(len(datasets[name])), args.eval_windows)
            datasets[name] = Subset(datasets[name], selected)
            data_summary[name]['evaluated_window_indices'] = selected
        data_summary[name]['evaluated_windows'] = len(datasets[name])
    print('Data splits:', {k: {x: y for x, y in v.items() if x != 'document_hashes' and
                              x != 'evaluated_window_indices'} for k, v in data_summary.items()}, flush=True)
    seed_all(args.seed)
    if args.checkpoint:
        print(f"Loading base checkpoint: {args.checkpoint}", flush=True)
        base_state = load_base_state(args.checkpoint)
        cfg = infer_config(base_state)
        base_hash = file_hash(args.checkpoint)
    else:
        cfg = RWKVConfig(vocab_size=tokenizer.vocab_size, n_embd=args.toy_dim,
                         n_layer=args.toy_layers, head_size=args.toy_head_size)
        initial = build_backbone(cfg)
        base_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
        del initial
        base_hash = "seeded_toy_initialization:" + str(args.seed)
    if tokenizer.vocab_size > cfg.vocab_size:
        raise ValueError("Tokenizer contains IDs beyond checkpoint vocabulary.")
    if source_metadata and source_metadata['token_id_max'] >= cfg.vocab_size:
        raise ValueError('Binidx token ID exceeds checkpoint vocabulary.')
    semantic = {k: v for k, v in vars(args).items()
                if k not in {"out_dir", "resume", "epochs", "max_steps", "save_every_steps", "config", "variants",
                             "wandb", "wandb_project", "wandb_entity", "wandb_group", "wandb_run_name", "wandb_mode"}}
    metadata = {"upstream_commit": UPSTREAM_COMMIT, "model_config": asdict(cfg), "base_sha256": base_hash,
                "data": data_summary, "source_data": source_metadata, "semantic_config": semantic,
                "history_protocol": "identical bounded left context for all arms; no external memory",
                "original_definition": "original SAM + shared embedding residual, no context/length gate",
                "improved_definition": "same SAM + per-layer values + context/length gate",
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "wkv_precision_policy": WKV_PRECISION_POLICY,
                "bf16_reduced_precision_reduction": False if device.type == "cuda" else None,
                "template_sha256": file_hash(args.template),
                "vocab_sha256": file_hash(args.vocab) if args.tokenizer == "rwkv" else None,
                "integration_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    manifest = out / "manifest.json"
    if args.resume and manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["signature"] != signature:
            raise ValueError("Resume manifest differs; use a separate experiment directory.")
    write_json(manifest, {**metadata, "signature": signature, "requested_config": vars(args)})
    summary_file = out / "comparison.json"
    comparison = json.loads(summary_file.read_text()) if args.resume and summary_file.exists() else {"runs": {}}
    comparison["manifest"] = str(manifest)
    slots, dim = len(set(layer_ids)), cfg.n_embd
    extra = slots * (cfg.vocab_size * dim + 2 * dim * dim + 2 * dim + 2)
    capacity_width = max(1, extra // (2 * dim * slots))
    for variant in variants:
        seed_all(args.seed)
        print(f"Building {variant}: {asdict(cfg)}", flush=True)
        model = RWKVRosaLM(cfg, variant=variant, inject_layers=layer_ids, scale=args.rosa_scale,
                           min_match_len=args.min_match_len, sparse_values=args.sparse_values,
                           value_shards=args.value_shards, sam_impl=args.sam_impl,
                           special_ids=tokenizer.special_ids, dtype=dtype, base_state=base_state,
                           activation_checkpointing=args.activation_checkpointing, capacity_width=capacity_width)
        model.place_devices(device, 'cpu' if args.rosa_device == 'cpu' else None)
        scope = configure_train_scope(model, args.train_scope, args.train_last_blocks)
        print('Parameter/device audit:', scope, flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        arm_dir = out / variant
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_signature = signature + ":" + variant
        tracker = ArmTracker(args, variant, arm_dir, arm_signature)
        tracker.summary('parameters', scope)
        try:
            result = train_variant(model, datasets, args, arm_signature, arm_dir, device, dtype, tracker=tracker)
        except BaseException:
            tracker.finish(failed=True)
            raise
        else:
            tracker.finish()
        result["parameters"] = scope
        if variant == "capacity":
            result["capacity_parameter_gap_to_improved"] = extra - 2 * dim * slots * capacity_width
        comparison["runs"][variant] = result
        write_json(summary_file, comparison)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return comparison


def main(argv=None):
    return run(parse_args(argv))
