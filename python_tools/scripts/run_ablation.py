#!/usr/bin/env python
"""Run ablation study: baseline + individual improvements, 8000 steps each."""
import subprocess, json, sys, os, glob

DATA_DIR = "../blendinfo dataset/_generated/prepared_w5_elev_v8"
BASE_CMD = [
    sys.executable, "scripts/train_blend_model_hf.py",
    "--data-dir", DATA_DIR,
    "--arch", "token",
    "--batch-size", "32",
    "--lr", "2e-4",
    "--eval-steps", "2000",
    "--save-steps", "50000",   # don't save intermediate checkpoints
    "--require-cuda",
    "--report-best-threshold",
    "--logging-steps", "500",
    "--max-steps", "8000",
]

EXPERIMENTS = [
    ("baseline",      []),
    ("logit_adj",     ["--use-logit-adj"]),
    ("asl",           ["--use-asl"]),
    ("feature_gate",  ["--use-feature-gate"]),
    ("cascaded",      ["--use-cascaded-heads"]),
    ("mixup",         ["--mixup-alpha", "0.2"]),
    ("mt_cp",         ["--use-mt-cp"]),
    ("logitadj+fg",   ["--use-logit-adj", "--use-feature-gate"]),
    ("all_fixed",     ["--use-asl", "--use-logit-adj", "--use-cascaded-heads",
                       "--mixup-alpha", "0.2", "--use-mt-cp", "--use-feature-gate"]),
]

def get_last_eval(out_dir):
    """Extract last eval metrics from trainer_state.json."""
    ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*/trainer_state.json")))
    final = os.path.join(out_dir, "final", "trainer_state.json")
    if os.path.exists(final):
        ckpts.append(final)
    # Also check root
    root_state = os.path.join(out_dir, "trainer_state.json")
    if os.path.exists(root_state):
        ckpts.append(root_state)

    for path in reversed(ckpts):
        with open(path) as f:
            state = json.load(f)
        for entry in reversed(state.get("log_history", [])):
            if "eval_loss" in entry:
                return entry
    return None

results = []
for name, extra_args in EXPERIMENTS:
    out_dir = f"./training_outputs/ablation_{name}"
    cmd = BASE_CMD + ["--out-dir", out_dir] + extra_args
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Args: {' '.join(extra_args) or '(none)'}")
    print(f"{'='*60}\n", flush=True)

    ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + "/..")

    metrics = get_last_eval(out_dir)
    if metrics:
        r = {
            "name": name,
            "f1": metrics.get("eval_blend_present_f1", "?"),
            "best_f1": metrics.get("eval_blend_present_best_f1", "?"),
            "dir_acc": metrics.get("eval_blend_dir_acc", "?"),
            "mask_exact": metrics.get("eval_blend_mask_pos_exact", "?"),
            "mask_bit": metrics.get("eval_blend_mask_pos_bit_acc", "?"),
            "loss": metrics.get("eval_loss", "?"),
        }
        results.append(r)
        print(f"\n>>> {name}: F1={r['f1']:.4f} best_F1={r['best_f1']} dir={r['dir_acc']:.4f} mask_ex={r['mask_exact']:.4f}\n")
    else:
        print(f"\n>>> {name}: NO EVAL METRICS FOUND\n")
        results.append({"name": name, "error": "no metrics"})

print("\n" + "="*80)
print("ABLATION RESULTS SUMMARY")
print("="*80)
print(f"{'Experiment':<16} {'F1@thr':>8} {'Best F1':>8} {'Dir Acc':>8} {'Mask Ex':>8} {'Mask Bit':>8} {'Loss':>8}")
print("-"*80)
for r in results:
    if "error" in r:
        print(f"{r['name']:<16} {'ERROR':>8}")
    else:
        f1 = f"{r['f1']:.4f}" if isinstance(r['f1'], float) else str(r['f1'])
        bf1 = f"{r['best_f1']:.4f}" if isinstance(r['best_f1'], float) else str(r['best_f1'])
        da = f"{r['dir_acc']:.4f}" if isinstance(r['dir_acc'], float) else str(r['dir_acc'])
        me = f"{r['mask_exact']:.4f}" if isinstance(r['mask_exact'], float) else str(r['mask_exact'])
        mb = f"{r['mask_bit']:.4f}" if isinstance(r['mask_bit'], float) else str(r['mask_bit'])
        lo = f"{r['loss']:.4f}" if isinstance(r['loss'], float) else str(r['loss'])
        print(f"{r['name']:<16} {f1:>8} {bf1:>8} {da:>8} {me:>8} {mb:>8} {lo:>8}")

# Save results to JSON
with open("./training_outputs/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to training_outputs/ablation_results.json")
