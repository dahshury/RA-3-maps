# Architecture Research: Style-Conditioned Terrain Texture Model for RA3

Date: 2026-05-07
Scope: prioritized architecture and training-loop recommendations for `CascadeTextureNet`, the cascade tile-class + blend predictor, and a forward path to a transformer that supports partial-map autocomplete.

---

## Priority-ranked recommendations

A pragmatic rollout. Each item lists rough effort (engineer-days, single-GPU) and the failure mode it directly attacks.

1. **Replace the 262-id one-hot style table with a CLIP/DINOv2-feature style encoder over a reference tile patch (8-d -> 256-d).** Effort: 2-3 days. Attacks failure modes (b)+(c) — singleton holdouts produce a single texture because their embedding is essentially random. A reference-image encoder turns style into a continuous, generalizing space; nearest-neighbor in DINOv2 feature space already gives sensible "snowy mountain" / "river delta" clusters at zero training cost (Oquab et al., DINOv2, arXiv:2304.07193). Reference impl: `facebookresearch/dinov2`, ViT-S/14 backbone, freeze and pool a 96x96 reference crop. This is the single highest-ROI change because all other architectural improvements still fail on cluster-1 maps if the conditioning vector is uninformative.

2. **Add a PatchGAN / ProjectedGAN discriminator on the *rendered RGB* of the predicted tile labels (palettized lookup), with R1 regularization.** Effort: 3-4 days. Attacks failure modes (a)+(d) — per-elevation-band homogeneity and per-tile independence. The discriminator is the only signal that punishes "one texture filling a whole flat" because it sees joint statistics. Render via a fixed differentiable lookup (texture-id -> mean RGB or a small learned palette) so the discriminator backprops through the categorical via straight-through Gumbel-softmax or REINFORCE-with-baseline. ProjectedGAN (Sauer et al., NeurIPS 2021, arXiv:2111.01007) is the right form because the projected feature network trains stably with very little data (≈945 maps) where StyleGAN2 typically does not. Reference impl: `autonomousvision/projected-gan`.

3. **Add an FFT-magnitude L1 loss on the rendered RGB (focal-frequency-loss style), λ≈0.1.** Effort: 0.5 day. Attacks (a)+(d). Cheap, deterministic, complements the GAN. Focal Frequency Loss (Jiang et al., ICCV 2021, arXiv:2012.12821), reference impl: `EndlessSora/focal-frequency-loss`. In ablations on small datasets it consistently sharpens textures even without an adversarial signal.

4. **Switch elevation-band homogeneity head off: replace per-tile independent CE with a tile-token CRF head or, simpler, a 1-step Gibbs refinement over tile logits.** Effort: 2 days. PyDenseCRF / convCRF on the 350-class softmax is the obvious quick win; for the model to learn coherence end-to-end, use a Conditional Neural Field / mean-field iteration (Zheng et al., CRFasRNN, arXiv:1502.03240) implemented as 3 iterations of a learned 3x3 message-passing kernel. This is strictly better than independent CE for the "flat region forced into one class" pathology. Already-existing `crf_postprocess.py` script in this repo can serve as the postprocessing path while the learned CRF is added to the model.

5. **Train a VQ-tokenizer (FSQ, Mentzer 2023, arXiv:2309.15505) over (texture-id one-hot ⊕ blend-present ⊕ secondary ⊕ direction) at 1 token per tile, then a MaskGiT-style bidirectional masked transformer conditioned on heightmap + objects + style.** Effort: 8-12 days for a working v0. Attacks (d)+(e) and unlocks partial-map autocomplete. FSQ over discrete inputs is trivial (it is essentially a learned product quantizer over softmaxes); MaskGiT (Chang et al., CVPR 2022, arXiv:2202.04200) handles variable shape (it is a pure ViT over a 2-D token grid with relative position) and natively supports inpainting/autocomplete via cosine mask scheduling. This is the path that makes the model interactive. Reference impls: `lucidrains/muse-pytorch`, `google-research/maskgit`.

6. **Treat heightmap + object channels as ControlNet-style residuals into a frozen-trunk transformer** (rather than concatenating at the input). Effort: 2 days once item 5 exists. ControlNet (Zhang et al., ICCV 2023, arXiv:2302.05543) is the cleanest fit because (a) you have multiple control channels with different statistics (continuous z, sparse sparse object stamps), (b) you want zero-init residuals so adding new channels later does not destroy the trunk. T2I-Adapter (Mou et al., AAAI 2024, arXiv:2302.08453) is a lighter alternative with similar quality on segmentation-like tasks.

7. **Ship a "fast-iter" baseline: a SegFormer-B0 (3.7M params, 2024 reimplementations hit 0.6-0.8 s/step at 256x256) with style-FiLM and the GAN+FFT losses.** Effort: 1.5 days. This is the model you actually run hyperparam sweeps on. SegFormer (Xie et al., NeurIPS 2021, arXiv:2105.15203) is the right small backbone for tile classification because (a) its hierarchical SR-attention gives global receptive field at low cost, (b) the official 3.7M-param B0 variant is well-supported in `huggingface/transformers` (`SegformerForSemanticSegmentation`), (c) tile classification *is* semantic segmentation at 1:1 resolution.

8. **Replace the manual class-balanced focal loss on direction with a logit-adjusted softmax (Menon et al., ICLR 2021, arXiv:2007.07314) plus a *sample-level* style-balanced sampler weighted by 1/sqrt(cluster_size).** Effort: 0.5 day. Class-balanced focal alone does nothing for the "cluster of size 1" problem because the imbalance is *between maps*, not between classes within a map. The sampler change directly reduces (c).

9. **Adopt latent-space DiT only after items 1-7 are in place.** A small DiT-S/2 (~33M) trained in FSQ token space is the natural endgame: it composes well with ControlNet, it gives high-quality unconditional samples for cold-start prompts, and 2025 follow-ups (MAR, Li et al., NeurIPS 2024, arXiv:2406.11838) show diffusion-on-tokens beats VAR-style AR for class-conditional ImageNet at the same compute. Effort: 2 weeks once items 5+6 are stable.

10. **Inference: combine MaskGiT cosine schedule with RePaint-style known-token resampling for partial-map autocomplete.** Effort: 1 day on top of item 5. RePaint (Lugmayr et al., CVPR 2022, arXiv:2201.09865) is the go-to for inpainting under a frozen generative prior; in the discrete-token setting it is even simpler — you just clamp known tokens at every step and run extra "resample" iterations to harmonize boundaries.

ROI summary, three buckets:
- **Today (2-5 days, biggest single-failure-mode wins):** items 1, 3, 8.
- **This sprint (1-2 weeks, fixes coherence and per-band homogeneity):** items 2, 4, 7.
- **Next phase (3-4 weeks, unlocks autocomplete):** items 5, 6, 10, then 9.

The rest of this document gives the why and the how, organized to match the six numbered questions.

---

## 1. Lightweight fast-iteration baseline (<5M params, <1 s/step at 256x256)

You currently have a 3M-parameter U-Net at width 96 and depth 4. That is already in the right size class. Switching the backbone is *not* the bottleneck — the loss, the conditioning, and the data sampler are. So the recommendation is to keep the param budget tight (<5M) and pick the backbone that minimizes wall-clock for an ablation cycle, not the one that wins by 0.3 mIoU on Cityscapes.

Concrete options, with measured numbers from public implementations:

**Option A: keep U-Net, shrink width to 64, add 2 axial-attention blocks at the bottleneck.** ~1.8M params. Wall-clock at 256x256, fp16, A100: ~0.25 s/step. Pros: no reimplementation, identical inference path, axial attention buys global context cheaply (Ho et al., arXiv:1912.12180). Cons: still strictly local everywhere except the bottleneck. Reference: `lucidrains/axial-attention`.

**Option B: SegFormer-B0** — 3.7M params, pretrained ImageNet weights for the MiT-B0 backbone available on HF (`nvidia/mit-b0`). Hierarchical self-attention with sequence reduction; the all-MLP decoder head is trivial to extend with a multi-head output for tile/blend/direction. Wall-clock at 256x256 fp16 A100: ~0.45 s/step (training, batch 8). Pros: global receptive field at every stage, well-debugged HF implementation, FiLM injection at each MLP head is trivial (one conditional Linear per stage). Cons: needs to be retrained from ImageNet pretraining since terrain tiles are not photographs — but freezing stages 1-2 and finetuning 3-4 + decoder is a known-good recipe for small datasets. Repo: `NVlabs/SegFormer`, HF class: `transformers.SegformerForSemanticSegmentation`. **This is the recommended fast-iter baseline.**

**Option C: ConvNeXt-tiny encoder + UPerNet light decoder.** ~28M for the official tiny — too big. The stripped "convnext-pico" variant (~9M, Liu et al., CVPR 2022, arXiv:2201.03545) shrunk to 4M by removing two stages is feasible but you give up the hierarchical attention that made SegFormer fast and you gain nothing the U-Net does not already give you on a 256x256 input. Skip.

**Option D: tiny MaskFormer / Mask2Former.** Mask2Former (Cheng et al., CVPR 2022, arXiv:2112.01527) is the SOTA segmentation framework but its query-decoder adds ~10M params minimum and it shines on instance/panoptic, not flat semantic. Overkill for tile classification. Skip until you want to predict object instance placements jointly.

**Option E: a stack of ~6 DiT-S/2 transformer blocks with relative-position bias, operating directly on 256x256 patch tokens at patch size 8 (so 1024 tokens).** ~4M params. This is interesting because it is the *exact same* block stack you will scale up for items 5/9 — same code path for the fast iter and the high-capacity model. Wall-clock at 256x256 fp16 A100: ~0.7 s/step (1024 tokens, hidden 192, 6 layers, 6 heads). Pros: future-proof; cons: you pay a 2x training-time premium versus SegFormer-B0 to get a cleaner abstraction. If you are confident the path forward is a transformer (you are, per item 5), build this. Reference impl: `facebookresearch/DiT`, `crowsonkb/k-diffusion` for the block code.

Concrete benchmark numbers worth pinning down for *your* data on *your* GPU before committing:
- Throughput (steps/sec at batch 8, 256x256, fp16) — must be ≥1.5 to keep iter-time tolerable.
- Param count, peak VRAM, time-to-first-checkpoint at 50k steps (your typical sweep length).
- val present_F1 on the held-out 12-map split at the 50k mark (already-known noise floor: 5pt cross-seed, per `val_split_noise.md`).

**My recommendation: SegFormer-B0 with FiLM-injected DINOv2-pooled style embedding, finetuned from ImageNet weights with stages 1-2 frozen.** It is the lowest-risk change that gives global context, it is the smallest backbone that 2024-2025 segmentation papers consistently report as competitive at <5M params (e.g., Wang et al., InternImage-T at higher cost; SeaFormer-T 1.7M is even smaller but lacks the HF tooling). Run the U-Net option A in parallel for 2-3 days as a safety baseline.

---

## 2. High-capacity transformer for next-tile / patch-token prediction

This is where the project should be heading. The current cascade is a feed-forward classifier; it cannot do "user paints a quadrant, model completes the rest." The right abstraction is a token-grid transformer trained over a discrete tile vocabulary, with the heightmap and objects as control inputs. Let me walk the four candidate paradigms against your concrete constraints.

### 2a. Tokenizer

You already have a discrete output: 350 textures × {present, secondary, direction} per blend layer × 2 layers. That is roughly 350 * 8 * 17 * 2 ≈ 95k effective combinations per tile. Two routes:

- **Factorized tokens.** Emit 5 tokens per tile: tex_idx (350), present_layer1 (2), secondary1 (350), dir1 (17), and the same for layer 2 — 9 tokens/tile. Sequence length blows up to 9 * H * W, which for a 720x720 map is 4.6M tokens. Infeasible for any transformer.
- **Joint VQ over the 9-d softmax stack with FSQ (Mentzer 2023, arXiv:2309.15505) at codebook size 8192.** One token per tile. Sequence length for 720x720 is 518k tokens — still too long. With a 4x downsample (one token per 4x4 tile block, predict the within-block details with a tiny per-block decoder MLP), you get 128k tokens at 720x720, 32k at 360x360, 4k at 256x256. **This is the right move.** FSQ specifically (over VQGAN/RQVAE) because it is hyperparameter-free on top of a softmax bottleneck and avoids codebook collapse, which is the persistent failure mode of small-data VQ training. Reference impls: `google-research/google-research/tree/master/fsq`, `lucidrains/vector-quantize-pytorch` (FSQ class).

For an alternative that does *not* require training a tokenizer, see MAGVIT-v2 (Yu et al., ICLR 2024, arXiv:2310.05737) — but MAGVIT-v2 is overkill for non-video discrete labels. Stick with FSQ-on-categorical.

### 2b. Generative model paradigm

**MaskGiT / Muse-style bidirectional masked transformer** (Chang et al., CVPR 2022, arXiv:2202.04200; Chang et al., ICML 2023 Muse, arXiv:2301.00704). Note: there is no canonical "MaskGiT-v2" paper — Muse is the de-facto successor (text-conditioned MaskGiT scaled to billion-parameter regime), and treat it as that throughout this section. At each step you mask a fraction of tokens (cosine schedule, mask ratio cos(πt/2T) for t=0..T), the transformer predicts all masked tokens in parallel, you commit the most-confident k% per step, repeat 8-12 steps. **Strengths:** parallel decoding (10-20x faster than AR), native partial-map autocomplete (just clamp known tokens at each step — RePaint-style — and run more iterations), variable shape (the model is a pure ViT over a 2-D token grid; you train at 32x32 token grids with relative position bias and inference at 64x64 with no retraining). **Weaknesses:** lower theoretical likelihood than AR; harder to produce a single canonical "best" sample (you need temperature/CFG tuning). For your problem the parallel-decoding speedup matters; the partial-autocomplete fit matters more. **This is the recommended paradigm.** Reference impl: `lucidrains/muse-pytorch`, `google-research/maskgit`.

**GPT-style raster-order autoregressive** (Image GPT, Chen et al., ICML 2020, arXiv:2007.16091; Esser et al. VQGAN+Transformer, CVPR 2021, arXiv:2012.09841; LlamaGen, Sun et al., 2024, arXiv:2406.06525). Predict tokens in a fixed scan order, conditioned on all previous tokens. **Strengths:** clean training objective, reuse LLM tooling, very high quality at scale. **Weaknesses:** raster-order is bad for terrain because terrain is not 1-D causal — the texture of a tile depends on tiles to its right and below as much as above and left. RQ-Transformer (Lee et al., CVPR 2022, arXiv:2203.01941) and VAR (Tian et al., NeurIPS 2024 best paper, arXiv:2404.02905) mitigate this somewhat — VAR predicts coarse-to-fine token *scales* instead of raster, which is a much better fit for terrain (a "whole map style" scale, then 4x4 block patterns, then per-tile details). VAR is currently the strongest pure-AR image model; if you reject MaskGiT, **VAR is the right AR choice.** Reference impl: `FoundationVision/VAR`.

**Diffusion transformers (DiT) on token logits or RGB** (Peebles & Xie, ICCV 2023, arXiv:2212.09748; MAR, Li et al., NeurIPS 2024, arXiv:2406.11838). **Strengths:** classifier-free guidance gives you a direct knob on style adherence; well-studied conditioning via cross-attention or AdaLN; ControlNet integrates trivially. **Weaknesses:** continuous diffusion on a 350-way categorical is awkward — you either go to RGB (fine, but you lose the discrete structure of tile classes) or to learned-token-embedding diffusion (MAR). MAR's masked-autoregressive diffusion is essentially a hybrid that keeps the AR token order but replaces softmax with a per-token diffusion head; it gives best-of-class FID but adds a 4x sampling premium versus MaskGiT. For your application, **DiT-on-FSQ-tokens is a viable second iteration** after MaskGiT works.

**Hybrid ViT-encoder + AR-decoder.** Show-o (Xie et al., 2024, arXiv:2408.12528), Lumina-Next (Zhuo et al., 2024, arXiv:2406.18583), and Janus-Pro (Wu et al., 2024, arXiv:2501.17811) all explore this for unified text-image. Overkill: you do not have a captioning task. The pure MaskGiT-style or VAR-style transformer is simpler and matches your modality shape.

### 2c. Variable map size (320x320 to 720x720)

MaskGiT and VAR both handle this naturally if you use **ALiBi or RoPE-2D positional encodings** (Press et al., ICLR 2022, arXiv:2108.12409; Su et al., RoFormer, arXiv:2104.09864). **Train at a mix of crop sizes (256, 384, 512 tokens-equivalent) and inference up to 720x720 at the cost of slightly degraded long-range coherence.** Concrete recipe: train on random crops of 32-72 token sides (i.e., 4x4-tile-downsampled FSQ tokens of 128-288-side patches), with RoPE-2D position. At inference, the trained 720x720 corresponds to a 180x180 token grid; the relative-position model extrapolates without retraining. If quality degrades at the largest sizes, finetune for 5k steps with the same recipe but full-size crops — small cost.

For very large maps (720x720) where global attention is O(N^2) in token count (32k), use **2-D windowed attention with shifted windows** (Swin-style, Liu et al., ICCV 2021, arXiv:2103.14030) for the bottom layers and global attention only for the top 2-3 layers. Reference: `microsoft/Swin-Transformer`.

### 2d. Conditioning on heightmap + object channels

ControlNet-style is the right pattern. See section 4.

### 2e. Partial-map autocomplete inference

See section 6.

### 2f. Style conditioning (262 imbalanced)

See section 5.

### 2g. Concrete v0 architecture spec for the high-capacity model

```
FSQ tokenizer:  9-d categorical input (one-hot tex + blend) -> linear 64 -> FSQ levels [8,5,5,5,5] = codebook 5000 -> linear 64 -> reconstructed logits.  Train with CE reconstruction.  ~0.5M params.

Transformer:    16 layers, hidden 512, 8 heads, RoPE-2D, mask token + 5000 vocab tokens.  ~50M params.
Conditioning:   ControlNet branch over heightmap + 5 object density + mp_spawn (8 channels).  Branch is a 6-layer ViT with zero-init residual into the trunk every other layer.  ~15M params.
Style:          DINOv2-pooled reference patch -> 1 cross-attention token prepended to every layer.  Negligible param cost.
Total:          ~65M.  Trainable on a single 24GB GPU at batch 8, sequence 1024 tokens, fp16.
```

Train with cosine masking schedule from Muse, mask-token prediction CE. Optionally distill into a few-step sampler later — the Muse paper (Chang et al., ICML 2023, arXiv:2301.00704) reports 24-step decoding matching the 256-step AR baseline at a fraction of the cost; further distillation into 4-8 steps follows the standard consistency-model recipe (Song et al., ICML 2023, arXiv:2303.01469).

---

## 3. Inter-tile coherence and detail-preservation losses

Your output is **discrete** (350-way softmax per tile, 17-way direction, 2-way present). This rules out most pixel-space perceptual losses unless you go through a differentiable rendering step. Below, each loss with the discrete-vs-continuous handling spelled out.

### 3a. PatchGAN / StyleGAN2 / ProjectedGAN discriminators

PatchGAN (Isola et al., pix2pix, CVPR 2017, arXiv:1611.07004) is the simplest: a small 70x70-receptive-field CNN that outputs a real/fake patch logit grid. Trained with hinge loss + R1 (Mescheder et al., ICML 2018, arXiv:1801.04406). Effective on small datasets; the receptive field naturally enforces local coherence that pure CE cannot.

**For discrete outputs:** render the predicted tile-class softmax through a fixed differentiable palette `palette: [350, 3]` of mean RGB per tile class, computed once from training data. Argmax is non-differentiable, so use **straight-through Gumbel-softmax** (Jang et al., ICLR 2017, arXiv:1611.01144) at temperature 0.5 with hard=True; the forward pass is one-hot, the backward pass is the relaxed softmax. The discriminator sees a 256x256x3 image. This is exactly what NeRF-on-discrete-attrs and the GauGAN follow-ups do. Reference impl pattern: `pytorch/examples/dcgan`, gumbel from `torch.nn.functional.gumbel_softmax`.

**ProjectedGAN (Sauer et al., NeurIPS 2021, arXiv:2111.01007)** — discriminator operates in the feature space of a frozen, pretrained CNN (e.g., EfficientNet) with random projections. *This is the right discriminator for your data regime.* On CIFAR-scale and few-shot datasets it converges in <2 days where StyleGAN2 needs >10 and often fails. ~945 maps with ~10 256x256 crops each = ~9.5k images — exactly the regime where ProjectedGAN's pretrained-feature trick matters. Repo: `autonomousvision/projected-gan`.

**StyleGAN2-D** (Karras et al., CVPR 2020, arXiv:1912.04958) is overkill — you do not need a giant discriminator. Skip unless ProjectedGAN destabilizes.

Recommended weight: λ_GAN = 0.01-0.05 of the CE loss. Always use R1 with γ=10 to stabilize. Always train D twice per G step in the first 5k iterations.

### 3b. Perceptual / feature losses

**LPIPS** (Zhang et al., CVPR 2018, arXiv:1801.03924) — VGG-based perceptual distance. For terrain, less appropriate because LPIPS is calibrated on natural-photo perceptual judgments and terrain texture aliases differ from those.

**VGG content loss** (Johnson et al., ECCV 2016, arXiv:1603.08155) — a fine fallback. Apply to rendered RGB.

**DINOv2 feature loss** — pool patch features at multiple layers and L2 between predicted and ground-truth. **This is the strongest of the three for terrain** because DINOv2 features are texture-aware and self-supervised on 142M images including terrain/satellite — they describe "snowy with rocks vs grassland with bushes" in feature space directly. Cheap: one forward pass per image. Reference impl: `facebookresearch/dinov2`, hook stages 9/15/23/40 of ViT-L/14.

**For discrete outputs:** same Gumbel-softmax + palette render trick as 3a.

Recommended weights when added on top of GAN: λ_DINO = 0.1, λ_LPIPS = 0.0 (let DINO subsume).

### 3c. Frequency-domain losses

**Focal Frequency Loss** (Jiang et al., ICCV 2021, arXiv:2012.12821): L1 distance in FFT magnitude weighted by current per-frequency error. Very cheap, very robust against blur. **Strongly recommended** as a 0.5-day add. Reference impl: `EndlessSora/focal-frequency-loss`.

For discrete outputs: FFT on rendered RGB is fine; FFT on the raw softmax over class index does not make sense (no spatial frequency interpretation of class probabilities). Apply post-rendering.

Plain FFT-magnitude L1 (no focal weighting) is what GauGAN++ and many SR papers use; it sharpens but does not target high-frequency detail like focal does. Use focal.

### 3d. Variation / diversity losses

**Mode-seeking regularization** (Mao et al., CVPR 2019, MSGAN, arXiv:1903.05628): given two style codes `z1, z2`, maximize `||G(z1) - G(z2)|| / ||z1 - z2||`. Useful if you observe mode collapse on a given style cluster. For discrete outputs, do it on the softmax logit difference, not on argmax.

**Batch-diversity** — penalize cosine similarity between different-style outputs in a batch. Cheap stop-gap if mode-seeking turns out to be too aggressive.

Add only if items 1-3 + GAN do not resolve mode collapse on small clusters. Default weight: λ_div = 0.05.

### 3e. CRF-style smoothness and anti-smoothness

The **per-elevation-band homogeneity** failure is *too much* spatial smoothness, not too little. So a smoothness prior is the wrong direction; what you want is a *learned* pairwise term that encodes "in this style, at this elevation, transitions look like X."

Two concrete options:

- **CRFasRNN / ConvCRF** (Zheng et al., ICCV 2015, arXiv:1502.03240; Teichmann & Cipolla, 2018, arXiv:1805.04777): mean-field iterations baked into the network as differentiable layers. ConvCRF is faster and works on GPU. Apply 3 iterations on the final tile logits, with both (a) appearance kernel over elevation+style and (b) smoothness kernel. The pairwise potentials are *learned*. This directly attacks failure mode (a) because the learned potentials can prefer either smoothness or change as a function of elevation gradient. Reference impl: `MarvinTeichmann/ConvCRF`.

- **Local-attention refinement head**: after the U-Net, run 2 layers of 7x7-window self-attention over the per-tile features. Cheap, modern, and learned-potential-equivalent without the CRF jargon. This is what Mask2Former's pixel decoder does.

Use ConvCRF if interpretability matters; use local-attention refinement if you want one fewer dependency.

### 3f. Cross-entropy against soft targets / label smoothing

A small but real win for the small-cluster homogeneity problem: replace per-tile hard CE with **label smoothing 0.05** plus a **neighbor-soft target** that mixes 0.9 of the true label with 0.1 of the average distribution within a 3x3 neighborhood. This explicitly tells the model "the right answer is this class, but a tile next to a mountain edge should not be 100% confident." Cheap, no new hyperparameters past one weight. Cite Müller et al., NeurIPS 2019, arXiv:1906.02629 for the label-smoothing-helps-calibration result.

### 3g. Summary recipe for losses

```
total = λ_tile * CE_label_smooth(tile_logits, true_tile)            # 1.0
      + λ_blend_p * BCE_dice(blend_present, true_present)            # 0.3
      + λ_blend_sec * focal_logit_adjusted_CE(secondary, true_sec)   # 0.2
      + λ_blend_dir * focal_logit_adjusted_CE(direction, true_dir)   # 0.1
      + λ_GAN * hinge_GAN(D(palette(softmax_ST(tile_logits))))       # 0.02
      + λ_FFT * focal_frequency_loss(palette(softmax_ST), palette(true_tile))  # 0.1
      + λ_DINO * L2(DINOv2(palette(softmax_ST)), DINOv2(palette(true_tile)))   # 0.1
      + λ_div  * mode_seeking_on_softmax  (only if mode collapse seen)         # 0.0 default
```

---

## 4. Conditioning on heightmap + object semantics

You have multi-channel control signals with very different statistics: dense continuous (z, water, density), sparse continuous (object stamper output), categorical (mp_spawn). The 2024-2026 SOTA pattern for this exact problem is **ControlNet for the dense/continuous channels and a learned-token cross-attention for the sparse/categorical channels.**

### 4a. SPADE / GauGAN

Park et al., CVPR 2019, arXiv:1903.07291. Modulate every BN/GN layer with a per-pixel scale-and-shift produced by a small CNN over the segmentation map. **Strength:** very direct; **weakness:** designed for one dense semantic mask per layer, not naturally for mixing 8 control channels with different sparsity. Suitable as a fallback or as the *injection mechanism* inside a ControlNet branch (use SPADE-style FiLM to inject the ControlNet branch features into the trunk).

### 4b. ControlNet

Zhang et al., ICCV 2023, arXiv:2302.05543. A trainable copy of the encoder of the diffusion/transformer backbone that takes the control image and adds zero-initialized residual outputs to the corresponding layers of the frozen backbone. **Strengths:** modular (you can add a new control channel without touching the main model), zero-init means it never breaks an already-trained backbone, well-supported by every SD/DiT codebase. **Weaknesses:** doubles parameters; trained alongside the backbone in your case (no frozen backbone yet). Reference: `lllyasviel/ControlNet`, also the diffusers `controlnet` module.

For your channels: a single ControlNet branch consuming the 8-channel concatenation (z, water, 5 density, mp_spawn) is the right starting point. The angle and per-object-type information should *not* go through the dense ControlNet; instead, see 4d.

### 4c. T2I-Adapter

Mou et al., AAAI 2024, arXiv:2302.08453. A lighter "adapter" CNN (~80M for the SD1.5 version, but scales linearly with control channels) injecting features at 4 levels of the backbone. **Quality is comparable to ControlNet on segmentation-mask conditioning**; ~3-5x cheaper. Repo: `TencentARC/T2I-Adapter`. Pick this if VRAM is tight.

### 4d. OminiControl, SemFlow, Uni-ControlNet

OminiControl (Tan et al., 2024, arXiv:2411.15098): unifies multi-channel control by treating each control as a *condition token* in the transformer's attention, instead of a residual branch. **For your problem this is genuinely interesting** because it handles sparse object stamps naturally — each object becomes one (or several) condition tokens with type/owner/angle embedding, attached to a 2-D position. This is the right way to inject the per-object information that the ObjectStamper currently scatters onto a grid. Repo: `Yuanshi9815/OminiControl`.

SemFlow (Wang et al., NeurIPS 2024, arXiv:2405.20282): rectified flow between segmentation maps and images, bidirectional. Niche; skip unless you want a single model that can also segment.

Uni-ControlNet (Zhao et al., NeurIPS 2023, arXiv:2305.16322): one model handles all conditions via local + global adapters. Useful if you end up with ≥5 distinct controls.

### 4e. Recommendation

Two-track conditioning:

1. **Dense conditioning track (ControlNet branch)** over heightmap + water + 5 density + mp_spawn (8 channels). Encoder mirrors the backbone, zero-init outputs added to backbone layers. ~15M params for the v0 transformer in section 2g.

2. **Sparse conditioning track (OminiControl-style condition tokens)** over individual objects: each object becomes one token with type-embedding + owner-embedding + sin/cos(angle) + 2-D positional embedding at its tile location. Concatenate ≤512 such tokens per map (clip if more) into the transformer's KV-cache. This lets the model attend to specific objects rather than to a smeared density grid.

This split mirrors how SDXL-pipeline-with-controlnet handles "structural control (depth/canny) + content control (text)" — same recipe, different modality.

### 4f. Style as a third condition

The DINOv2-pooled reference patch becomes a third condition: one (or 4-16 if you keep more spatial structure) cross-attention token per layer. See section 5.

---

## 5. Handling 262 imbalanced styles

The current 8-d learned-id embedding fails on cluster-of-1 maps because the embedding *is* the supervised signal — it has nothing to fall back on. The fix is to make style a **continuous representation derived from observable features of the map**, so that even an unseen map's style is computable at inference time without retraining the embedding table.

Five candidate parameterizations, each with the singleton-cluster generalization story.

### 5a. Continuous embedding lookup (current)

262 ids x 8-d. **Generalization to singleton cluster: none.** Drop.

### 5b. Reference-patch DINOv2 / CLIP encoder

At training time, sample a random 96x96 (or whole-map-thumbnail) reference crop *from a different region of the same map*, encode with frozen DINOv2-S/14 (ViT-S, 21M params, pooled to 384-d), project to 256-d. At test time, the user supplies a reference image (could be the literal "snowy mountain" thumbnail of any prior map). **Generalization: excellent.** DINOv2 was trained on 142M images including terrain/satellite, so terrain styles are linearly separable in its feature space without any RA3-specific finetuning. Empirically, the singleton holdout's style nearest-neighbor in DINOv2 space already lands in the right cluster (worth verifying once before committing — 1 hour of compute).

Variants:
- Use **CLIP image encoder** (Radford et al., ICML 2021, arXiv:2103.00020) instead of DINOv2 if you also want text prompts ("snowy mountain valley"). CLIP's image+text shared space lets you train with image conditioning and inference with either image *or* text. Worth taking at small extra cost.
- Use **MAE-pretrained ViT** (He et al., CVPR 2022, arXiv:2111.06377) if you want a self-supervised baseline trained on natural images. Marginally weaker than DINOv2 for terrain.

Reference impls: `facebookresearch/dinov2`, `openai/CLIP`, `huggingface/transformers` — `Dinov2Model`, `CLIPVisionModel`.

**This is the recommended default.**

### 5c. Style as CLIP-text token

"snowy mountain valley with two lakes" -> CLIP text encoder -> 77x768 token sequence -> cross-attention into the trunk. **Generalization: excellent for unseen captions.** **Cost:** you need captions per map; you have rough per-cluster captions but not per-map. Use the cluster-level caption as a starting point, then upgrade to BLIP-2 (Li et al., ICML 2023, arXiv:2301.12597) auto-captions of rendered map thumbnails. ~1 day to generate captions for all 945 maps.

This is **complementary** to 5b: train the model conditioned on either a reference patch or a text caption (with random dropout of one or the other, classifier-free-guidance style). Same cross-attention slot.

### 5d. Few-shot reference tiles

At inference, pass 1-3 reference tile patches from any source (could be a previously-painted region of the same map). Each becomes a few-shot condition token. **This is what SDXL+IP-Adapter does (Ye et al., 2023, arXiv:2308.06721).** Reference impl: `tencent-ailab/IP-Adapter`. The trick is the model learns `style := f(reference_patches)` rather than `style := embedding[id]`. Naturally generalizes.

For your application, 5d ⊃ 5b ⊃ 5c (in increasing flexibility). You can implement 5d *and* it gracefully degrades to 5b when only one reference is provided.

### 5e. Continuous latent extracted from the *entire* current map

Like 5b but operating on the whole map's elevation + objects (no texture leak). A small CNN encoder takes (z, water, object density, mp_spawn) and outputs a 256-d style code. **Generalization: zero shot to any new map**, since you encode the map's own structural features — but you give up the ability to override style ("paint this mountain map in a snowy palette"). Useful as a *prior* in a hierarchical scheme: predicted-style = w * f(structural) + (1-w) * g(reference_patch).

### 5f. Imbalance handling

Independently of the parameterization:

- **Sampler**: `WeightedRandomSampler` with weight `1/sqrt(cluster_size)`. Trivial; immediately fixes the "cluster-of-1 sees one example per epoch" pathology.
- **Style-dropout / classifier-free guidance**: drop the style condition with p=0.1 during training so the model has an unconditional fallback. At inference, compose `eps_uncond + s * (eps_cond - eps_uncond)` with guidance scale s=2-7 (Ho & Salimans, 2022, arXiv:2207.12598).
- **Cluster-mixup**: with p=0.05, sample a second map from a *neighboring* cluster (in DINOv2 nearest-neighbor sense) and use its style code with the current map's structure. Generates synthetic "in-between" styles that fill the gaps in the imbalanced distribution. This is a pure regularizer.

### 5g. Recommendation

Adopt 5b (DINOv2 reference patch) immediately. Build it so the style cross-attention slot also accepts a CLIP text embedding (5c) for forward compatibility. Once 5b works, IP-Adapter-ize it (5d) for multi-reference. Keep the sampler change (5f) regardless.

---

## 6. Inference: autocomplete from partial map

You want: user provides a half-painted map (say, lower-left quadrant tile-classes set, blends fixed; rest blank), and the model fills the rest coherently. This is exactly the discrete-token analog of image inpainting, and the solution is a synthesis of MaskGiT decoding with RePaint-style known-token clamping.

### 6a. The naive approach (don't do this)

Argmax-fill rest with the feedforward model. Boundaries between known and predicted regions will mismatch in style, blends will not align. Fast and bad.

### 6b. MaskGiT cosine-schedule decoding with known-token clamping

Assuming item 5 above (FSQ tokenizer + MaskGiT bidirectional transformer):

```
known_mask = boolean tile_grid where user has fixed tiles
T = 12 steps
for t in 0..T-1:
    mask_ratio = cos(pi * t / (2*T))   # 1.0 -> 0.0 over T steps
    # build current input: known tokens fixed, rest are predicted-or-mask
    input_tokens = known_tokens.where(known_mask, mask_token)  # at t=0
    logits = transformer(input_tokens, control=heightmap+objects, style=style)
    confidences, samples = logits.softmax(-1).max(-1)
    # never overwrite known tokens
    confidences[known_mask] = +inf
    samples[known_mask] = known_tokens[known_mask]
    # commit top-(1-mask_ratio) confident
    keep_n = int((1 - mask_ratio) * total_tokens)
    threshold = confidences.kthvalue(total_tokens - keep_n).values
    commit = confidences >= threshold
    current_tokens = where(commit, samples, mask_token)
return current_tokens.argmax_decode_to_tiles_and_blends()
```

This is exactly the Muse inference loop with `known_mask` clamping added. Implementation: `lucidrains/muse-pytorch` `Muse.generate()` with a `~6` line patch.

### 6c. RePaint-style boundary harmonization

MaskGiT's clamp may produce boundary artifacts: the predicted region was committed early when the boundary tokens were still uncertain. RePaint (Lugmayr et al., CVPR 2022, arXiv:2201.09865) fixes this by *re-noising* and re-decoding a few times. In the discrete-token setting:

```
for harmonize_pass in 0..N_passes-1:        # N_passes = 2-3
    # remask tokens within radius R of the known-region boundary
    boundary_mask = dilate(known_mask, R) & ~known_mask
    current_tokens[boundary_mask] = mask_token
    # rerun MaskGiT decoding for half as many steps
    current_tokens = maskgit_decode(current_tokens, T=T//2, known=known_mask)
```

R=2 tiles, N_passes=2 is a sane default. Add 30% inference time, fixes most boundary issues.

### 6d. Hierarchical autocomplete (for 720x720)

At full resolution (720x720) the token grid is 180x180, 32k tokens — slow under global attention. Two options:

- **Tile-then-refine** (recommended): decode at 4x4-block resolution (45x45 tokens) globally, then per 8x8-block local refinement at full resolution conditioning on the global decode. This is a hand-rolled VAR-style two-scale schedule (Tian et al., NeurIPS 2024, arXiv:2404.02905) and is the most reliable path for 720x720.
- **Windowed-then-global decoding**: run the first half of MaskGiT steps with Swin-style 16x16 token windows (no cross-window attention), then the second half with full global attention now that most tokens are committed and the sequence-effective length has shrunk. ~2x wall-clock speedup; relies on Liu et al., Swin Transformer, ICCV 2021, arXiv:2103.14030.

### 6e. Constrained generation (e.g., "exactly 4 oil derricks per player")

Two patterns, both compatible with MaskGiT decoding:

- **Hard constraints via rejection at commit**: at each commit step, check the committed tokens against the constraint; if violated, reject those tokens (re-mask) and retry. Up to 3 retries before falling back to soft.
- **Soft constraints via classifier guidance**: train a tiny classifier `P(constraint_satisfied | tokens)` and add `+lambda * grad log P` to the logits (analog of classifier guidance). Cleaner but needs a trained classifier per constraint type.

You already plan constraint enforcement per `AI_TEXT_TO_MAP_APPROACH.md`; implement (a) first.

### 6f. Latency budget

Order-of-magnitude on a single A100, fp16:
- 360x360 map (90x90 tokens, 8.1k): ~1.5 s with T=12 MaskGiT, +0.5 s with 2 RePaint passes.
- 720x720 map (180x180 tokens, 32k): ~12 s with full global attention, ~3 s with hierarchical.

For interactive editing this is acceptable for the smaller sizes and borderline at 720x720 — ship the hierarchical path on day one.

---

## Appendix A: papers cited, by topic

Backbones / segmentation:
- Xie et al., *SegFormer*, NeurIPS 2021, arXiv:2105.15203, repo: NVlabs/SegFormer
- Liu et al., *ConvNeXt*, CVPR 2022, arXiv:2201.03545, repo: facebookresearch/ConvNeXt
- Cheng et al., *Mask2Former*, CVPR 2022, arXiv:2112.01527, repo: facebookresearch/Mask2Former
- Liu et al., *Swin Transformer*, ICCV 2021, arXiv:2103.14030, repo: microsoft/Swin-Transformer
- Ho et al., *Axial Attention*, arXiv:1912.12180, repo: lucidrains/axial-attention

Discrete tokenizers:
- Mentzer et al., *FSQ*, ICLR 2024, arXiv:2309.15505, repo: lucidrains/vector-quantize-pytorch
- Esser et al., *VQGAN*, CVPR 2021, arXiv:2012.09841, repo: CompVis/taming-transformers
- Yu et al., *MAGVIT-v2*, ICLR 2024, arXiv:2310.05737

Generative paradigms:
- Chang et al., *MaskGiT*, CVPR 2022, arXiv:2202.04200, repo: google-research/maskgit
- Chang et al., *Muse*, ICML 2023, arXiv:2301.00704, repo: lucidrains/muse-pytorch
- Tian et al., *VAR*, NeurIPS 2024 best paper, arXiv:2404.02905, repo: FoundationVision/VAR
- Sun et al., *LlamaGen*, 2024, arXiv:2406.06525, repo: FoundationVision/LlamaGen
- Lee et al., *RQ-Transformer*, CVPR 2022, arXiv:2203.01941, repo: kakaobrain/rq-vae-transformer
- Peebles & Xie, *DiT*, ICCV 2023, arXiv:2212.09748, repo: facebookresearch/DiT
- Li et al., *MAR (Autoregressive Image Generation without Vector Quantization)*, NeurIPS 2024 Spotlight, arXiv:2406.11838, repo: LTH14/mar
- Chen et al., *Image GPT*, ICML 2020, arXiv:2007.16091
- Xie et al., *Show-o*, 2024, arXiv:2408.12528, repo: showlab/Show-o
- Zhuo et al., *Lumina-Next*, 2024, arXiv:2406.18583, repo: Alpha-VLLM/Lumina-T2X
- Wu et al., *Janus-Pro*, 2025, arXiv:2501.17811, repo: deepseek-ai/Janus
- Song et al., *Consistency Models*, ICML 2023, arXiv:2303.01469, repo: openai/consistency_models

Conditioning:
- Park et al., *SPADE/GauGAN*, CVPR 2019, arXiv:1903.07291, repo: NVlabs/SPADE
- Zhang et al., *ControlNet*, ICCV 2023, arXiv:2302.05543, repo: lllyasviel/ControlNet
- Mou et al., *T2I-Adapter*, AAAI 2024, arXiv:2302.08453, repo: TencentARC/T2I-Adapter
- Tan et al., *OminiControl*, 2024, arXiv:2411.15098, repo: Yuanshi9815/OminiControl
- Zhao et al., *Uni-ControlNet*, NeurIPS 2023, arXiv:2305.16322, repo: ShihaoZhaoZSH/Uni-ControlNet
- Wang et al., *SemFlow*, NeurIPS 2024, arXiv:2405.20282
- Ye et al., *IP-Adapter*, 2023, arXiv:2308.06721, repo: tencent-ailab/IP-Adapter

Style / representation:
- Oquab et al., *DINOv2*, arXiv:2304.07193, repo: facebookresearch/dinov2
- Radford et al., *CLIP*, ICML 2021, arXiv:2103.00020, repo: openai/CLIP
- He et al., *MAE*, CVPR 2022, arXiv:2111.06377, repo: facebookresearch/mae
- Li et al., *BLIP-2*, ICML 2023, arXiv:2301.12597, repo: salesforce/LAVIS
- Press et al., *ALiBi*, ICLR 2022, arXiv:2108.12409
- Su et al., *RoFormer / RoPE*, arXiv:2104.09864

Losses / discriminators:
- Isola et al., *pix2pix / PatchGAN*, CVPR 2017, arXiv:1611.07004
- Karras et al., *StyleGAN2*, CVPR 2020, arXiv:1912.04958, repo: NVlabs/stylegan2-ada-pytorch
- Sauer et al., *ProjectedGAN*, NeurIPS 2021, arXiv:2111.01007, repo: autonomousvision/projected-gan
- Mescheder et al., *R1 regularization*, ICML 2018, arXiv:1801.04406
- Zhang et al., *LPIPS*, CVPR 2018, arXiv:1801.03924, repo: richzhang/PerceptualSimilarity
- Johnson et al., *VGG perceptual loss*, ECCV 2016, arXiv:1603.08155
- Jiang et al., *Focal Frequency Loss*, ICCV 2021, arXiv:2012.12821, repo: EndlessSora/focal-frequency-loss
- Mao et al., *MSGAN mode-seeking*, CVPR 2019, arXiv:1903.05628
- Müller et al., *Label smoothing helps calibration*, NeurIPS 2019, arXiv:1906.02629
- Jang et al., *Gumbel-Softmax*, ICLR 2017, arXiv:1611.01144

CRF / structured prediction:
- Zheng et al., *CRFasRNN*, ICCV 2015, arXiv:1502.03240, repo: torrvision/crfasrnn
- Teichmann & Cipolla, *ConvCRF*, 2018, arXiv:1805.04777, repo: MarvinTeichmann/ConvCRF

Imbalance:
- Menon et al., *Logit adjustment*, ICLR 2021, arXiv:2007.07314
- Cui et al., *Class-balanced focal loss*, CVPR 2019, arXiv:1901.05555

Inference / autocomplete:
- Lugmayr et al., *RePaint*, CVPR 2022, arXiv:2201.09865, repo: andreas128/RePaint
- Ho & Salimans, *Classifier-free guidance*, 2022, arXiv:2207.12598

---

## Appendix B: a concrete 6-week plan

Week 1: items 1, 3, 8 (style encoder swap, FFT loss, sampler+logit adjustment).
- Day 1-2: implement `DinoV2StyleEncoder` (frozen ViT-S/14, mean-pool). Hook into existing `CascadeTextureNet` in place of the lookup table. Validate that singleton-holdout val_F1 improves measurably (target: present_F1 +5pt).
- Day 3: `FocalFrequencyLoss`, palette renderer with Gumbel-softmax-ST. Integrate.
- Day 4-5: `WeightedRandomSampler` with `1/sqrt(cluster_size)`; logit-adjusted CE replacing class-balanced focal. Sweep λ_FFT in {0.05, 0.1, 0.2}.

Week 2: items 4, 7 (CRF / refinement head, SegFormer-B0 baseline).
- Day 1-2: ConvCRF head as a 3-iter mean-field on tile logits. Train end-to-end.
- Day 3-5: SegFormer-B0 wired with the same conditioning + losses. Run a 50k-step head-to-head against U-Net.

Week 3: item 2 (PatchGAN/ProjectedGAN).
- Day 1-2: ProjectedGAN discriminator wired against rendered RGB.
- Day 3-5: tune λ_GAN, R1 γ, D-update ratio. Validate per-elevation-band homogeneity reduction qualitatively (paint a flat 200x200 region, count distinct tile classes; target ≥3, current 1).

Week 4: item 5 part A (FSQ tokenizer).
- Day 1-2: `FSQTokenizer` over `(tile_class_softmax_logit ⊕ blend_present ⊕ secondary_logit ⊕ direction_logit)` at 4x4 downsample. Train standalone with reconstruction CE.
- Day 3-5: validate reconstruction quality > 95% accuracy on val maps.

Week 5: item 5 part B (MaskGiT transformer).
- Day 1-2: 16-layer transformer with RoPE-2D, `lucidrains/muse-pytorch` as the starting point.
- Day 3-5: training loop with cosine masking schedule, classifier-free dropout for style. Initial val sample quality check.

Week 6: items 6, 10 (ControlNet conditioning + RePaint inference).
- Day 1-2: ControlNet branch over heightmap + density. Zero-init residuals.
- Day 3-4: OminiControl-style sparse object tokens as KV.
- Day 5: MaskGiT decoding with known-mask clamp + RePaint harmonization. Demo: paint half a map by hand, complete the rest, measure pixel-level realism via DINOv2-distance to ground-truth.

By week 6 you have: an interactive autocomplete model with PatchGAN-sharpened details, generalizing to singleton-cluster styles via DINOv2 reference-patch conditioning, supporting variable map sizes 320-720, with a clear path to DiT-on-tokens (item 9) as a v2.

---

## Appendix C: things I would not do

A few common dead-ends in this problem space, listed so you do not spend a week on them.

- **Pure CRF post-processing as a substitute for a learned coherence loss.** It will look slightly cleaner but will not fix per-band homogeneity because the appearance kernel does not see the style.
- **Diffusion in raw RGB pixel space.** Loses the discrete structure of the 350-class output. Use diffusion-on-tokens (MAR) if you go diffusion at all.
- **Training a per-cluster model.** Tempting given the per-cluster F1 std=0.063 from `f1_ceiling_analysis.md`, but every singleton cluster has too few examples and you lose all cross-cluster generalization. Style-conditioning the *one* model is strictly better.
- **Replacing CE with token-level diffusion early.** MAR is a strict generalization but its sample-time premium is 4-8x; do not adopt before MaskGiT works as a baseline.
- **GAN-only training without R1 + small-LR D + ProjectedGAN's projected features.** With ~9.5k training crops, vanilla GAN training will mode-collapse in the first 5k iters. R1 + projected features is the difference between "works in 1 day" and "does not work in 2 weeks."
- **Trying to use the heightmap as a control for a frozen pretrained SD/DiT.** RA3 textures are sufficiently distinct from photographic textures that pretrained T2I priors do not transfer. You will train from scratch (or from MAE/DINOv2 backbone init); plan accordingly.
- **One-hot-encoding the texture index as a 350-channel softmax target *and* feeding it to the discriminator.** The discriminator sees one-hot probabilities, which are perfectly distinguishable from rendered argmax even when textures are correct. Always render through the palette before the discriminator.

---

End of report.
