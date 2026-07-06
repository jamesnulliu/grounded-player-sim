# Milestone G — LLM-agent deployment (RESOLVED 2026-07-05)

**Outcome.** The 2026-07 pivot *aspired* to make the LLM the headline (via a
hidden≫verbal win) and to shore up the board-native proof with a strong Maia
backbone. **Neither panned out, and that's the honest result:**

* **G3 (done):** injected state helps a real LLM's think-time (verbal−none
  −0.005, 3 seeds), but **hidden does NOT beat verbal** — the LLM reads the note
  *semantically*, so text is the efficient channel. hidden≫verbal stays
  board-native (RQ6); the LLM ordering is the reverse (a **backbone-dependent**
  finding). So the LLM is a **deployment** result, not the headline.
* **G1 / Maia (deprioritized, do not run):** a strong backbone absorbs the move
  signal, and the timing headline is backbone-independent by construction — so
  Maia can't help (see the G1 row below).

**Net: consolidate.** Board-native (RQ6) stays the headline; lead with timing;
move + the LLM are honest secondary results. The runbook below is retained for
reference (what G1–G4 *would* entail).

---

## 1. The claims under test

| Exp | Claim | Decision rule |
|-----|-------|---------------|
| **G1** | ~~Latent beats memoryless on timing with a strong Maia backbone.~~ **DEPRIORITIZED (2026-07-05) — do not run.** Two reasons: (a) a strong backbone *absorbs the move signal* (conv already −0.027→−0.003; latent+Maia move ≈ null likely — a result that *weakens* the paper); (b) the timing head reads **only the latent**, so Maia **cannot change** the timing result — it is backbone-independent by construction. So the weak-backbone objection is answered *architecturally*, not by running Maia. | N/A — skip. Defend timing analytically (timing head is latent-only); lead with timing, report move as honest secondary. |
| **G2** | In an actual LLM, the state helps **think-time** prediction (dense completion-NLL SFT) — the sharp probe RL was too sparse for. | with-state NLL < no-state NLL on the held-out tail. |
| **G3** | **DONE (v1, 3 seeds) — hidden does NOT beat verbal in the LLM.** verbal wins (hidden−verbal +0.0034); state still helps (verbal−none −0.005). The LLM reads the note *semantically*, so text is the efficient channel. hidden≫verbal stays board-native (RQ6); the LLM ordering is the reverse. | The honest claim is the **backbone-dependent ordering**, not "hidden wins in the LLM." `results/g3_llm.txt`, `scripts/g3_hidden.py`. |
| **G4** | The per-individual evolving latent adds value over **released** SOTA (ChessMimic / Allie / Maia-3), not a reconstructed proxy. | (baseline+z) − baseline < 0 on held-out timing/move, per released model. |

**Gating order (RESOLVED).** G3 ran first. **hidden < verbal did NOT hold** in
the LLM (verbal wins, hidden−verbal +0.0034) — so, per the pre-registered rule,
the LLM is **not** the hidden≫verbal headline. It is the *deployment* (injected
state helps think-time, verbal−none −0.005) plus the **backbone-dependent
channel-ordering** finding, and **board-native (G1/RQ6) stays the headline**.
design.md §10 applied: the emphasis call was made on the numbers.

---

## 2. What is already landed (CPU-tested, no GPU)

The whole hidden channel is built and unit-tested (`tests/test_hidden_prefix.py`
+ `tests/test_llm_hidden.py`, in the 151-test suite):

* **`gps.policy.hidden_prefix.HiddenPrefixProjector`** — the trainable
  `latent → [n_prefix, hidden_size]` soft-prompt bridge (lazy torch, forked-RNG
  seed, `parameters()` for joint SFT). Mirrors `NeuralInjector` so they co-train.
* **`prepend_prefix()`** — the exact "prefix rows first, then token embeddings"
  concat the HIDDEN forward performs. A **toy-LM mechanism check** proves the
  projector alone (frozen LM) measurably steers completion loss down — the
  hidden channel carries trainable signal *before* we spend GPU.
* **`gps.policy.sglang_backbone`** — HIDDEN wiring: `enable_hidden`, `latent_dim`,
  `projector()`, `hidden_prefix_embeds()`; `build_prompt` leaves the text
  byte-identical for HIDDEN (channel-only RQ6); `move_logprobs` routes HIDDEN and
  **fails loudly** at the GPU `input_embeds` boundary (never drops the latent).
* **`gps.experiments.llm_hidden`** — the G2/G3 SFT entry point:
  `build_examples()` (causal per-decision (prompt, completion, latent) assembly;
  `hidden` carries the full evolving latent, prompt identical to `none`; `verbal`
  adds the note) and `assemble_completion_step()` (input embeds + attention mask
  + the label mask that scores **only** the completion). `run_hidden_sft_
  condition()` sketches the fit loop; `main()` documents the wiring.

---

## 3. GPU runbook

All fit on the existing **2×A100**; the binding constraint is dev time. Run the
Maia track and the LLM track in parallel, one A100 each.

### G3 / G2 — hidden-vs-verbal LLM SFT (run first)

Resources: 1×A100 (LoRA/frozen base ~30–40 GB; cheaper than the full-param SFT
already run). Qwen3 weights cached (~16 GB). Set `HF_HUB_OFFLINE=1`.

1. **Wire `run_hidden_sft_condition`** (it already composes the tested pieces):
   `transformers.AutoModelForCausalLM` + tokenizer, optional PEFT LoRA
   (`target_modules` q/k/v/o); build a
   `HiddenPrefixProjector(latent_dim, model.config.hidden_size, n_prefix)`.
2. Fit a `NeuralInjector` per player (or load the board-native one), then for
   `channel in ("none", "verbal", "hidden")`: `build_examples(traj, injector,
   SGLangBackbone(enable_hidden=True, latent_dim=…), channel=…, target="time")`
   and `run_hidden_sft_condition(model, tok, examples, projector=…)`.
3. **Report** held-out `hidden − verbal` and `with-state − none`, ≥3 seeds,
   bootstrap over players. Log to W&B `gps-llm-sft-hidden`
   (`WANDB_ENTITY=jamesnulliu-university-of-southern-california`).
4. Also wire the **sglang `input_embeds`** path in
   `SGLangBackbone._hidden_move_logprobs` (prepend the projected rows to each
   `"…\nMove: <move>"` continuation's token embeds) for hidden *inference*.

Target: `target="time"` is the primary channel (state → timing is the robust
signature); also run `target="move"`.

### G1 — Maia-2 D-vs-B — DEPRIORITIZED (do NOT run; kept for reference)

**Why we are not running this (2026-07-05):** a strong backbone absorbs the move
signal (conv already dropped move D−B −0.027→−0.003), so latent+Maia on moves is
likely a null that *weakens* the paper; and the timing headline is
backbone-independent by construction (the timing head reads only the latent), so
Maia cannot change it. The weak-backbone objection is answered *architecturally*.
The steps below are what running it *would* entail, if a reviewer ever insists.

Resources: 1 GPU ≥16 GB (Maia-2 is tens of M params), <1 GB weights.

1. **`gps/policy/maia_backbone.py`** — load the pretrained Maia-2 checkpoint,
   adapt its board encoding, and expose `encode_batch` / `trajectory_loss` /
   `per_traj_move_nll` (the protocol `board_native.py` defines). Latent
   conditioning: a small hidden-vector adapter before Maia's policy head; the
   **timing head reads only the latent** (as in board_native — that's why timing
   is backbone-invariant). NOTE: not yet written — needs the real weights +
   Maia-2's exact encoding/policy-vocabulary; do not ship a scratch tower under
   the Maia name.
2. Rerun the E-C timing/move D-vs-B (`gps train-ec … --backbone maia`) on the
   clocked cohorts. Expect the timing win to hold; the move ceiling rises.

### G4 — head-to-head vs released SOTA

Benchmark against **released weights**, not the hand-built Elo+clock+complexity
proxy (E-C6, Spearman 0.41 ≈ ChessMimic): **ChessMimic** (code+weights out),
**Allie**, **Maia-3**. Report `(baseline + z) − baseline` per model.

---

## 4. Reproduce the CPU scaffolding

```bash
pip install -e '.[dev]' && pip install -e '.[train]'   # torch for the projector
pytest tests/test_hidden_prefix.py tests/test_llm_hidden.py -q   # 20 tests
```
