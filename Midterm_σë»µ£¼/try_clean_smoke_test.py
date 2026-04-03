# ═══════════════════════════════════════════════════════════════════
# Smoke Test with Full Diagnostics
# 用于替换 baseline-qwen2-5.ipynb 的 Smoke Test cell
#
# 相比旧版增加：
# 1. 每条测试 prompt 打印完整生成诊断（token 数、是否 EOS、原始文本）
# 2. 打印 extract/sanitize/validate 每步的失败原因
# 3. 修复旧版 model.base_model.model 在 merge_and_unload 后不存在的 bug
# 4. 用 GENERATE_KWARGS (不含 max_length) 避免警告刷屏
# ═══════════════════════════════════════════════════════════════════

print('=' * 60)
print('SMOKE TEST: Pre-submission Diagnostics')
print('=' * 60)

# ── 1. 环境检查 ──
_impl = getattr(model.config, '_attn_implementation', 'unknown')
_cls = type(model).__name__
_dtype = next(model.parameters()).dtype
_dev = next(model.parameters()).device

print(f'\n[ENV] Model class:       {_cls}')
print(f'[ENV] attn_impl:         {_impl}')
print(f'[ENV] dtype:             {_dtype}')
print(f'[ENV] Device:            {_dev}')
print(f'[ENV] EOS IDs:           {EOS_TOKEN_IDS}')
print(f'[ENV] Batch size:        {CONFIG.get("inference_batch_size", 4)}')
print(f'[ENV] max_new_tokens:    {CONFIG["max_new_tokens"]}')
print(f'[ENV] gen_config.max_length: {getattr(model.generation_config, "max_length", "NOT SET")}')

_problems = []
if _impl != 'sdpa':
    _problems.append(f'attn_implementation={_impl}, expected sdpa')
if _dtype != torch.float16:
    _problems.append(f'dtype={_dtype}, expected fp16')
if hasattr(model, 'peft_config'):
    _problems.append('LoRA not merged (PeftModel still active)')
if getattr(model.generation_config, 'max_length', None) is not None:
    _problems.append(f'generation_config.max_length={model.generation_config.max_length} (will cause warnings)')

if _problems:
    print(f'\n*** {len(_problems)} PROBLEM(S) DETECTED ***')
    for p in _problems:
        print(f'  - {p}')
else:
    print('\n[OK] All environment checks passed.')

# ── 2. 详细单条推理测试 (debug=True) ──
test_prompts = [
    'a simple red circle on white background',
    'a green tree with brown trunk',
    'a blue five-pointed star icon',
]

print(f'\n{"─" * 60}')
print(f'Running {len(test_prompts)} test prompts with FULL DIAGNOSTICS...\n')

single_results = []
t0 = time.time()

for idx, p in enumerate(test_prompts):
    print(f'--- Prompt {idx+1}/{len(test_prompts)}: "{p}" ---')
    t1 = time.time()
    svg, status = generate_svg(p, debug=True)
    elapsed = time.time() - t1
    single_results.append((p, svg, status, elapsed))
    print(f'  Result: [{status}] {elapsed:.1f}s, SVG len={len(svg)}\n')

single_total = time.time() - t0
print(f'Single-mode total: {single_total:.1f}s for {len(test_prompts)} prompts')

# ── 3. EOS 工作状况判断（基于上面 debug=True 的输出已经可见） ──
# 如果所有测试 prompt 的耗时都接近 max_new_tokens / 速度，说明 EOS 从不生成
_avg_time = single_total / max(len(test_prompts), 1)
_expected_max_time = CONFIG['max_new_tokens'] / 30  # 30 tok/s 基准
if _avg_time > _expected_max_time * 0.8:
    print(f'\n*** WARNING: Avg {_avg_time:.1f}s/prompt suggests model always generates max_new_tokens. ***')
    print('    EOS/<|im_end|> may never be generated (training quality issue).')
else:
    print(f'\n[OK] Avg {_avg_time:.1f}s/prompt — model appears to stop before max_new_tokens.')

# ── 4. 批量推理测试 ──
bs = CONFIG.get('inference_batch_size', 4)
print(f'\n{"─" * 60}')
print(f'Running batch test (batch_size={bs})...\n')

t0 = time.time()
batch_results = generate_svg_batch(test_prompts)
batch_elapsed = time.time() - t0

for p, (svg, status) in zip(test_prompts, batch_results):
    print(f'  [{status:>20}] len={len(svg):>5}  prompt="{p}"')

print(f'\n  Batch total: {batch_elapsed:.1f}s for {len(test_prompts)} prompts')

# ── 5. 提交时间估算 ──
per_prompt = batch_elapsed / max(len(test_prompts), 1)
total_prompts = 1000
est_min = per_prompt * total_prompts / 60

print(f'\n{"═" * 60}')
print(f'SUBMISSION ESTIMATE')
print(f'  Speed:      {per_prompt:.2f} s/prompt')
print(f'  Prompts:    {total_prompts}')
print(f'  ETA:        {est_min:.0f} min ({est_min/60:.1f} hours)')
print(f'{"═" * 60}')

# ── 6. 质量汇总 ──
valid = sum(1 for _, s in batch_results if s == 'ok')
print(f'\n[QUALITY] Valid: {valid}/{len(batch_results)}')
for p, (svg, status) in zip(test_prompts, batch_results):
    if status == 'ok':
        print(f'  OK   "{p}" → {len(svg)} chars')
    else:
        print(f'  FAIL "{p}" → {status}')

# ── 7. 最终建议 ──
print(f'\n{"─" * 60}')
if valid == 0 and len(batch_results) > 0:
    print('*** ALL test SVGs failed. Check the diagnostic output above. ***')
    print('    Most common causes:')
    print('    1. Model generates text but not SVGs (training quality issue)')
    print('    2. Model generates SVGs but they fail XML parsing (sanitize_svg)')
    print('    3. Training data format mismatch (check format_chat template)')
    print('    Run the detailed diagnostics above to determine which.')
if est_min > 120:
    print(f'\n*** Estimated {est_min:.0f} min is too slow for Kaggle (limit ~540 min). ***')
elif est_min > 60:
    print(f'\nEstimated {est_min:.0f} min — within limits but could be faster.')
else:
    print(f'\nSpeed looks good for submission.')
print()
