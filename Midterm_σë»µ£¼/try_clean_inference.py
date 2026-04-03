# ═══════════════════════════════════════════════════════════════════
# Cell 17 综合修复方案 v3
# 用于替换 baseline-qwen2-5.ipynb 的 Cell 17 + Smoke Test
#
# 解决的三个问题：
#
# P1: RuntimeError — Unsloth fast inference 在 Qwen2.5 上 cos/sin
#     缓存维度不对齐 (analysis_to_this_error.md 根因分析)
#     → 修复：不用 Unsloth fast path，走干净 HF generate
#
# P2: valid=0 fallback=1000 — 所有生成都走了 fallback
#     → 修复：添加详细诊断输出，定位是 extract/sanitize/validate
#       哪一步失败
#
# P3: 24.5s/prompt 6.8h 总耗时 — 从不提前停止
#     → 修复：正确清理 generation_config (消除 max_length=32768
#       冲突)，确保 EOS 生效，SDPA 加速
#
# 用法：复制到 Colab / Kaggle 的 Cell 17 位置
# ═══════════════════════════════════════════════════════════════════

import gc, importlib, time, re
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# ── 路径配置 ──
# Colab（默认生效）
_LORA_DIR = CONFIG['output_dir']
# Kaggle（需要时取消注释，注释掉上面那行）：
# _LORA_DIR = '/kaggle/input/your-lora-dataset'

# ════════════════════════════════════════════════════════════
# Phase 1: 环境清理 — 无条件释放训练态对象
# ════════════════════════════════════════════════════════════

# 1a. 清除 Unsloth 对 qwen2 模块的全局补丁
import transformers.models.qwen2.modeling_qwen2 as _qwen2
importlib.reload(_qwen2)
print('[P1] Reloaded qwen2 module — Unsloth forward patches removed')

# 1b. 删除所有训练态模型对象（不复用！这是分析文档的核心建议）
try:
    del model
except NameError:
    pass
try:
    del base_model
except NameError:
    pass
try:
    del trainer
except NameError:
    pass
try:
    del peft_model
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()
_free_gb = torch.cuda.mem_get_info()[0] / 1e9
print(f'[P1] GPU cleared. Free VRAM: {_free_gb:.1f} GB')

# ════════════════════════════════════════════════════════════
# Phase 2: 加载干净模型 + LoRA 合并
# ════════════════════════════════════════════════════════════

# 2a. 从 HuggingFace 加载基础模型，显式指定 attn_implementation
#     这比 post-hoc 设 config._attn_implementation 更可靠
base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG['model_name'],
    device_map='auto',
    torch_dtype=torch.float16,
    attn_implementation='sdpa',
)
print(f'[P2] Base model loaded: {type(base_model).__name__}')

# 2b. 全量替换 __class__ → 干净的 HF 类（belt-and-suspenders）
#     即使 importlib.reload 已还原模块级定义，from_pretrained 可能
#     通过缓存的 AUTO_MODEL_MAPPING 引用了旧的（被 Unsloth 补丁过的）类
base_model.__class__ = _qwen2.Qwen2ForCausalLM
base_model.model.__class__ = _qwen2.Qwen2Model
_n_layers = len(base_model.model.layers)
for _layer in base_model.model.layers:
    _layer.__class__ = _qwen2.Qwen2DecoderLayer
    _layer.self_attn.__class__ = _qwen2.Qwen2Attention
    _layer.mlp.__class__ = _qwen2.Qwen2MLP
    _layer.input_layernorm.__class__ = _qwen2.Qwen2RMSNorm
    _layer.post_attention_layernorm.__class__ = _qwen2.Qwen2RMSNorm
base_model.model.norm.__class__ = _qwen2.Qwen2RMSNorm
base_model.config._attn_implementation = 'sdpa'
print(f'[P2] Class-swapped {_n_layers} layers → clean HF + sdpa')

# 2c. 加载 LoRA adapter 并合并进基础权重
#     merge_and_unload() 消除 PeftModel 包装的 252 层 per-token
#     Python 调度开销（之前 9.5 tok/s 慢速的主因之一）
_peft = PeftModel.from_pretrained(base_model, _LORA_DIR)
model = _peft.merge_and_unload()
del _peft, base_model
gc.collect()
print(f'[P2] LoRA merged into base weights — plain {type(model).__name__}')

# 2d. 清除所有残留的 Unsloth forward hooks
_hooks_cleared = 0
for _m in model.modules():
    _hooks_cleared += len(_m._forward_hooks) + len(_m._forward_pre_hooks)
    _m._forward_hooks.clear()
    _m._forward_pre_hooks.clear()
print(f'[P2] Cleared {_hooks_cleared} residual hooks')

model.eval()

# ════════════════════════════════════════════════════════════
# Phase 3: Tokenizer + Generation Config (解决 P3)
# ════════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained(_LORA_DIR)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3a. 构建 EOS token IDs 列表
_eos_ids = []
if hasattr(model.config, 'eos_token_id'):
    v = model.config.eos_token_id
    _eos_ids = list(v) if isinstance(v, list) else [v]
_im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
if _im_end_id and _im_end_id not in _eos_ids:
    _eos_ids.append(_im_end_id)
EOS_TOKEN_IDS = _eos_ids

# 3b. 重置 generation_config，解决 max_length=32768 冲突
#     之前 model.generation_config.max_length = None 不生效，
#     因为 Qwen2.5 的 generation_config.json 强制设了 max_length=32768。
#     正确做法：用全新的 GenerationConfig 替换，只保留必要参数。
model.generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=EOS_TOKEN_IDS,
)
print(f'[P3] generation_config reset (no max_length conflict)')
print(f'[P3] EOS token IDs: {EOS_TOKEN_IDS}')

# ════════════════════════════════════════════════════════════
# Phase 4: 断言验证 — 在花 GPU 时间之前确认所有前置条件
# ════════════════════════════════════════════════════════════

_impl = getattr(model.config, '_attn_implementation', 'unknown')
_dtype = next(model.parameters()).dtype
_on_gpu = all(p.device.type == 'cuda' for p in model.parameters())
_cls = type(model).__name__
_alloc = torch.cuda.memory_allocated() / 1e9

print(f'\n{"=" * 55}')
print(f'  INFERENCE MODEL DIAGNOSTICS')
print(f'{"=" * 55}')
print(f'  Model class:       {_cls}')
print(f'  attn_impl:         {_impl}')
print(f'  dtype:             {_dtype}')
print(f'  All on GPU:        {_on_gpu}')
print(f'  GPU mem allocated:  {_alloc:.2f} GB')
print(f'  Is PeftModel:      {hasattr(model, "peft_config")}')
print(f'  EOS IDs:           {EOS_TOKEN_IDS}')
print(f'  gen_config.max_length: {getattr(model.generation_config, "max_length", "NOT SET")}')
print(f'{"=" * 55}')

assert _cls == 'Qwen2ForCausalLM', f'Expected Qwen2ForCausalLM, got {_cls}'
assert _impl == 'sdpa', f'Expected sdpa, got {_impl}'
assert _dtype == torch.float16, f'Expected fp16, got {_dtype}'
assert _on_gpu, 'Some params on CPU!'
assert not hasattr(model, 'peft_config'), 'LoRA not merged!'

# 4b. Forward pass 速度基准
_test_ids = tokenizer('hello', return_tensors='pt').input_ids.to(model.device)
torch.cuda.synchronize()
_t0 = time.time()
with torch.no_grad():
    for _ in range(20):
        model(_test_ids)
torch.cuda.synchronize()
_fwd_ms = (time.time() - _t0) / 20 * 1000
print(f'\n  Forward latency:   {_fwd_ms:.1f} ms/step')
if _fwd_ms > 50:
    print(f'  *** WARNING: {_fwd_ms:.0f}ms too high. Expected <20ms A100, <50ms T4.')
else:
    print(f'  Forward speed OK.')
print()

# ════════════════════════════════════════════════════════════
# Phase 5: Inference helpers
# ════════════════════════════════════════════════════════════

SVG_EXTRACT_RE = re.compile(r'<svg.*?</svg>', flags=re.IGNORECASE | re.DOTALL)

GENERATE_KWARGS = dict(
    max_new_tokens=CONFIG['max_new_tokens'],
    do_sample=True,
    temperature=CONFIG['temperature'],
    top_p=CONFIG['top_p'],
    repetition_penalty=CONFIG['repetition_penalty'],
    eos_token_id=EOS_TOKEN_IDS,
    pad_token_id=tokenizer.pad_token_id,
)


def extract_svg(text):
    m = SVG_EXTRACT_RE.search(text)
    return m.group(0).strip() if m else ''


def _postprocess(decoded, prompt, debug=False):
    """返回 (svg, status)。debug=True 打印每一步的中间结果。"""
    raw = extract_svg(decoded)
    if not raw:
        if debug:
            print(f'    [FAIL] extract_svg returned empty')
            print(f'    [FAIL] decoded text (first 500 chars):')
            print(f'           {decoded[:500]}')
        return fallback_svg(prompt), 'extract_fail'

    cleaned = sanitize_svg(raw)
    if not cleaned:
        if debug:
            print(f'    [FAIL] sanitize_svg returned empty')
            print(f'    [FAIL] raw SVG (first 300 chars): {raw[:300]}')
        return fallback_svg(prompt), 'sanitize_fail'

    ok, reason = validate_svg(cleaned)
    if not ok:
        if debug:
            print(f'    [FAIL] validate_svg: {reason}')
            print(f'    [FAIL] cleaned SVG (first 300 chars): {cleaned[:300]}')
        return fallback_svg(prompt), f'validate_fail:{reason}'

    if debug:
        print(f'    [OK] valid SVG, {len(cleaned)} chars')
    return cleaned, 'ok'


def generate_svg(prompt, debug=False):
    """单条推理。debug=True 打印生成细节。"""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[-1]

    with torch.no_grad():
        out = model.generate(**inputs, **GENERATE_KWARGS)

    gen_ids = out[0][input_len:]
    n_gen = len(gen_ids)
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    if debug:
        hit_eos = n_gen < CONFIG['max_new_tokens']
        print(f'  [gen] {n_gen} tokens ({"stopped at EOS" if hit_eos else "HIT MAX"})')
        print(f'  [gen] decoded len={len(decoded)}, first 300 chars:')
        print(f'        {decoded[:300]}')

    return _postprocess(decoded, prompt, debug=debug)


def generate_svg_batch(prompts):
    """批量推理 (left-padding)。"""
    bs = CONFIG.get('inference_batch_size', 4)
    all_results = []

    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        chats = []
        for p in batch:
            msgs = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': p},
            ]
            chats.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            ))

        inputs = tokenizer(
            chats, return_tensors='pt', padding=True,
            truncation=True, max_length=CONFIG['max_seq_length'],
        ).to(model.device)
        input_len = inputs['input_ids'].shape[-1]

        with torch.no_grad():
            out = model.generate(**inputs, **GENERATE_KWARGS)

        for j, prompt in enumerate(batch):
            gen_ids = out[j][input_len:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_results.append(_postprocess(decoded, prompt))

    return all_results


print('Inference setup complete. Ready for smoke test.\n')
