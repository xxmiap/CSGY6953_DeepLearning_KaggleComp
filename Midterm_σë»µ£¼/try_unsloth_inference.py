# ═══════════════════════════════════════════════════════════════
# Cell 17 替代方案：使用 Unsloth 原生推理
# 直接粘贴到 Colab 中替换当前 Cell 17
#
# 支持三种情况：
#   A) 训练刚完成，model 还在内存中 → 直接用
#   B) 运行时重启过，model 不存在 → 从路径重新加载（Colab / Kaggle）
# ═══════════════════════════════════════════════════════════════

from unsloth import FastLanguageModel
from peft import PeftModel
import gc

# ── Kaggle 路径（需要时取消注释，并注释掉下方 Colab 路径） ──
# _LORA_DIR = '/kaggle/working/qwen25_3b_svg_lora'
# # 如果 LoRA 是作为 Kaggle Dataset 上传的，改为：
# # _LORA_DIR = '/kaggle/input/your-lora-dataset-name/qwen25_3b_svg_lora'

# ── Colab 路径（当前生效） ──
_LORA_DIR = CONFIG['output_dir']

# ── 判断是否需要重新加载 ──
_need_reload = False
try:
    model  # 检查变量是否存在
    print('[info] model found in memory, reusing')
except NameError:
    _need_reload = True
    print('[info] model not in memory, reloading from disk')

if _need_reload:
    gc.collect()
    torch.cuda.empty_cache()

    # 加载基础模型（Unsloth 方式）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG['model_name'],
        max_seq_length=CONFIG['max_seq_length'],
        dtype=None,
        load_in_4bit=True,
    )

    # 加载已保存的 LoRA 适配器
    model = PeftModel.from_pretrained(model, _LORA_DIR)
    print(f'[info] Loaded base model + LoRA from {_LORA_DIR}')

# ── 切换到 Unsloth 推理模式 ──
FastLanguageModel.for_inference(model)
print('[ok] FastLanguageModel.for_inference() applied')

tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── EOS tokens ──
_eos_ids = []
if hasattr(model.config, 'eos_token_id'):
    v = model.config.eos_token_id
    _eos_ids = v if isinstance(v, list) else [v]
_im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
if _im_end_id and _im_end_id not in _eos_ids:
    _eos_ids.append(_im_end_id)
EOS_TOKEN_IDS = _eos_ids

model.generation_config.max_length = None

print(f'EOS token IDs: {EOS_TOKEN_IDS}')
alloc = torch.cuda.memory_allocated() / 1e9
print(f'GPU mem: {alloc:.2f} GB')

# ── Inference helpers ──

SVG_EXTRACT_RE = re.compile(r'<svg.*?</svg>', flags=re.IGNORECASE | re.DOTALL)

GENERATE_KWARGS = dict(
    max_new_tokens=CONFIG['max_new_tokens'],
    do_sample=True,
    temperature=CONFIG['temperature'],
    top_p=CONFIG['top_p'],
    repetition_penalty=CONFIG['repetition_penalty'],
    eos_token_id=EOS_TOKEN_IDS,
)


def extract_svg(text):
    m = SVG_EXTRACT_RE.search(text)
    return m.group(0).strip() if m else ''


def _postprocess(decoded, prompt):
    raw = extract_svg(decoded)
    if not raw:
        return fallback_svg(prompt), 'extract_fail'
    cleaned = sanitize_svg(raw)
    if not cleaned:
        return fallback_svg(prompt), 'sanitize_fail'
    ok, reason = validate_svg(cleaned)
    if not ok:
        return fallback_svg(prompt), f'validate_fail:{reason}'
    return cleaned, 'ok'


def generate_svg(prompt):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, **GENERATE_KWARGS)
    gen_ids = out[0][inputs['input_ids'].shape[-1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f'  [debug] generated {len(gen_ids)} tokens, decoded len={len(decoded)}')
    print(f'  [debug] first 300 chars: {decoded[:300]}')
    return _postprocess(decoded, prompt)


def generate_svg_batch(prompts):
    """Unsloth 推理模式下逐条生成（Unsloth 不支持 batch padding generate）"""
    all_results = []
    for p in prompts:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': p},
        ]
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **GENERATE_KWARGS)
        gen_ids = out[0][inputs['input_ids'].shape[-1]:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        all_results.append(_postprocess(decoded, p))
    return all_results
