# 训练质量优化策略

基于当前 baseline 的分析（smoke test 3/3 valid 但语义质量一般），按预期收益排序。

---

## 1. 模型选择：换用 Qwen2.5-Coder-3B-Instruct

当前使用 `Qwen2.5-3B-Instruct`（通用对话模型）。SVG 本质上是结构化 XML 代码，推荐切换为同参数量的代码专用模型：

| 模型 | 参数量 | 优势 | Unsloth ID |
|------|--------|------|------------|
| **Qwen2.5-Coder-3B-Instruct** | 3.09B | 代码生成专精，对 XML/SVG 结构理解更强 | `unsloth/Qwen2.5-Coder-3B-Instruct` |
| Qwen2.5-3B-Instruct | 3.09B | 当前 baseline，通用能力强 | `unsloth/Qwen2.5-3B-Instruct` |
| Phi-3.5-mini-instruct | 3.82B | 推理能力强，接近 4B 上限 | `unsloth/Phi-3.5-mini-instruct` |

**首选 Qwen2.5-Coder-3B-Instruct**，理由：
- SVG 是代码，代码模型对标签闭合、属性语法、嵌套结构有更好的先验
- 同系列模型，LoRA、ChatML 模板、tokenizer 全部兼容，切换成本为零
- Phi-3.5-mini 虽大 0.7B，但使用不同的 chat 模板，需要改 prompt 格式

**修改方式：** 只改 CONFIG 一行：
```python
'model_name': 'unsloth/Qwen2.5-Coder-3B-Instruct',
```

---

## 2. 数据量：大幅增加训练样本

当前 `train.csv` 有 50,000 条数据，数据分布如下：

| SVG 字符上限 | 可用条数 | 占比 |
|-------------|---------|------|
| ≤ 2500（当前） | 29,082 | 58.2% |
| ≤ 3000 | 33,568 | 67.1% |
| ≤ 4000 | 40,315 | 80.6% |
| ≤ 5000 | 44,683 | 89.4% |

当前设置 `max_svg_train_chars=2500` + `max_train_samples_per_source=12000`，实际只用了 50K 中的 12K（24%）。

**建议调整：**

```python
'max_svg_train_chars': 4000,          # 2500 → 4000，可用数据从 29K → 40K
'max_train_samples_per_source': 30000, # 12000 → 30000
```

这样训练数据从 12K 增加到 ~30K，覆盖更多形状类型和 SVG 模式。

> 注意：`max_svg_train_chars=4000` 对应 ~1200-1600 tokens，加上 prompt 和 ChatML 模板 ~100 tokens，总计 ~1300-1700 tokens，仍在 `max_seq_length=2048` 之内。

---

## 3. 训练轮次：1 epoch → 2-3 epochs

当前 1 epoch / 12K samples = 375 steps（effective batch size = 32）。对于学习一个全新的输出模态（SVG），这远远不够。

**建议：**

```python
'num_train_epochs': 3,
'gradient_accumulation_steps': 16,  # 32 → 16，更新频率翻倍
```

关键指标：
- 数据量 30K × 3 epochs = 90K samples 被看到
- Steps = 90K / 16 = 5,625 steps
- 每 step 使用 batch_size=1 × grad_accum=16 = 16 的有效 batch
- A100 上预计训练时间：~2-3 小时

配合调整学习率和 warmup：
```python
'learning_rate': 1e-4,    # 2e-4 → 1e-4，多 epoch 时降低避免过拟合
'warmup_ratio': 0.03,     # 0.05 → 0.03，因总步数增加
```

---

## 4. LoRA rank：r=16 → r=32

当前 `r=16` 对于学习通用对话足够，但 SVG 生成需要模型学习全新的输出语法结构（XML 标签、坐标数值、路径命令等），需要更高的适配容量。

**建议：**

```python
'lora_r': 32,
'lora_alpha': 32,   # 保持 alpha/r = 1
```

可训练参数从 ~27M 增加到 ~54M（3B 模型的 ~1.8%），训练速度几乎不变，但表达能力显著增强。

如果 A100 显存允许，也可以尝试 `r=64`。

---

## 5. System Prompt 优化

当前 system prompt 较为笼统。可以更明确地引导模型使用基本图元：

```python
SYSTEM_PROMPT = (
    'You are an SVG code generator. Given a text description, output a single '
    '<svg> element. Rules:\n'
    '- Use xmlns="http://www.w3.org/2000/svg", width="256", height="256", '
    'viewBox="0 0 256 256"\n'
    '- Prefer basic shapes (circle, rect, ellipse, polygon) over complex paths\n'
    '- Use meaningful colors that match the description\n'
    '- Output ONLY the SVG code, no explanation'
)
```

关键改进：明确鼓励使用 `<circle>`, `<rect>` 等基本图元而非万物皆 `<path>`。当前模型倾向于用 Bezier 曲线画圆（579 tokens），如果改为 `<circle>` 只需 ~50 tokens。

---

## 6. 推理参数优化

当前推理偏保守，可以调整以提高输出多样性和质量：

```python
'temperature': 0.7,          # 0.3 → 0.7，当前过低导致输出刻板
'top_p': 0.9,                # 0.8 → 0.9
'repetition_penalty': 1.0,   # 1.05 → 1.0，对 SVG 坐标的惩罚会破坏数值模式
'max_new_tokens': 512,       # 768 → 512，合格 SVG 不需要这么长
```

或者采用**多次采样取最优**策略（如果时间允许）：

```python
# 每个 prompt 生成 N 次，取第一个 valid 的
def generate_svg_best_of_n(prompt, n=3):
    for _ in range(n):
        svg, status = generate_svg(prompt)
        if status == 'ok':
            return svg, 'ok'
    return svg, status  # 最后一次的结果
```

---

## 7. 训练数据增强

### 7a. Prompt 标准化改进

当前 `_clean_prompt` 只做简单正则清理。可以进一步标准化：

- 统一大小写：`"A Red Circle"` → `"a red circle"`
- 去除冗余修饰：`"a very simple basic red circle shape"` → `"a red circle"`

### 7b. SVG 训练目标优化

在 `_normalize_comp` 中对 SVG 进行更积极的简化：

- 移除 `<title>`, `<desc>`, `<metadata>` 等不影响渲染的标签
- 规范化颜色格式：`rgb(255,0,0)` → `#FF0000`
- 四舍五入浮点坐标：`128.39746` → `128.4`（减少 token 数）
- 移除空白和换行（压缩 token 使用）

这些处理能让同样的视觉内容用更少的 tokens 表达，使模型在 `max_seq_length` 内学到更复杂的 SVG。

---

## 8. 推理加速（已在代码中修改）

这些在 notebook 中已实现或将实现：

- `attn_implementation="sdpa"` — Flash Attention 加速
- `eos_token_id` 包含 `<|im_end|>` — 提前停止
- `inference_batch_size=8` — 增大 batch
- `model.generation_config.max_length = None` — 消除 warning

---

## 优先级总结

| 优先级 | 策略 | 预期改进 | 实现难度 |
|--------|------|---------|---------|
| **P0** | 换 Qwen2.5-Coder-3B | 语义理解 + SVG 结构 | 改 1 行 |
| **P0** | 增加数据量 12K → 30K | 覆盖更多模式 | 改 2 行 |
| **P1** | epochs 1 → 3 | 充分学习 | 改 2 行 |
| **P1** | LoRA r=16 → 32 | 适配容量 | 改 2 行 |
| **P1** | 优化 System Prompt | 引导基本图元 | 改 1 处 |
| **P2** | 推理参数调优 | 输出多样性 | 改 4 行 |
| **P2** | SVG 训练目标简化 | 降低 token 浪费 | 新增 1 个函数 |
| **P3** | Prompt 标准化 | 减少噪声 | 新增 1 个函数 |
| **P3** | 多次采样策略 | valid rate 提升 | 新增 1 个函数 |

建议先一次性应用 P0 + P1（改动量极小），跑一轮看效果后再决定 P2/P3。
