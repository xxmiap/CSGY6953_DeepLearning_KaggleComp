# 一、文档目标与范围

## 1.1 任务背景

输入：

* `prompt`（自然语言）
  输出：
* `svg`（结构化文本/XML）

这是一个典型：

> **text → structured text generation（弱结构约束）问题**

---

## 1.2 EDA 的核心目标（必须明确）

你的 EDA 不是为了“了解数据”，而是为了回答以下**关键决策问题**：

### Q1. 数据是否干净？

→ 是否需要清洗（非法SVG、异常样本）

### Q2. 数据是否一致？

→ SVG表达是否标准化（viewBox / width 等）

### Q3. 数据是否可学？

→ 长度、复杂度是否超出模型能力

### Q4. 数据是否有噪声？

→ prompt 与 svg 是否匹配

### Q5. 数据分布是否均衡？

→ 简单 vs 复杂 SVG 的比例

---

# 二、EDA脚本总体结构

建议你的EDA脚本分为5个模块：

```text
EDA/
├── 1_basic_stats.py
├── 2_svg_structure.py
├── 3_length_analysis.py
├── 4_quality_checks.py
├── 5_prompt_svg_alignment.py
```

每个模块必须输出：

* 数值统计（DataFrame）
* 可视化（hist / boxplot）
* 结论（log or markdown）

---

# 三、模块一：基础统计（Basic Stats）

## 3.1 必做内容

### (1) 数据规模

```python
n_samples
n_unique_prompt
n_unique_svg
```

---

### (2) 重复情况

```python
duplicate_rows
duplicate_svg
duplicate_prompt
```

---

### (3) prompt–svg 映射关系

```python
# 一个 svg 对应多少 prompt
svg_prompt_count = df.groupby('svg')['prompt'].nunique()
```

---

## 3.2 输出内容

* 表格：重复率
* 分布图：一个svg对应多少prompt

---

## 3.3 你要回答的问题

* 是否存在严重重复？
* 是否存在 one-to-many mapping？

---

# 四、模块二：SVG结构分析（核心模块）

这是整个EDA最关键部分。

---

## 4.1 标签分布

统计：

```python
path_count
circle_count
rect_count
line_count
```

---

## 4.2 每个SVG的复杂度

定义：

```python
complexity = number_of_elements
```

---

## 4.3 属性使用情况

统计：

* 是否使用 `viewBox`
* 是否使用 `width/height`
* fill / stroke 使用比例

---

## 4.4 输出

* 柱状图：标签分布
* 直方图：复杂度分布

---

## 4.5 关键问题

* SVG 是否“统一风格”？
* 是否存在极复杂样本？

---

# 五、模块三：长度与token分析（直接影响LLM）

---

## 5.1 字符长度

```python
prompt_len = len(prompt)
svg_len = len(svg)
```

---

## 5.2 token长度（推荐）

使用 tokenizer：

```python
tokenizer(prompt)
tokenizer(svg)
```

---

## 5.3 分布分析

输出：

* prompt长度分布
* svg长度分布
* joint plot（prompt vs svg）

---

## 5.4 关键问题

* 是否存在极长SVG（> context limit）
* 是否需要截断或过滤

---

# 六、模块四：数据质量检查（必须做）

---

## 6.1 SVG合法性

```python
XML parse success rate
```

---

## 6.2 数值异常

检查：

* NaN
* 空字符串
* 极短SVG

---

## 6.3 数值精度

分析：

```python
float precision distribution
```

---

## 6.4 输出

* 非法SVG比例
* 异常样本数量

---

## 6.5 关键问题

* 是否必须清洗SVG？
* 是否需要round float？

---

# 七、模块五：prompt–svg一致性（进阶）

---

## 7.1 方法（简单版）

使用 embedding：

```python
similarity(prompt, svg_text)
```

---

## 7.2 异常检测

* similarity 过低 → 可疑样本

---

## 7.3 输出

* similarity分布
* low-score样本列表

---

## 7.4 注意

这是：

> **分析性推论（不是严格真值）**

---

# 八、输出规范（必须遵守）

你的EDA脚本最终应该输出：

---

## 8.1 一份 summary report（必须）

结构如下：

```text
1. 数据规模与重复情况
2. SVG结构分析
3. 长度与token分析
4. 数据质量问题
5. 建议的数据清洗策略
```

---

## 8.2 可视化

至少包含：

* 长度分布图
* SVG复杂度分布
* 标签分布
* similarity分布（可选）

---

## 8.3 清洗建议（最关键）

EDA最终必须输出：

```text
建议：
- 删除非法SVG（x%）
- 标准化viewBox（y%存在width/height）
- 截断长SVG（top 1%）
- 保留多prompt–单svg
```

---

# 九、你需要避免的误区（重要）

---

## ❌ 误区1：EDA变成“画图展示”

→ 必须服务决策

---

## ❌ 误区2：过早清洗

→ 先观察，再决定

---

## ❌ 误区3：忽略SVG结构

→ 这是本任务的核心

---

## ❌ 误区4：只看平均值

→ 一定要看 tail（极端值）

---

# 十、建议执行顺序（工程实践）

---

## Step 1（10分钟）

* basic stats
* 长度分布

---

## Step 2（20分钟）

* SVG结构解析
* 标签统计

---

## Step 3（20分钟）

* 合法性检查
* 异常检测

---

## Step 4（可选）

* embedding一致性分析

---

# 十一、进一步推进（研究级）

如果你想提升到论文级别，可以扩展：

* SVG → AST → complexity metric
* prompt语义聚类
* curriculum learning设计
* tokenizer设计（结构感知）

---

# 最后的评价（客观）

你当前阶段如果直接做模型，是不稳的。

从工程角度：

> **EDA质量 ≈ 项目最终效果的上限**