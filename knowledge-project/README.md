# 图灵中心知识图谱项目

本项目面向“以 Alan Turing 为中心的知识图谱”构建任务，目标是从 Wikipedia 风格文本中自动完成数据爬取、实体抽取、属性抽取、关系抽取、知识融合以及 Neo4j 导入，形成一个可查询、可扩展、可复现的小型人物主题知识图谱。

当前版本不是端到端的大模型生成式图谱系统，而是一套“规则 + 统计模型 + NLI 验证”的工程化流水线。它强调：

- 围绕 `Alan Turing` 的中心化图谱构建
- 先保证精度，再逐步扩展召回
- 每一步都有中间产物，便于诊断和调试
- 最终结果可直接导入 Neo4j

## 一、项目总览

完整流程如下：

1. 爬取图灵主题语料
2. 构造弱监督 NER 训练数据
3. 训练实体识别模型
4. 抽取实体属性
5. 抽取实体关系
6. 融合属性与关系，生成最终图谱节点和边
7. 导入 Neo4j

## 二、环境依赖

### 1. 推荐环境

- Python `3.10` 及以上
- Windows / Linux / macOS 均可
- 支持 CPU 运行
- 如有 CUDA，可自动使用 GPU 加速

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 主要依赖说明

- `wikipedia`：用于抓取 Wikipedia 页面内容、链接和分类
- `torch`：用于 NER 模型训练与推理
- `transformers`：用于 token classification 与 NLI 模型
- `tqdm`：用于进度条展示
- `neo4j`：用于将最终图谱导入 Neo4j
- `scikit-learn`：用于训练集/验证集划分与评估辅助

## 三、项目结构

```text
knowledge-project/
|-- README.md
|-- requirements.txt
|-- docs/
|   `-- schema.md
|-- scripts/
|   |-- scraper.py
|   |-- train_ner_bilstm_crf.py
|   |-- extract_attributes.py
|   |-- extract_relations_bert.py
|   |-- fuse_knowledge.py
|   `-- import_to_neo4j.py
`-- data/
    |-- raw/
    |   |-- turing_core_corpus.jsonl
    |   |-- turing_support_corpus.jsonl
    |   |-- turing_noise_corpus.jsonl
    |   |-- turing_schema_corpus.jsonl
    |   `-- turing_schema_sources.csv
    |-- intermediate/
    |   `-- ner_char_bio.jsonl
    `-- output/
        |-- attributes.jsonl
        |-- attributes_dev.jsonl
        |-- relations_dev.jsonl
        |-- relations_dev_debug.jsonl
        |-- kg_dev_nodes.jsonl
        |-- kg_dev_edges.jsonl
        |-- ner_diagnosis.json
        `-- ner_bilstm_crf/
            |-- config.json
            |-- model.safetensors
            |-- tokenizer.json
            |-- tokenizer_config.json
            |-- tag_vocab.json
            |-- char_vocab.json
            |-- gazetteer.json
            `-- ner_config.json
```

## 四、数据构成

### 1. 当前语料规模

当前仓库内已经存在的主要数据文件规模如下：

- `data/raw/turing_core_corpus.jsonl`
- `data/raw/turing_support_corpus.jsonl`
- `data/raw/turing_noise_corpus.jsonl`
- `data/raw/turing_schema_corpus.jsonl`
- `data/raw/turing_schema_sources.csv`

### 2. 当前中间结果与输出规模

- `data/intermediate/ner_char_bio.jsonl`：弱标注 NER 样本
- `data/output/attributes.jsonl`：属性抽取结果
- `data/output/relations_dev.jsonl`：中心化关系抽取结果
- `data/output/kg_dev_nodes.jsonl`：融合后的节点
- `data/output/kg_dev_edges.jsonl`：融合后的边

### 3. 各类数据文件含义

#### `data/raw/`

- `turing_core_corpus.jsonl`
  - 与 Alan Turing 或一跳核心实体强相关的页面语料
- `turing_support_corpus.jsonl`
  - 作为背景支撑的页面语料
- `turing_noise_corpus.jsonl`
  - 有意保留的一部分噪声语料，用于测试系统鲁棒性
- `turing_schema_corpus.jsonl`
  - 合并后的下游抽取总语料，默认作为实体、属性、关系抽取输入
- `turing_schema_sources.csv`
  - 页面级来源清单和爬取记录

#### `data/intermediate/`

- `ner_char_bio.jsonl`
  - 弱监督生成的 NER 训练数据
  - 每条记录包含文本和字符级 BIO 标签

#### `data/output/`

- `attributes.jsonl`
  - 属性抽取结果
- `relations_dev.jsonl`
  - 关系抽取结果
- `kg_dev_nodes.jsonl`
  - 融合后的最终节点
- `kg_dev_edges.jsonl`
  - 融合后的最终边
- `ner_bilstm_crf/`
  - 训练好的 NER 模型及其配置文件

## 五、各模块说明

## 1. 数据爬取模块

**脚本**

[`scripts/scraper.py`]

### 方法原理

本模块使用 `wikipedia` 库抓取图灵主题页面，但不是无约束地抓取所有链接页面，而是采用“主题裁剪 + 页面评分”的方式控制语料质量。

具体流程如下：

1. 从预定义的 schema 种子实体出发
2. 获取页面正文、链接、分类信息
3. 根据以下信号对页面进行评分：
   - 是否命中核心种子标题
   - 是否在正文中频繁出现图灵相关关键词
   - 是否出现与知识图谱 schema 相关的关系提示短语
   - 页面分类是否命中计算机、密码学、历史等主题提示
4. 将页面划分为：
   - `core`
   - `support`
   - `noise`
5. 输出原始语料和来源清单

本质上，这是一个“围绕图灵主题进行约束式爬取”的模块，而不是通用开放域爬虫。

### 操作指南

直接执行即可。默认会输出多份分层语料，并自动生成合并语料。

### 启动命令

```bash
python scripts/scraper.py
```

### 输出文件

- `data/raw/turing_core_corpus.jsonl`
- `data/raw/turing_support_corpus.jsonl`
- `data/raw/turing_noise_corpus.jsonl`
- `data/raw/turing_schema_corpus.jsonl`
- `data/raw/turing_schema_sources.csv`

## 2. 实体抽取模块

**脚本**

[`scripts/train_ner.py`]

### 方法原理

本模块采用弱监督标注 + Transformer 序列标注 NER + 规则回退融合的混合式实体抽取方案，
在无大量人工标注数据的前提下，实现高精度、高召回的多类型实体识别。

它包含三个阶段：

#### 阶段 1：弱监督标注数据构造

通过以下信息自动生成字符级 BIO 标签：

- gazetteer 词表
- 手工规则模式
- 图灵主题别名和核心实体别名

这样可以在没有大量人工标注的情况下快速构造 NER 训练集。

#### 阶段 2：NER 模型训练

训练时使用 `AutoModelForTokenClassification`，将字符级弱标签转换为 token 级标签后进行监督训练。

当前训练流程支持：

- 自动划分 train/dev
- 每个 epoch 后输出 Dev `P / R / F1`
- 同时输出实体级 `Gold / Pred / TP`

这里的 Dev 指标已经改为实体级 span 评估，而不是简单 token 级统计。

#### 阶段 3：推理与规则融合

预测时会将：

- 模型识别结果
- 规则实体
- gazetteer 命中结果

进行合并，提升低频实体类型的覆盖率。

### 当前支持的实体类型

- `Person`
- `Organization`
- `Location`
- `Concept`
- `Artifact`
- `Event`
- `Publication`
- `Honor`

### 操作指南

#### 第一步：生成弱监督训练数据

```bash
python scripts/train_ner.py prepare --input data/raw/turing_schema_corpus.jsonl --output data/intermediate/ner_char_bio.jsonl
```

#### 第二步：训练 NER 模型

```bash
python scripts/train_ner.py train --data data/intermediate/ner_char_bio.jsonl --output-dir data/output/ner_bilstm_crf --epochs 8 --batch-size 8 --lr 1e-4
```

#### 第三步：测试单句实体识别

```bash
python scripts/train_ner.py predict --model-dir data/output/ner_bilstm_crf --text "Alan Turing worked at Bletchley Park."
```

### 输出文件

- `data/intermediate/ner_char_bio.jsonl`
- `data/output/ner/`

## 3. 属性抽取模块

**脚本**

[`scripts/extract_attributes.py`]

### 方法原理

属性抽取采用“类型约束候选生成 + NLI 验证”的两阶段方案。

#### 第一步：候选属性生成

基于实体类型和句面规则，生成属性候选。例如：

- `Person`
  - `birth_date`
  - `death_date`
  - `nationality`
  - `occupation`
  - `aliases`
- `Organization`
  - `established_year`
  - `org_type`
  - `location`
- `Publication`
  - `publish_year`
  - `venue`

候选生成依赖：

- 正则规则
- 日期模式
- 国家/职业词表
- 页面主实体推断
- 图灵/Joan Clarke 等核心人物别名归一

#### 第二步：NLI 验证

对每个属性候选构造自然语言假设句，然后使用 cross-encoder NLI 模型做蕴含判断，只有通过阈值的候选才会保留。

这种设计的优点是：

- 比直接生成属性更稳
- 更易排查错误
- 每类属性都可以独立调阈值

### 操作指南

先保证已经训练好 NER 模型，再运行属性抽取。

### 启动命令

```bash
python scripts/extract_attributes.py --input data/raw/turing_schema_corpus.jsonl --ner-dir data/output/ner --output data/output/attributes.jsonl --threshold 0.7
```

### 输出文件

- `data/output/attributes.jsonl`

## 4. 关系抽取模块

**脚本**

[`scripts/extract_relations_bert.py`]

### 方法原理

关系抽取采用“schema 约束 + 规则优先 + NLI 辅助”的中心化关系抽取方案。

每种关系都定义了：

- 主体类型
- 客体类型
- 触发词模式
- 候选配对策略
- 抽取模式

系统不是见到任意两个实体就尝试配对，而是必须满足 schema 约束后才会继续抽取。

#### 抽取逻辑

1. 先基于句内 trigger 生成关系候选
2. 对高精度规则直接保留
3. 对较弱规则使用 NLI 模型辅助判断
4. 对输出结果再做“图灵中心化过滤”

#### 当前实现中的重要机制

- 页面主语代词归一
  - 例如 `He` 归一到 `Alan Turing`
- 中心化过滤
  - 优先保留与图灵直接相连的边
  - 对扩展边使用更高阈值
- 离线安全加载 NLI 模型
  - 默认优先本地缓存
  - 如果模型不可用，可退化为 rule-only 模式

### 当前主要关系类型

- `BORN_IN`
- `DIED_IN`
- `EDUCATED_AT`
- `WORKED_AT`
- `COLLEAGUE_OF`
- `PROPOSED`
- `WORKED_ON`
- `AUTHORED`
- `PARTICIPATED_IN`
- `AFFECTED_BY`
- `HONORED_BY`
- `NAMED_AFTER`

### 操作指南

#### 默认离线模式

```bash
python scripts/extract_relations_bert.py --input data/raw/turing_schema_corpus.jsonl --ner-dir data/output/ner_bilstm_crf --output data/output/relations_dev.jsonl
```

#### 若本地没有缓存模型，允许联网下载

```bash
python scripts/extract_relations_bert.py --input data/raw/turing_schema_corpus.jsonl --ner-dir data/output/ner_bilstm_crf --output data/output/relations_dev.jsonl --allow-remote-model
```

### 输出文件

- `data/output/relations_dev.jsonl`

## 5. 知识融合模块

**脚本**

[`scripts/fuse_knowledge.py`](E:\Coding\knowledge-project\knowledge-project\scripts\fuse_knowledge.py)

### 方法原理

知识融合模块负责把属性抽取结果和关系抽取结果变成真正的图谱节点和边。

它主要完成以下工作：

- 实体文本清洗与标准化
- 图灵中心实体归一
- 关系类型合法性检查
- 多来源关系聚合
- 属性冲突裁决
- 图中心距离计算
- 节点/边层级划分
  - `core`
  - `expanded`

这一阶段决定了最终进入图谱的数据质量，因此它不是简单去重，而是一个轻量级的实体对齐与事实裁决过程。

### 操作指南

输入为：

- 关系抽取结果
- 属性抽取结果

输出为：

- 图谱节点文件
- 图谱边文件

### 启动命令

```bash
python scripts/fuse_knowledge.py --rel-file data/output/relations_dev.jsonl --attr-file data/output/attributes.jsonl --output-rel data/output/kg_dev_edges.jsonl --output-ent data/output/kg_dev_nodes.jsonl
```

### 输出文件

- `data/output/kg_dev_nodes.jsonl`
- `data/output/kg_dev_edges.jsonl`

## 6. 图谱导入模块

**脚本**

[`scripts/import_to_neo4j.py`]

### 方法原理

本模块用于将融合后的 JSONL 节点和边导入 Neo4j。

主要处理逻辑：

- 为 `:Entity(id)` 创建唯一约束
- 每个节点至少带 `:Entity` 标签
- 再根据实体类型追加 `:Person`、`:Organization` 等标签
- 对嵌套属性、列表属性做 Neo4j 兼容化处理
- 将边写入为 Neo4j 关系类型

### 操作指南

导入前请先启动 Neo4j 服务，并确认连接参数正确。

### 启动命令

```bash
python scripts/import_to_neo4j.py --uri bolt://localhost:7687 --user neo4j --password 12345678 --nodes data/output/kg_dev_nodes.jsonl --edges data/output/kg_dev_edges.jsonl
```

### 输出结果

- Neo4j 图数据库中的节点和边

## 六、当前局限

- 语料规模已经扩展到 371 篇，但由于当前图谱是“图灵中心化过滤”模式，最终图规模仍偏小
- `Honor` 和 `Publication` 的识别质量仍弱于 `Person`、`Location`
- 部分关系类型仍然较依赖句面 trigger
- 弱监督标注会给 NER 训练带来噪声
- 属性抽取目前仍是候选驱动，不是穷尽式抽取

## 九、未来改进方向

### 1. 提升 NER 监督质量

- 补人工标注 dev/test 集
- 扩充 `Honor`、`Publication`、`Event` 样本
- 对弱标注噪声做更细的置信度建模

### 2. 提升关系抽取召回率

- 在保持精度的前提下扩充 schema
- 引入更细粒度的句法配对策略
- 支持跨句关系抽取

### 3. 提升属性抽取覆盖率

- 丰富人物、机构、出版物属性类型
- 从“句级抽取”扩展到“段级/页面首段抽取”

### 4. 提升融合与归一能力

- 更强的别名归一
- 引入 Wikidata 辅助消歧
- 做跨模块置信度校准

### 5. 提升评估体系

- 构建 NER 金标准评测集
- 构建属性/关系 dev 集
- 增加端到端图谱质量指标

### 6. 提升工程化程度

- 将硬编码参数迁移到配置文件
- 增加统一命令行配置入口
- 支持更多图数据库或图文件导出格式
