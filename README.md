# Turing Knowledge Graph Project

这是一个用于构建基于图灵（Turing）的知识图谱全流程项目。本项目从维基百科等数据源采集语料，通过弱监督策略结合 BiLSTM-CRF 模型进行命名实体识别（NER），并利用规则进行属性与关系抽取，最后经过实体消歧、知识融合，将构建好的知识图谱导入到 Neo4j 图数据库中进行存储与查询。

## 环境依赖

请确保您的环境中已安装 Python 3.8 或以上版本。项目依赖的第三方库已整理在 `requirements.txt` 中。

### 安装依赖库

```bash
pip install -r requirements.txt
```

使用的主要库包括：
- `wikipedia`: 用于从维基百科爬取语料数据。
- `torch`: PyTorch 深度学习框架，用于构建和训练 BiLSTM-CRF NER 模型。
- `tqdm`: 用于在终端显示处理进度条。
- `neo4j`: Python 的 Neo4j 官方驱动，用于与图数据库交互。

## 操作指南与启动指令

整个知识图谱构建流程分为以下几步，请按顺序执行相应脚本：

### 1. 数据收集与梳理
通过给定的种子实体列表，从维基百科获取相关页面的摘要和正文语料。
- **实现方法:** 使用 `wikipedia` 库调用维基百科 API，依据预定义的种子词条（如“艾伦·图灵”及相关机构、人物）抓取非结构化文本内容，建立项目底层事实语料库。
- **启动指令:** `python scripts/scraper.py`
- **输出文件:** 
  - `data/raw/turing_schema_corpus.jsonl` (原始语料)
  - `data/raw/turing_schema_sources.csv` (数据源记录)

### 2. NER模型训练与实体抽取
本项目采用基于词典和规则等弱监督方式自动生成训练数据，并训练字符级的 BiLSTM-CRF 模型用于命名实体识别。
- **实现方法:** 结合了深度语义特征与概率图模型约束：
  - **BiLSTM (双向长短期记忆网络):** 负责捕获句子中字符级的前后置上下文语义，充分提取特征。
  - **CRF (条件随机场):** 在 BiLSTM 的输出基础上，加入序列标签的全局转移概率约束（例如限定 `I-PER` 之前必须是 `B-PER` 或 `I-PER`），以确保提取出的实体边界合法且全局最优。
- **数据准备 (标注):** `python scripts/train_ner_bilstm_crf.py --mode prepare`
  - **中间输出:** `data/intermediate/ner_char_bio.jsonl`
- **模型训练:** `python scripts/train_ner_bilstm_crf.py --mode train`
  - **模型输出:** `data/output/ner_bilstm_crf/` (包含词汇表 `char_vocab.json`, `tag_vocab.json` 以及模型权重 `model.pt`)

### 3. 属性抽取
使用预训练或规则逻辑抽取实体对应的属性信息。
- **实现方法:** 基于在前一步 NER 中提取出的命名实体，辅助使用正则表达式（Regex）和预设的结构化词法模板，将陈述性句子中对应的实体属性（如出生日期、国籍等）准确截取。
- **启动指令:** `python scripts/extract_attributes.py`
- **输出文件:** 
  - `data/output/attributes_rule/attrs.csv`
  - `data/output/attributes_rule/attrs.jsonl`

### 4. 关系抽取
在抽取出实体后，通过触发词和子句边界规则，抽取实体之间的三元组关系。
- **实现方法:** 采用基于触发词（Trigger Words）和启发式边界逻辑的方法。通过识别预先定义好的代表某种关系的动词或短语（比如“出生于”、“曾就读于”等），依据标点符号和子句边界定位该触发词左右的“主语（Subject）”与“宾语（Object）”实体，最终组成关系三元组。
- **启动指令:** `python scripts/extract_relations.py`
- **输出文件:** 
  - `data/output/relations_rule/edges.csv`
  - `data/output/relations_rule/edges.jsonl`

### 5. 实体消歧与归一化
针对抽取的实体进行字符串清洗和规范化（例如去冠词、统一大小写缩写等），确保同一个实体在图谱中有唯一的节点表示。
- **实现方法:** 应用基于文本相似度与规则清洗的算法。去除不同表达中的停用词或冠词，并通过特定字典做全称与缩写（如“Turing”与“Alan Turing”）的一致性对齐，为每一个现实世界中的对象分配图谱内的唯一节点 ID。
- **启动指令:** `python scripts/normalize_entities.py`
- **输出文件:** 
  - `data/output/kg_final/nodes.csv` (规范节点)
  - `data/output/kg_final/attrs_normalized.csv` (规范属性)
  - `data/output/kg_final/edges_normalized.csv` (规范关系)

### 6. 知识融合 
将多源数据、规则和模型抽取相互补充的数据进行冲突检测与权重融合，生成项目的主控（Master）知识数据。
- **实现方法:** 采用基于冲突检测和启发式投票策略的方式。当对同一实体抽取出矛盾关系或属性时，依靠频率统计权重、规则可信度权重来选择最优信息源，舍弃错误或冗余的信息，从而将粗糙的三元组合并为高置信度事实。
- **启动指令:** `python scripts/knowledge_fusion.py`
- **输出文件:** 
  - `data/output/kg_master/master_attrs.csv`
  - `data/output/kg_master/master_edges.csv`

### 7. 知识存储 (Neo4j)
将最终融合的主控节点和边数据导入 Neo4j 图数据库。执行前请确保本地或远程 Neo4j 数据库已启动，并在脚本中配置对应的连接信息。
- **实现方法:** 利用 Neo4j 的官方 Python 驱动程序及 Cypher 图查询语言，设置各实体节点 ID 的唯一约束并执行批量插入与边建立操作，以支持随后复杂网络关系的快速图遍历计算。
- **启动指令:** `python scripts/import_to_neo4j.py`
- **输出目标:** Neo4j Database

## 项目结构

```
knowledge-project/
├── requirements.txt            # 项目依赖列表
├── docs/
│   └── schema.md               # 知识图谱本体（Schema）定义文档
├── scripts/
│   ├── scraper.py              # 数据采集脚本
│   ├── train_ner_bilstm_crf.py # 弱监督数据生成与NER模型训练脚本
│   ├── extract_attributes.py   # 属性抽取脚本
│   ├── extract_relations.py    # 关系抽取脚本
│   ├── normalize_entities.py   # 实体规范化与消歧脚本
│   ├── knowledge_fusion.py     # 知识融合脚本
│   └── import_to_neo4j.py      # Neo4j图数据库导入脚本
└── data/
    ├── raw/                    # 原始抓取数据
    ├── intermediate/           # 中间结果（如NER格式化训练数据）
    └── output/                 # 各个抽取阶段输出的结构化数据
        ├── ner_bilstm_crf/     # 模型文件与词表
        ├── attributes_rule/    # 原始属性抽取结果
        ├── relations_rule/     # 原始关系抽取结果
        ├── kg_final/           # 归一化后的实体、属性、关系
        └── kg_master/          # 最终融合的知识图谱主控数据
```

## 数据结构
在数据方面，主要有以下几类格式：
*   **JSONL语料文件 (`*.jsonl`)**: 以 JSON Lines 格式存储。爬取的原数据包含网页 URL、标题和文本块；模型训练过程中包含文本字符串及对应的 BIO 对应序列标签；抽取结果包含 `subject`, `relation`, `object` 等信息。
*   **CSV表格数据 (`*.csv`)**: 用于存储实体节点定义、规范化的关系边（source_id, target_id, relation_type）及属性（node_id, attr_name, attr_value）。主要作为标准化输出并在最后入库时进行读取。
*   **模型物料 (`*.pt`, `*.json`)**: PyTorch 相关的张量数据以及字符与 Tags 的映射表。
具体本体定义规则和实体类别详情请参考 `docs/schema.md`。

## 未来改进方向
1. **模型升级:** 从当前的 BiLSTM-CRF 升级到基于 Transformer 架构的大模型（如 BERT、RoBERTa）或是可以直接使用 LLM (大语言模型) 提示词工程来做实体和关系的联合抽取，以提高准确率。
2. **关系抽取策略:** 目前采用基于规则与边界截断的关系抽取，后续可以引入深度学习的关系分类或依存句法分析，提升复杂句式的抽取成功率。
3. **更完备的实体消歧:** 引入图嵌入技术或外部知识库 (如 Wikidata)，改进仅仅依赖字符串本身进行实体对齐的方法。
4. **多模态与多源扩建:** 不再仅依赖维基百科，可以增加学术论文或专著书籍的数据进行信息提取，解决数据稀疏性和单一来源倾向。
5. **增量更新机制:** 建立任务流与增量存储通道，当语料或知识源发生变化时能以最小代价更新图谱而无需重新全量运行所有步骤。
