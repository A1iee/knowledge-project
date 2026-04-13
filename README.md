# 图灵知识图谱工程

## 项目概述
本项目以 Alan Turing 为核心对象，构建可扩展、可查询、可追溯的知识图谱。
当前实现了从语料构建到实体/关系导出 CSV 的完整流水线，并支持导入 Neo4j 可视化。

## 项目状态（2026-04-13）
- 已完成：Wikipedia 语料构建
- 已完成：实体抽取（Rule / CRF）
- 已完成：实体规范化与基础消歧
- 已完成：关系抽取与去重
- 已完成：中间结果与图谱结果导出（CSV）

## 技术栈
- 数据来源：Wikipedia API（python wikipedia）
- 抽取实现：Python
- 图数据格式：CSV（nodes / edges）
- 图数据库目标：Neo4j + Cypher

## 数据模型
详细定义见 `docs/schema.md`。

实体类型：
- Person
- Organization
- Theory
- Work
- Event

关系类型：
- STUDIED_AT
- WORKED_AT
- PROPOSED
- PUBLISHED
- PARTICIPATED_IN
- INFLUENCED
- RELATED_TO

## 项目结构

```text
.
├─ README.md
├─ data/
│  ├─ raw/
│  ├─ intermediate/
│  └─ output/
├─ docs/
│  └─ schema.md
├─ scripts/
│  ├─ build_wikipedia_corpus.py
│  ├─ run_extraction_rule.py
│  └─ run_extraction_crf.py
└─ src/
   └─ kg_extraction/
      ├─ __init__.py
      ├─ pipeline.py
      └─ rules.py
```

## 快速开始

### 1. 安装依赖

基础依赖：

```bash
pip install wikipedia
```

CRF 额外依赖：

```bash
pip install sklearn-crfsuite
```

建议 Python 3.10 及以上版本运行本项目。

### 2. 构建语料

默认构建：

```bash
python scripts/build_wikipedia_corpus.py
```

默认输出：`data/raw/turing_corpus.jsonl`

常用参数：
- `--filename`：输出文件名（仍保存到 `data/raw/`）
- `--titles`：自定义页面标题
- `--lang`：语言（默认 `en`）
- `--source`：来源标识（默认 `wikipedia`）

示例：

```bash
python scripts/build_wikipedia_corpus.py --titles "Alan Turing" "Turing machine" --filename turing_custom.jsonl
```

### 3. 运行抽取

规则抽取：

```bash
python scripts/run_extraction_rule.py --input data/raw/turing_corpus.jsonl
```

CRF 抽取（推荐独立语料）：

```bash
python scripts/build_wikipedia_corpus.py --filename turing_corpus_crf.jsonl
python scripts/run_extraction_crf.py --input data/raw/turing_corpus_crf.jsonl
```

说明：
- Rule 与 CRF 已使用独立启动脚本
- CRF 不会自动回退到 Rule

### 4. CRF 常见问题

报错：`CRF training data is empty`

含义：当前语料未命中足够词典实体，无法构造 CRF 弱标注训练样本。

处理方式：
- 增加图灵主题页面（人物、组织、理论、作品、事件）
- 扩充 `src/kg_extraction/rules.py` 中 `GAZETTEER`
- 先运行 rule 检查语料质量，再切换到 crf

## 流程说明（从语料到图谱）

1. `build_wikipedia_corpus.py` 构建 `data/raw/*.jsonl`
2. `run_extraction_rule.py` 或 `run_extraction_crf.py` 抽取 mention
3. pipeline 自动执行实体消歧与关系抽取
4. 导出 `data/intermediate/*.csv` 与 `data/output/*.csv`

## 输出文件说明

中间结果：
- `data/intermediate/entity_mentions*.csv`：实体提及
- `data/intermediate/entities_resolved*.csv`：实体规范化与消歧
- `data/intermediate/relation_candidates*.csv`：关系候选与证据

最终结果：
- `data/output/nodes*.csv`：节点
- `data/output/edges*.csv`：边

命名规则：
- Rule 结果：`nodes.csv`、`edges.csv` 等默认文件名
- CRF 结果：`nodes_crf.csv`、`edges_crf.csv` 等带 `_crf` 后缀

## 结果复核建议

建议优先人工抽样检查：
- `entity_mentions_crf.csv`：实体类型是否合理
- `entities_resolved_crf.csv`：是否误合并/误拆分
- `relation_candidates_crf.csv`：证据句与关系是否一致

建议抽样规模：
- 快速验收：50 到 100 条
- 小型金标：200 到 500 句

## Neo4j 导入与可视化

### 1. 准备文件

优先导入：
- `data/output/nodes_crf.csv`
- `data/output/edges_crf.csv`

### 2. 放入 import 目录

- 在 Neo4j Desktop 的 DBMS 设置中查看 import 目录
- 复制 CSV 文件到 import 目录

### 3. 执行导入

可选：清空测试图谱。

```cypher
MATCH (n) DETACH DELETE n;
```

导入节点：

```cypher
LOAD CSV WITH HEADERS FROM 'file:///nodes_crf.csv' AS row
WITH row WHERE row.entity_id IS NOT NULL AND row.entity_id <> ''
MERGE (n:Entity {entity_id: row.entity_id})
SET n.name = row.name,
    n.label = row.label,
    n.aliases = row.aliases,
    n.source = row.source;
```

导入关系：

```cypher
LOAD CSV WITH HEADERS FROM 'file:///edges_crf.csv' AS row
WITH row
WHERE row.start_id IS NOT NULL AND row.start_id <> ''
  AND row.end_id IS NOT NULL AND row.end_id <> ''
MATCH (s:Entity {entity_id: row.start_id})
MATCH (t:Entity {entity_id: row.end_id})
MERGE (s)-[r:RELATED_TO {relation: row.relation}]->(t)
SET r.evidence = row.evidence,
    r.source = row.source,
    r.confidence = row.confidence;
```

### 4. 导入后校验

```cypher
MATCH (n:Entity) RETURN count(n) AS node_count;
MATCH ()-[r]->() RETURN count(r) AS edge_count;
```

推荐截图：
- 全图概览
- `Alan Turing` 1 跳子图

中心子图查询：

```cypher
MATCH (a:Entity {name: 'Alan Turing'})-[r]-(b)
RETURN a, r, b
LIMIT 80;
```

## 当前实现边界

已实现：
- 句级实体抽取（Rule / CRF）
- 基础实体消歧（名称、标签、上下文）
- 句内关系抽取（触发词 + 类型约束）

待增强：
- 跨句指代与复杂关系
- 人工复核闭环与自动评估
- 更细粒度关系属性

## 关键时间节点
- 第一阶段：第 5 周课前完成仓库可访问版本
- 第二阶段：2026-04-30 00:00 前完成图谱构建并输出最终可视化截图
