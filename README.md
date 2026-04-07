# 图灵知识图谱工程

## 项目概述
本项目以 Alan Turing 为核心对象，构建可扩展、可查询、可追溯的知识图谱。
当前已实现从语料采集到实体/关系 CSV 导出的最小可用流水线，支持后续导入 Neo4j。

## 当前进度（2026-04-07）
- 已完成：维基百科语料抓取脚本
- 已完成：实体抽取、实体规范化与消歧、关系抽取（规则驱动）
- 已完成：中间结果与最终节点/边 CSV 导出
- 已完成：schema 文档与目录结构整理

## 技术栈
- 数据获取：Wikipedia API（python wikipedia）
- 数据处理：Python
- 图数据格式：CSV（nodes/edges）
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

## 当前项目结构

```text
.
├─ README.md
├─ data/
│  ├─ raw/
│  │  └─ turing_corpus.jsonl
│  ├─ intermediate/
│  │  ├─ entity_mentions.csv
│  │  ├─ entities_resolved.csv
│  │  └─ relation_candidates.csv
│  └─ output/
│     ├─ nodes.csv
│     └─ edges.csv
├─ docs/
│  └─ schema.md
├─ scripts/
│  ├─ build_wikipedia_corpus.py
│  └─ run_extraction.py
└─ src/
   └─ kg_extraction/
      ├─ __init__.py
      ├─ pipeline.py
      └─ rules.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install wikipedia
```

### 2. 构建图灵语料库

```bash
python scripts/build_wikipedia_corpus.py
```

默认输出：`data/raw/turing_corpus.jsonl`

可选参数：
- `--filename`：指定输出文件名（仍保存到 `data/raw/`）
- `--titles`：指定抓取页面标题列表
- `--lang`：语言，默认 `en`
- `--source`：来源标识，默认 `wikipedia`

示例：

```bash
python scripts/build_wikipedia_corpus.py --titles "Alan Turing" "Turing machine" --filename turing_custom.jsonl
```

### 3. 运行抽取流水线

```bash
python scripts/run_extraction.py
```

可选参数：
- `--input`：指定语料 JSONL 路径

示例：

```bash
python scripts/run_extraction.py --input data/raw/turing_corpus.jsonl
```

## 流水线输出说明

中间结果：
- `data/intermediate/entity_mentions.csv`：实体提及结果
- `data/intermediate/entities_resolved.csv`：实体规范化与消歧结果
- `data/intermediate/relation_candidates.csv`：关系候选与证据

最终结果：
- `data/output/nodes.csv`：图谱节点表
- `data/output/edges.csv`：图谱关系表

## 当前已实现能力

实体抽取：
- 分句
- 候选实体识别
- 类型判定
- mention 记录生成

实体消歧：
- 名称规范化
- 消歧打分
- 合并或新建实体
- 统一实体 ID 生成

关系抽取：
- 句内实体对生成与触发词匹配
- 类型/方向约束校验
- 关系去重与证据合并

## 下一步计划
- 增加人工复核清单导出（低置信实体/关系）
- 增加 Neo4j 导入脚本与校验查询
- 提升跨句指代与复杂关系抽取能力

## 关键时间节点
- 第一阶段：第 5 周课前完成仓库可访问版本
- 第二阶段：2026-04-30 00:00 前完成图谱构建并输出最终可视化截图
