# 图灵知识图谱 Schema

## 1. 目标
本 schema 的目的不是覆盖全部事实，而是先把可入库、可查询、可去重的最小结构固定下来。

后续所有抽取结果都必须先映射到这里定义的实体类型和关系类型，再进入 CSV 和 Neo4j。

## 2. 实体类型

### Person
- 含义：人物
- 示例：Alan Turing
- 必填字段：`entity_id`, `name`, `label`, `source`
- 建议字段：`birth_year`, `death_year`, `nationality`, `aliases`

### Organization
- 含义：机构、学校、公司、实验室、政府部门等
- 示例：University of Cambridge, Bletchley Park
- 必填字段：`entity_id`, `name`, `label`, `source`
- 建议字段：`location`, `aliases`

### Theory
- 含义：理论、方法、概念、模型
- 示例：Turing Machine, Turing Test
- 必填字段：`entity_id`, `name`, `label`, `source`
- 建议字段：`year`, `description`, `aliases`

### Work
- 含义：论文、著作、报告、文章、专利等成果
- 示例：On Computable Numbers
- 必填字段：`entity_id`, `name`, `label`, `source`
- 建议字段：`year`, `venue`, `type`, `aliases`

### Event
- 含义：事件、会议、任职、战争相关节点、获奖等
- 示例：World War II, Enigma codebreaking
- 必填字段：`entity_id`, `name`, `label`, `source`
- 建议字段：`period`, `description`, `aliases`

## 3. 关系类型

### STUDIED_AT
- 方向：`Person -> Organization`
- 含义：人物在某机构求学

### WORKED_AT
- 方向：`Person -> Organization`
- 含义：人物在某机构工作、任职或参与科研

### PROPOSED
- 方向：`Person -> Theory`
- 含义：人物提出理论、模型或概念

### PUBLISHED
- 方向：`Person -> Work`
- 含义：人物发表或出版作品

### PARTICIPATED_IN
- 方向：`Person -> Event`
- 含义：人物参与某事件

### INFLUENCED
- 方向：`Person -> Person|Theory|Work|Event`
- 含义：人物对其他人物、理论、作品或事件产生影响

### RELATED_TO
- 方向：`Any -> Any`
- 含义：弱关系、补充关系、暂未细分关系

## 4. 统一约束

### 4.1 命名规则
- 同一实体只保留一个 canonical name
- 别名统一放在 `aliases`
- 英文名优先保留原始拼写，不强行翻译
- 同一机构、同一作品、同一理论不得重复建点

### 4.2 消歧规则
- 相同名称但上下文不一致时，默认拆分为不同实体
- 相同名称但指代明显一致时，合并到同一 `entity_id`
- 无法判断时先保留为待审核，不要强行合并

### 4.3 关系规则
- 每条关系都必须有 `source`
- 每条关系都尽量附带 `evidence`，最好是原句或原文片段
- 关系方向必须固定，不能同义双向混用
- `RELATED_TO` 只作为兜底，不作为主关系

### 4.4 入库规则
- 节点表只收标准化后的实体
- 边表只收标准化后的关系
- 抽取结果先进入中间结果，再导出 CSV
- 置信度低的结果单独存档，不直接入库

## 5. 推荐 CSV 字段

### 5.1 当前实现（output/nodes.csv）
- `entity_id`
- `name`
- `label`
- `aliases`
- `source`

说明：当前版本仅输出最小可导入字段，优先保证稳定导入与可追溯。

### 5.2 当前实现（output/edges.csv）
- `start_id`
- `end_id`
- `relation`
- `evidence`
- `source`
- `confidence`

说明：`confidence` 目前由规则策略给出（`high`/`medium`）。

### 5.3 规划扩展字段（未全部实现）
- 节点扩展：`description`, `year`, `location`, `type_specific_fields`
- 边扩展：`extract_method`, `disputed`, `time_scope`

## 6. 最小验收标准
- 所有抽出的实体都能归入 5 种实体类型之一
- 所有抽出的关系都能映射到 7 种关系类型之一
- 每个实体和关系都能追溯到来源
- 去掉重复、歧义后仍能稳定导入 Neo4j

## 7. 中间结果 Schema（已实现）

### 7.1 intermediate/entity_mentions.csv
- `doc_id`
- `sentence_id`
- `mention_text`
- `canonical_name`
- `label`
- `context`
- `source`
- `evidence`
- `confidence`

### 7.2 intermediate/entities_resolved.csv
- `entity_id`
- `name`
- `label`
- `aliases`
- `source`
- `evidence`
- `confidence`
- `pending_review`

### 7.3 intermediate/relation_candidates.csv
- `start_id`
- `end_id`
- `relation`
- `evidence`
- `source`
- `confidence`
- `extract_method`
- `disputed`

## 8. 当前实现边界（2026-04-07）

### 8.1 已完成
- 句级实体抽取（词典 + 规则）
- 实体规范化与基础消歧（名称、类型、年份/组织上下文）
- 句内关系抽取（触发词 + 类型/方向约束）
- 关系去重与证据合并

### 8.2 待增强
- 跨句指代消解
- 低置信度人工复核闭环
- 多来源冲突裁决策略
- 更细粒度关系时间属性

## 9. 数据流与入库映射
- 原始语料：`data/raw/turing_corpus.jsonl`
- 抽取中间层：`data/intermediate/*.csv`
- 入库结果层：`data/output/nodes.csv`, `data/output/edges.csv`

导入 Neo4j 时建议仅使用 `data/output` 下文件；`data/intermediate` 用于审计、回溯与误差分析。


