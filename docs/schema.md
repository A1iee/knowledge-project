# 图灵中心知识图谱 Schema

## 1. 设计目标

本 Schema 面向“以 Alan Turing 为中心的人物知识图谱”构建，目标不是覆盖所有外围知识，而是优先保证围绕图灵本人的事实质量、关系可解释性和图谱可扩展性。

图谱按两层组织：

- `core`：只保留与 `Alan Turing` 直接一跳或二跳强相关的事实
- `expanded`：保留核心实体之间延展出的外围关系，便于后续补充背景知识

本 Schema 同时约束三类内容：

1. 实体类型（Node Labels）
2. 关系类型（Edge Types）
3. 属性规范（Node Properties）

## 2. 分层原则

### 2.1 Core 层

满足以下任一条件的事实进入 `core`：

- 关系的起点或终点是 `Alan Turing`
- 关系两端都属于图灵的一跳实体，且关系能直接解释图灵的生平、学术贡献、工作背景、合作网络或历史影响

### 2.2 Expanded 层

满足以下条件的事实进入 `expanded`：

- 实体位于图灵两跳范围内
- 事实对理解图灵相关人物、机构、概念、著作、事件有辅助价值
- 事实不直接描述图灵本人，但与核心层实体存在稳定连接

## 3. 实体类型

### 3.1 Person

- 含义：自然人
- 核心属性：`uid`, `name`, `birth_date`, `death_date`, `nationality`, `occupation`, `aliases`, `source`
- 示例：`Alan Turing`, `Joan Clarke`, `Alonzo Church`, `Max Newman`

### 3.2 Organization

- 含义：学校、研究机构、政府部门、实验室、工作单位
- 核心属性：`uid`, `name`, `org_type`, `established_year`, `aliases`, `source`
- 示例：`King's College, Cambridge`, `Bletchley Park`, `University of Manchester`

### 3.3 Location

- 含义：城市、地区、国家或具体地理地点
- 核心属性：`uid`, `name`, `country`, `aliases`, `source`
- 示例：`London`, `Manchester`, `Princeton`

### 3.4 Concept

- 含义：理论、概念、方法、问题、测试
- 核心属性：`uid`, `name`, `domain`, `description`, `aliases`, `source`
- 示例：`Turing machine`, `Turing test`, `Halting problem`

### 3.5 Artifact

- 含义：设备、系统、工程产物、机器
- 核心属性：`uid`, `name`, `artifact_type`, `creation_year`, `description`, `aliases`, `source`
- 示例：`Bombe`, `Automatic Computing Engine`, `Enigma machine`

### 3.6 Event

- 含义：历史事件、任命、审判、战争、纪念活动
- 核心属性：`uid`, `name`, `start_date`, `end_date`, `description`, `aliases`, `source`
- 示例：`World War II`, `1952 Conviction`, `2013 Royal Pardon`

### 3.7 Publication

- 含义：论文、文章、书籍、报告
- 核心属性：`uid`, `title`, `publish_year`, `journal_or_venue`, `doi`, `aliases`, `source`
- 示例：`On Computable Numbers`, `Computing Machinery and Intelligence`

### 3.8 Honor

- 含义：奖项、纪念物、纪念活动、法案或以人物命名的荣誉对象
- 核心属性：`uid`, `name`, `honor_type`, `year`, `description`, `source`
- 示例：`Turing Award`, `Alan Turing law`, `Bank of England 50 note`

## 4. 核心关系类型

以下关系是本项目推荐保留的主关系集合，关系名应在抽取、融合、入库阶段保持一致。

### 4.1 生平与社会关系

#### BORN_IN

- Domain: `Person`
- Range: `Location`
- 含义：人物出生地

#### DIED_IN

- Domain: `Person`
- Range: `Location`
- 含义：人物逝世地

#### EDUCATED_AT

- Domain: `Person`
- Range: `Organization`
- 含义：人物就读或接受教育的机构

#### WORKED_AT

- Domain: `Person`
- Range: `Organization`
- 含义：人物工作、任职或长期服务的机构

#### COLLEAGUE_OF

- Domain: `Person`
- Range: `Person`
- 含义：同事、共事者、研究合作伙伴

### 4.2 学术与技术贡献

#### PROPOSED

- Domain: `Person`
- Range: `Concept`
- 含义：提出理论、测试、思想或概念

#### WORKED_ON

- Domain: `Person`
- Range: `Concept` or `Artifact`
- 含义：参与研究、设计、改进或破解某一对象

#### AUTHORED

- Domain: `Person`
- Range: `Publication`
- 含义：撰写论文、书籍或文章

### 4.3 事件与影响

#### PARTICIPATED_IN

- Domain: `Person` or `Organization`
- Range: `Event`
- 含义：参与某个历史事件、战争或项目

#### AFFECTED_BY

- Domain: `Person`
- Range: `Event`
- 含义：受到某一历史事件、审判、法案或社会事件影响

### 4.4 荣誉与纪念

#### HONORED_BY

- Domain: `Person`
- Range: `Honor`
- 含义：人物被某奖项、纪念项目、法案或纪念物致敬

#### NAMED_AFTER

- Domain: `Honor` or `Organization` or `Artifact`
- Range: `Person`
- 含义：荣誉、机构或物品以某人物命名

## 5. 当前项目不再使用的关系命名约束

为避免抽取阶段出现语义漂移，以下做法不再推荐：

- 不再新增 `Work`, `Award`, `Theory`, `Misc` 这类未定义实体类型
- 不再使用语义边界不清晰的兜底关系
- 不再通过“包含 Turing 关键字自动加分”的方式突出中心人物

如果某条候选事实无法映射到本 Schema 的实体类型和关系类型，应优先丢弃，而不是强行入图。

## 6. 输出建议

最终节点和边建议额外携带以下字段，便于核心层与扩展层管理：

- 节点：`layer`, `center_distance`, `aliases`, `source_titles`
- 边：`layer`, `is_core`, `centered_on`, `evidence`, `source`, `confidence`, `weight`

其中：

- `layer` 取值为 `core` 或 `expanded`
- `centered_on` 固定为 `alan_turing`
- `center_distance` 表示节点到图灵中心节点的最短距离

## 7. 目标图谱形态

最终图谱应满足：

- `Alan Turing` 是唯一中心节点
- 与图灵直接相关的一跳事实优先高精度保留
- 二跳实体仅在与图灵主线相关时进入核心层
- 每条关系都有可追溯证据句和来源
- 错关系宁可漏掉，也不进入最终图谱
