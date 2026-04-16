# 图灵主题知识图谱 Schema

## 1. 设计目标
本 Schema 面向“特定历史人物图谱”场景，以 Alan Turing 为核心，支持三类目标：

1. 抽取层：让文本抽取有明确的实体与关系边界，降低漏抽和乱连。
2. 存储层：可直接映射到 Neo4j（节点标签 + 关系类型 + 属性）。
3. 迭代层：允许先落最小字段，再逐步扩展，减少“节点多、边很稀疏”的问题。

本 Schema 定义三个核心维度：

1. 实体类型（Entities / Node Labels）
2. 关系类型（Relations / Edge Types）
3. 属性规范（Attributes / Properties）

## 2. 实体类型（Entity Types）

### 2.1 Person（人物）
- 含义：自然人。
- 核心属性：uid, name, birth_date, death_date, nationality, occupation, aliases, source
- 示例：Alan Turing, Joan Clarke, Winston Churchill

### 2.2 Organization（组织/机构）
- 含义：学校、政府、军方、实验室、公司等。
- 核心属性：uid, name, org_type, established_year, aliases, source
- 示例：King's College Cambridge, Bletchley Park, University of Manchester

### 2.3 Location（地点）
- 含义：城市、地区、国家或具体地理点。
- 核心属性：uid, name, country, latitude, longitude, aliases, source
- 示例：London, Manchester, Princeton

### 2.4 Concept（理论/概念）
- 含义：理论、方法、模型、问题、测试。
- 核心属性：uid, name, domain, description, aliases, source
- 示例：Turing Machine, Turing Test, Halting Problem, Morphogenesis

### 2.5 Artifact（设备/系统/工程产物）
- 含义：硬件、系统、机器、工程实现。
- 核心属性：uid, name, artifact_type, creation_year, description, aliases, source
- 示例：Bombe, Automatic Computing Engine (ACE), Enigma machine

### 2.6 Event（历史事件）
- 含义：战争、审判、任命、赦免、纪念活动等事件。
- 核心属性：uid, name, start_date, end_date, description, aliases, source
- 示例：World War II, 1952 Conviction, 2013 Royal Pardon

### 2.7 Publication（著作/论文）
- 含义：论文、文章、报告、书籍。
- 核心属性：uid, title, publish_year, journal_or_venue, doi, aliases, source
- 示例：On Computable Numbers

### 2.8 Honor（荣誉/纪念）
- 含义：奖项、勋章、法案、货币纪念、雕像等。
- 核心属性：uid, name, honor_type, year, description, source
- 示例：Turing Award, OBE, Bank of England £50 note

## 3. 关系类型（Relation Types）

## 3.1 人物社会与生平关系（Social & Life）

### BORN_IN
- Domain: Person
- Range: Location
- 含义：出生地关系

### DIED_IN
- Domain: Person
- Range: Location
- 含义：逝世地关系

### EDUCATED_AT
- Domain: Person
- Range: Organization
- 含义：求学经历

### WORKED_AT
- Domain: Person
- Range: Organization
- 含义：任职或长期工作

### COLLEAGUE_OF
- Domain: Person
- Range: Person
- 含义：同事/合作研究关系

### FRIEND_OF
- Domain: Person
- Range: Person
- 含义：朋友、伴侣或紧密私人关系

## 3.2 学术与技术成就关系（Academic & Achievements）

### PROPOSED
- Domain: Person
- Range: Concept
- 含义：提出理论或概念

### INVENTED
- Domain: Person
- Range: Artifact
- 含义：发明/主导研发设备或系统

### AUTHORED
- Domain: Person
- Range: Publication
- 含义：撰写论文或著作

### WORKED_ON
- Domain: Person
- Range: Artifact or Concept
- 含义：参与破解、研究、改进某对象

## 3.3 事件与荣誉关系（Events & Honors）

### PARTICIPATED_IN
- Domain: Person or Organization
- Range: Event
- 含义：参与某事件

### AFFECTED_BY
- Domain: Person
- Range: Event
- 含义：受某历史事件影响

### AWARDED
- Domain: Person
- Range: Honor
- 含义：被授予荣誉

### NAMED_AFTER
- Domain: Honor or Organization
- Range: Person
- 含义：奖项、机构或纪念物以某人命名
