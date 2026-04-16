import argparse
import csv
from pathlib import Path
from neo4j import GraphDatabase

def create_constraints(tx):
    # 为每种 Label 创建唯一约束，大幅提升导入和查询速度
    labels = ["Person", "Organization", "Location", "Honor"]
    for label in labels:
        try:
            tx.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        except Exception as e:
            print(f"Warning on constraint for {label}: {e}")

def import_nodes(tx, nodes_path: Path):
    with nodes_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            # 将以 '|' 分割的别名转为 Python 列表，Neo4j 原生支持列表属性
            aliases = [a.strip() for a in row["aliases"].split("|") if a.strip()]
            
            # Neo4j 不支持参数化 Label，采用安全字符串拼接 (Label 是受控的)
            query = f"""
            MERGE (n:{label} {{id: $id}})
            SET n.name = $name,
                n.aliases = $aliases
            """
            tx.run(query, id=row["id"], name=row["name"], aliases=aliases)

def import_attributes(tx, attrs_path: Path):
    with attrs_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["entity_uid"]
            attr_name = row["attribute_name"]
            attr_val = row["attribute_value"]
            sources = row["fused_sources"]
            
            # Neo4j 不支持参数化属性名。
            # 这里巧妙利用 CASE 构建数组：如果已有该属性，则追加(支持多职业等)；否则创建新数组/单值
            query = f"""
            MATCH (n {{id: $uid}})
            SET n.{attr_name} = CASE 
                WHEN n.{attr_name} IS NULL THEN [$val] 
                WHEN $val IN n.{attr_name} THEN n.{attr_name}
                ELSE n.{attr_name} + $val 
            END,
            n._{attr_name}_sources = $sources
            """
            tx.run(query, uid=uid, val=attr_val, sources=sources)

def import_edges(tx, edges_path: Path):
    with edges_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_type = row["relation"]
            
            # Neo4j 不支持参数化关系类型
            query = f"""
            MATCH (a {{id: $start_uid}})
            MATCH (b {{id: $end_uid}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.sources = $sources,
                r.evidence_count = toInteger($count)
            """
            tx.run(query, 
                start_uid=row["start_uid"], 
                end_uid=row["end_uid"], 
                sources=row["fused_sources"], 
                count=row["evidence_count"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", default="data/output/kg_final/nodes.csv")
    p.add_argument("--attrs", default="data/output/kg_master/master_attrs.csv")
    p.add_argument("--edges", default="data/output/kg_master/master_edges.csv")
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    args = p.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    try:
        with driver.session() as session:
            print("1. Creating Constraints...")
            session.execute_write(create_constraints)

            print("2. Importing Nodes...")
            session.execute_write(import_nodes, Path(args.nodes))

            print("3. Importing Attributes...")
            session.execute_write(import_attributes, Path(args.attrs))

            print("4. Importing Relations (Edges)...")
            session.execute_write(import_edges, Path(args.edges))

        print("Import completely successfully!")
    finally:
        driver.close()

if __name__ == "__main__":
    main()