from py2neo import Graph as Py2NeoGraph, Node, Relationship
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import subprocess
import pprint

# Neo4j Auraの接続情報をシークレットから取得
uri = "neo4j+s://bb2d8548.databases.neo4j.io:7687"
user = "neo4j"
password = "EgSwGZ8qdRHtWQhWqEyXmhr9VU8L4a769M9dZo1pnLs"

# Neo4jに接続
try:
    graph = Py2NeoGraph(uri, auth=(user, password))
    print("Connection to Neo4j successful!")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")

# 修正後のクエリを作成
cypher_queries = []

# 人物ノードの作成
person_nodes = [
    "(:Person {name: '私', role: '語り手', occupation: '教師・著者', characteristic: '三人の娘の父', document_type: '芽生'})"
]
cypher_queries.append(f"CREATE {', '.join(person_nodes)};")


# 人物間の関係
relation_queries = [
    "MATCH (p1:Person {name: '私'}), (p2:Person {name: '妻'}) CREATE (p1)-[:MARRIED_TO {document_type: '芽生'}]->(p2);"
]
cypher_queries.extend(relation_queries)

# 各クエリをNeo4jに送信して実行
for query in cypher_queries:
    query = query.strip()  # 各行の前後の空白を削除
    if query:  # 空行でないことを確認
        try:
            graph.run(query)
            print(f"Query executed successfully: {query}")
        except Exception as e:
            print(f"Error executing query: {query}\n{e}")
