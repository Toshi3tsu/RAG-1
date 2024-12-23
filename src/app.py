import os
import io
import re
import streamlit as st
import tempfile
import pandas as pd
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
import base64
import zipfile
import unicodedata
from langchain.graphs.neo4j_graph import Neo4jGraph
import math

# Wide modeをデフォルトで有効にし、ダークモードにする
st.set_page_config(layout="wide")

# キーワードとドキュメントテキストのUnicode正規化を行う関数
def normalize_text(text):
    # NFKC正規化を適用
    return unicodedata.normalize('NFKC', text)

# 1. ファイルの読み込みや処理中のエラー処理
def load_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        import chardet
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)['encoding']
            return raw_data.decode(detected_encoding)
    except Exception as e:
        st.error(f"An error occurred while reading the file: {str(e)}")
        return None

# 2. ファイル名がクエリに含まれているかチェック
def file_name_in_query(file_name, query):
    return file_name.lower() in query.lower()

# 3. メタデータ抽出
def extract_metadata(chunk, chat, current_metadata=None):
    tool_description = {
        "type": "function",
        "function": {
            "name": "update_literary_metadata",
            "description": "与えられたテキストチャンクから文学的メタデータを抽出または更新する",
            "parameters": {
                "type": "object",
                "properties": {
                    "Title": {"type": "string", "description": "作品のタイトル"},
                    "Author": {"type": "string", "description": "作品の著者"},
                    "Publication_year": {"type": "string", "description": "出版年"},
                    "Genre": {"type": "string", "description": "作品のジャンル"},
                    "Setting_period": {"type": "string", "description": "物語の舞台となる時代"},
                    "Setting": {"type": "string", "description": "物語の舞台や環境"},
                    "Characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Name": {"type": "string", "description": "登場人物の名前"},
                                "Role": {"type": "string", "description": "登場人物の役割"},
                                "Age": {"type": "integer", "description": "登場人物の年齢"},
                                "Description": {"type": "string", "description": "登場人物の特徴や背景"}
                            }
                        },
                        "description": "主要な登場人物とその詳細情報"
                    },
                    "Point_of_view": {"type": "string", "description": "物語の視点"},
                    "Style": {"type": "string", "description": "作品の文体"},
                    "Themes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "作品の主要テーマ（複数可）"
                    },
                    "Target_audience": {"type": "string", "description": "作品の対象読者"}
                },
                "required": ["Title", "Author", "Publication_year", "Genre", "Setting_period", "Setting", "Characters", "Point_of_view", "Style", "Themes", "Target_audience"]
            }
        }
    }

    system_message = SystemMessage(content="""
    あなたはテキスト分析の専門家です。与えられたテキストチャンクからメタデータを日本語で抽出/作成または更新してください。既存の情報がある場合は、それを考慮して更新してください。
    ・新しい情報が見つかった場合は追加し、既存の情報と矛盾する場合は最も適切と思われる情報を選択してください。
    ・情報が見つからない場合は、既存の情報を保持するか、記入しないでください。
    特に以下の点に注意してください：
    1. 登場人物の情報はできるだけ詳細に抽出し、名前、役割、年齢、特徴を含めてください。
    2. テーマは複数抽出し、配列として返してください。
    3. 既存のメタデータと新しく抽出した情報を統合する際は、より詳細または正確な情報を優先してください。
    """)

    human_message = HumanMessage(content=f"このテキストチャンクを分析し、メタデータを提供または更新してください。現在のメタデータ：{current_metadata}\n\nテキストチャンク：\n{chunk}")

    try:
        response = chat.invoke([system_message, human_message], tools=[tool_description])
        print("Response:", response)  # 追加: responseの内容を確認
        if isinstance(response, AIMessage) and response.tool_calls:
            tool_call = response.tool_calls[0]
            print("Tool Call:", tool_call)  # 追加: tool_callの内容を確認
            # 修正: function.nameではなくtool_call['name']を使用
            if tool_call['name'] == "update_literary_metadata":
                # 修正: function.argumentsではなくtool_call['args']を使用
                updated_metadata = tool_call['args']
                print("Extracted metadata:")
                print(json.dumps(updated_metadata, indent=2, ensure_ascii=False))
            else:
                updated_metadata = {}
                print("No metadata extracted.")
        else:
            updated_metadata = {}
            print("No metadata extracted.")
    except Exception as e:
        print(f"Error in metadata extraction: {e}")
        updated_metadata = {}

    # Ensure all required keys are present and update only if new information is available
    for key in tool_description['function']['parameters']['properties'].keys():
        if key not in updated_metadata or updated_metadata[key] == 'Unknown':
            if current_metadata and key in current_metadata:
                updated_metadata[key] = current_metadata[key]
            else:
                updated_metadata[key] = 'Unknown'

    # Merge character information
    if current_metadata and 'Characters' in current_metadata:
        if isinstance(current_metadata['Characters'], str):
            current_characters = [{'Name': char.split(',')[0].strip(), 'Role': char.split(',')[1].strip() if ',' in char else 'Unknown'}
                                    for char in current_metadata['Characters'].split(';')]
        else:
            current_characters = current_metadata['Characters']

        current_characters_dict = {char['Name']: char for char in current_characters}

        new_characters = updated_metadata.get('Characters', [])
        if isinstance(new_characters, str):
            new_characters = [{'Name': char.split(',')[0].strip(), 'Role': char.split(',')[1].strip() if ',' in char else 'Unknown'}
                                for char in new_characters.split(';')]

        for new_char in new_characters:
            if isinstance(new_char, dict) and 'Name' in new_char:
                if new_char['Name'] in current_characters_dict:
                    current_characters_dict[new_char['Name']].update(new_char)
                else:
                    current_characters_dict[new_char['Name']] = new_char
            elif isinstance(new_char, str):
                name = new_char.split(',')[0].strip()
                role = new_char.split(',')[1].strip() if ',' in new_char else 'Unknown'
                if name in current_characters_dict:
                    current_characters_dict[name]['Role'] = role
                else:
                    current_characters_dict[name] = {'Name': name, 'Role': role}

        updated_metadata['Characters'] = list(current_characters_dict.values())

    # Merge themes
    if current_metadata and 'Themes' in current_metadata:
        if isinstance(current_metadata['Themes'], str):
            current_themes = [current_metadata['Themes']]
        else:
            current_themes = current_metadata['Themes']

        updated_themes = updated_metadata.get('Themes', [])
        if isinstance(updated_themes, str):
            updated_themes = [updated_themes]

        updated_metadata['Themes'] = list(set(current_themes + updated_themes))

    return updated_metadata

# 4. あらすじ抽出
def extract_synopsis(chunk, chat, previous_synopsis="", chunk_id=None):
    tool_description = {
        "type": "function",
        "function": {
            "name": "update_literary_synopsis",
            "description": "与えられたテキストチャンクからあらすじを抽出または更新する",
            "parameters": {
                "type": "object",
                "properties": {
                    "Synopsis": {
                        "type": "string",
                        "description": "更新されたあらすじのテキスト"
                    },
                },
                "required": ["Synopsis"]
            }
        }
    }

    system_message = SystemMessage(content="""
    あなたはテキスト分析の専門家です。与えられたテキストチャンクからあらすじを日本語で作成または更新する任務を担当しています。
    あらすじとは、細かいセリフや行動ではなく、ストーリー展開をまとめたものです。
    前のあらすじが提供されている場合は、それを考慮して新しいチャンクのあらすじを作成してください。
    新しい情報が見つかった場合は追加し、既存の情報と矛盾する場合は最も適切と思われる情報を選択してください。
    あらすじは全体で2000文字以内に収めてください。
    """)

    human_message = HumanMessage(content=f"""
    このテキストチャンクを分析し、あらすじを作成または更新するために `update_literary_synopsis` 関数を呼び出してください。
    前のあらすじ：{previous_synopsis}

    テキストチャンク：
    {chunk}
    """)

    updated_synopsis = ""

    try:
        response = chat.invoke(
            [system_message, human_message],
            tools=[tool_description]
        )
        print("Response:", response)
        if isinstance(response, AIMessage) and 'tool_calls' in response.additional_kwargs:
            for tool_call in response.additional_kwargs['tool_calls']:
                function_name = tool_call.get("function", {}).get("name")
                arguments = tool_call.get("function", {}).get("arguments", "{}")
                print("Function name:", function_name)
                print("Tool arguments:", arguments)
                if function_name == "update_literary_synopsis":
                    updated_synopsis = json.loads(arguments).get("Synopsis", "")
                    print("Updated synopsis:", updated_synopsis)
                else:
                    print("Different tool called or no tool called")
        else:
            # 関数が呼び出されなかった場合のフォールバック
            print("No tool calls found in response. Using the response as synopsis.")
            updated_synopsis = previous_synopsis
    except Exception as e:
        print(f"Error processing response: {e}")

    return updated_synopsis

# 新しいツールの定義: メタデータに基づく回答と十分性の判定
def metadata_tool_description():
    return {
        "type": "function",
        "function": {
            "name": "generate_metadata_response",
            "description": "メタデータに基づいて回答を生成し、その回答が質問に十分に答えているかを判定し、説明を提供する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "Metadata_based_Answer": {"type": "string", "description": "メタデータに基づく回答"},
                    "Sufficiency": {"type": "string", "enum": ["十分", "不十分"], "description": "回答が質問に十分に答えているかどうか"},
                    "Explanation": {"type": "string", "description": "回答の十分性に関する説明"}
                },
                "required": ["Metadata_based_Answer", "Sufficiency", "Explanation"]
            }
        }
    }

# Define your tools (functions) here
tools = [extract_metadata, extract_synopsis]

# 5. メタデータに基づいた回答
def metadata_based_gpt_response(query, metadata_df, selected_file, chat):
    if metadata_df.empty:
        return {"Metadata_based_Answer": "メタデータが利用できません。", "Sufficiency": "不十分", "Explanation": "メタデータが存在しません。"}

    # 選択されたファイルに対応するメタデータをフィルタリング
    file_metadata_df = metadata_df[metadata_df['Source'] == selected_file]

    if file_metadata_df.empty:
        return {"Metadata_based_Answer": "指定されたファイルのメタデータが見つかりません。", "Sufficiency": "不十分", "Explanation": "指定されたファイルのメタデータが存在しません。"}

    # 最初のマッチするメタデータを使用
    metadata = file_metadata_df.iloc[0].to_dict()
    response_text = f"メタデータによると、'{metadata.get('Title', '不明')}'というタイトルの文書は{metadata.get('Author', '不明')}によって{metadata.get('Publication_year', '不明')}年に出版されました。"
    response_text += f"これは{metadata.get('Setting_period', '不明')}の{metadata.get('Setting', '不明')}を舞台とした{metadata.get('Genre', '不明')}作品です。"
    response_text += f"主要な登場人物には{metadata.get('Characters', '不明')}が含まれています。"
    response_text += f"物語は{metadata.get('Point_of_view', '不明')}の視点から{metadata.get('Style', '不明')}のスタイルで語られています。"
    response_text += f"主なテーマは{metadata.get('Themes', '不明')}で、{metadata.get('Target_audience', '不明')}を対象としています。"
    response_text += f"\n\n概要: {metadata.get('Synopsis Display', '不明')}"

    # ツールコール用のシステムメッセージ
    system_message = SystemMessage(content="""
    あなたはテキスト分析の専門家です。以下の質問とメタデータに基づく回答を受け取り、この回答が質問に十分に答えているかどうかを判断してください。
    十分な場合は'十分'、不十分な場合は'不十分'と回答し、その理由を簡潔に説明してください。
    最終的に、'Metadata_based_Answer'、'Sufficiency'、'Explanation'を含むJSON形式で回答を返してください。
    """)

    # ツールコール用の人間メッセージ
    human_message = HumanMessage(content=f"質問: {query}\n\nメタデータに基づく回答: {response_text}")

    # ツールの定義
    tool = metadata_tool_description()

    try:
        response = chat.invoke([system_message, human_message], tools=[tool])
        print("Metadata Tool Response:", response)
        if isinstance(response, AIMessage) and 'tool_calls' in response.additional_kwargs:
            for tool_call in response.additional_kwargs['tool_calls']:
                function_name = tool_call.get("function", {}).get("name")
                arguments = tool_call.get("function", {}).get("arguments", "{}")
                print("Function name:", function_name)
                print("Tool arguments:", arguments)
                if function_name == "generate_metadata_response":
                    tool_response = json.loads(arguments)
                    metadata_based_answer = tool_response.get("Metadata_based_Answer", "回答が生成されませんでした。")
                    sufficiency = tool_response.get("Sufficiency", "不明")
                    explanation = tool_response.get("Explanation", "説明が生成されませんでした。")
                    return {
                        "Metadata_based_Answer": metadata_based_answer,
                        "Sufficiency": sufficiency,
                        "Explanation": explanation
                    }
                else:
                    print("Different tool called or no tool called")
        else:
            print("No tool calls found in response")
    except Exception as e:
        print(f"GPTチェック中にエラーが発生しました: {e}")
        return {
            "Metadata_based_Answer": response_text,
            "Sufficiency": "不十分",
            "Explanation": f"十分性チェック中にエラーが発生しました: {str(e)}"
        }

    return {
        "Metadata_based_Answer": response_text,
        "Sufficiency": "不十分",
        "Explanation": "十分性を判断できませんでした。"
    }

def format_characters(characters):
    if not isinstance(characters, list):
        return str(characters)  # Return as string if not a list
    return ', '.join([f"{char['Name']} ({char['Role']}, Age: {char.get('Age', 'Unknown')})" for char in characters])

def merge_faiss_stores(store1, store2):
    combined_vectors = store1.index.reconstruct_n(0, store1.index.ntotal)  # 全ベクトルを取得
    combined_ids = list(store1.index.docstore.keys())
    for doc in store2.index.docstore:
        combined_vectors = np.vstack([combined_vectors, store2.index.reconstruct(doc)])
        combined_ids.append(doc)
    # 新しいFAISSインデックスを作成
    new_store = FAISS(embeddings, store1.index)
    new_store.index.add(combined_vectors)
    new_store.index.docstore = {id_: combined_vectors[i] for i, id_ in enumerate(combined_ids)}
    return new_store

# Neo4jGraphの初期化（接続情報は環境に合わせて変更）
neo4j_url = st.secrets["NEO4J_URI"]
neo4j_user = st.secrets["NEO4J_USERNAME"]
neo4j_password = st.secrets["NEO4J_PASSWORD"]
try:
    # Neo4jGraphインスタンスを作成
    graph = Neo4jGraph(neo4j_url, neo4j_user, neo4j_password)
    st.write("Neo4jに接続成功しました")
    
    # クエリの例（基本的なクエリをテストする）
    query = """
    MATCH (e:Event)
    RETURN e LIMIT 5
    """
    result = graph.query(query)
    # st.write("クエリ結果:", result)

except Exception as e:
    st.error(f"Neo4jへの接続エラーが発生しました: {str(e)}")

def fetch_neo4j_data(file_name, keywords):
    document_type = file_name.replace(".txt", "")
    
    # キーワードリストからdocument_type（ファイル名）を取り除く
    filtered_keywords = [keyword for keyword in keywords if keyword != document_type]
    
    # クエリを動的に構築して複数のノードタイプを検索
    query = """
    MATCH (n)
    WHERE (n:Animal OR n:Event OR n:Item OR n:Location OR n:Organization OR n:Person OR n:Place OR n:Team)
    AND n.document_type = $document_type
    AND (
        """ + " OR ".join([f"""
        n.name CONTAINS $keyword{i} 
        OR n.characteristic CONTAINS $keyword{i} 
        OR any(k in keys(n) WHERE (
            (n[k] IS NOT NULL AND n[k] CONTAINS $keyword{i}) OR
            (any(x in n[k] WHERE toString(x) CONTAINS $keyword{i}))
        ))""" for i in range(len(filtered_keywords))]) + """
    )
    RETURN n
    """
    
    # パラメータのマッピング
    params = {"document_type": document_type}
    for i, keyword in enumerate(filtered_keywords):
        params[f"keyword{i}"] = keyword
    
    # クエリ実行
    result = graph.query(query, params=params)
    return result

# Function for multiplication
def multiply_fun(x, y):
    try:
        return x * y
    except Exception as e:
        return f"Error in multiplication: {str(e)}"

# Streamlitアプリの設定
st.title("RAG System Demo")

# セッション状態の初期化
if 'mode' not in st.session_state:
    st.session_state.mode = 'RAG'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'split_docs' not in st.session_state:
    st.session_state.split_docs = {}
if 'current_files' not in st.session_state:
    st.session_state.current_files = []
if 'metadata_df' not in st.session_state:
    st.session_state.metadata_df = pd.DataFrame(columns=[
        'Source', 'Title', 'Author', 'Publication_year', 'Genre', 'Setting_period',
        'Setting', 'Characters', 'Point_of_view', 'Style', 'Synopsis Display', 'Themes',
        'Target_audience'
    ])
if 'metadata_extracted_files' not in st.session_state:
    st.session_state.metadata_extracted_files = []

# サイドバー
with st.sidebar:
    st.header("Settings")

    # デフォルトAPIキー
    default_api_key = st.secrets["api_key"] # ここにデフォルトのAPIキーをセット

    # OpenAI APIキーの入力（セッション状態に保存）
    if 'api_key' not in st.session_state:
        st.session_state.api_key = default_api_key

    # APIキー入力欄のデフォルト値にセッションキーをセット
    api_key = st.text_input("Enter your OpenAI API key", type="password", value=st.session_state.api_key)
    os.environ["OPENAI_API_KEY"] = api_key

    # ユーザーが新しいAPIキーを入力した場合は更新
    if api_key and api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")

    # デフォルトidf
    default_idf_rate = 20.0
    
    # OpenAI APIキーの入力（セッション状態に保存）
    if 'idf_rate' not in st.session_state:
        st.session_state.idf_rate = default_idf_rate
    
    # APIキー入力欄のデフォルト値にセッションキーをセット
    idf_rate_input = st.text_input("Enter idf rate", value=str(st.session_state.idf_rate))
    
    # ユーザーが新しいAPIキーを入力した場合は更新
    try:
        idf_rate = float(idf_rate_input)  # 入力値を数値に変換
        if idf_rate != st.session_state.idf_rate:
            st.session_state.idf_rate = idf_rate  # 数値として保存
    except ValueError:
        st.error("Invalid input: Please enter a valid number for idf_rate")

    st.markdown("---")

    st.header("Score Balancing Settings")

    # ベクトルスコアとキーワードスコアのバランスを設定するスライダー
    alpha = st.slider("Set balance between vector and keyword scores (Vector weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    beta = 1 - alpha  # キーワードスコアのバランス（1 - ベクトルスコアの重み）

    st.markdown(f"**Vector Score Weight (alpha):** {alpha}")
    st.markdown(f"**Keyword Score Weight (beta):** {beta}")

    st.markdown("---")

    # チャンキング方式の選択
    chunking_method = st.radio(
        "Choose the chunking method:",
        ("Pre-chunked file", "Chunk by character count")
    )

    if chunking_method == "Chunk by character count":
        # チャンクサイズとオーバーラップをスライダーで指定
        chunk_size = st.slider("Chunk size (characters):", min_value=100, max_value=2000, value=1500)
        chunk_overlap = st.slider("Chunk overlap (characters):", min_value=0, max_value=1000, value=200)

    st.markdown("---")

    # ファイルアップロード
    uploaded_file = st.file_uploader("Upload a TXT file", type="txt")

    if uploaded_file is not None:
        if uploaded_file.name in st.session_state.current_files:
            st.warning(f"File '{uploaded_file.name}' has already been processed.")
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # チャンキング処理
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    text = load_file(tmp_file_path)
                    if text is not None:
                        doc = Document(page_content=text, metadata={"source": uploaded_file.name})

                        if chunking_method == "Chunk by character count":
                            # 文字数でチャンキング
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            new_split_docs = text_splitter.split_documents([doc])
                        else:
                            # 既にチャンキングされたテキストを処理
                            # ここで例のようなコードを用いて、各チャンクを処理
                            sentences = text.split("\n")
                            chunk_boundaries = [i for i, line in enumerate(sentences) if "Chunk" in line]
                            chunks = []
                            for i in range(len(chunk_boundaries)-1):
                                chunk = sentences[chunk_boundaries[i]:chunk_boundaries[i+1]]
                                chunks.append(" ".join(chunk))
                            new_split_docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name, "chunk_id": f"Chunk {i+1}"}) for i, chunk in enumerate(chunks)]

                        for i, doc in enumerate(new_split_docs):
                            doc.metadata['chunk_id'] = f"{uploaded_file.name}-Chunk{i+1}"

                        st.session_state.split_docs[uploaded_file.name] = new_split_docs
                        st.session_state.current_files.append(uploaded_file.name)

                        try:
                            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                            if st.session_state.vectorstore is None:
                                st.session_state.vectorstore = FAISS.from_documents(new_split_docs, embeddings)
                            else:
                                st.session_state.vectorstore.add_documents(new_split_docs)
                            st.session_state.document_processed = True
                            st.success("Document processed successfully!")
                        except Exception as e:
                            st.error(f"An error occurred while processing the document: {str(e)}")
                    else:
                        st.error("Failed to read the document. Please try again.")

    st.markdown("---")

    # ファイル選択用のセレクトボックス
    if st.session_state.current_files:
        selected_file = st.selectbox("Select a file for metadata extraction", st.session_state.current_files)
    else:
        selected_file = None

    # メタデータ抽出ボタン 
    if selected_file and st.button("Extract Metadata and Synopsis"):
        with st.spinner("Extracting metadata and synopsis..."):
            try:
                chat = ChatOpenAI(temperature=0, model="gpt-4o")
                current_metadata = {}
                previous_synopsis = ""  # 文字列として初期化
                for i, doc in enumerate(st.session_state.split_docs[selected_file]):
                    print(f"\nProcessing chunk {i+1}/{len(st.session_state.split_docs[selected_file])}:")
                    current_metadata = extract_metadata(doc.page_content, chat, current_metadata)
                    updated_synopsis = extract_synopsis(doc.page_content, chat, previous_synopsis, doc.metadata.get('chunk_id'))
                    if updated_synopsis:
                        previous_synopsis = updated_synopsis  # 累積あらすじを更新
                    current_metadata['Source'] = doc.metadata.get('source', '')

                # あらすじをメタデータに追加
                current_metadata['Synopsis'] = previous_synopsis[:2000]  # 2000文字以内に制限

                # キャラクター情報を適切にフォーマット
                if 'Characters' in current_metadata and isinstance(current_metadata['Characters'], list):
                    formatted_characters = []
                    for char in current_metadata['Characters']:
                        char_info = f"{char.get('Name', 'Unknown')} ({char.get('Role', 'Unknown')}"
                        if 'Age' in char and char['Age'] != 'Unknown':
                            char_info += f", Age: {char['Age']}"
                        char_info += ")"
                        if 'Description' in char and char['Description'] != 'Unknown':
                            char_info += f" - {char['Description']}"
                        formatted_characters.append(char_info)
                    current_metadata['Characters'] = '; '.join(formatted_characters)

                # あらすじをシンプルに表示用に設定
                current_metadata['Synopsis Display'] = current_metadata['Synopsis']

                # メタデータをDataFrameに追加
                st.session_state.metadata_df = pd.concat([st.session_state.metadata_df, pd.DataFrame([current_metadata])], ignore_index=True)
                st.session_state.metadata_extracted_files.append(selected_file)
                st.success("Metadata and synopsis extracted and updated successfully!")
            except Exception as e:
                st.error(f"An error occurred while extracting metadata: {str(e)}")

    st.markdown("---")

    # データのダウンロード
    if not st.session_state.metadata_df.empty or st.session_state.split_docs:
        if st.button("Download Metadata"):
            if not st.session_state.metadata_df.empty:
                # メタデータのダウンロード
                csv_metadata = st.session_state.metadata_df.to_csv(index=False)
                b64_metadata = base64.b64encode(csv_metadata.encode()).decode()
                href_metadata = f'<a href="data:file/csv;base64,{b64_metadata}" download="metadata.csv">Download Metadata CSV File</a>'
                st.markdown(href_metadata, unsafe_allow_html=True)
            else:
                st.info("No metadata available to download.")

        if st.session_state.split_docs:
            if st.button("Download Chunks"):
                # チャンクデータのダウンロード
                chunk_data = []
                for file, docs in st.session_state.split_docs.items():
                    for doc in docs:
                        chunk_data.append({
                            "Source": doc.metadata['source'],
                            "Chunk ID": doc.metadata['chunk_id'],
                            "Content": doc.page_content
                        })
                df_chunks = pd.DataFrame(chunk_data)
                csv_chunks = df_chunks.to_csv(index=False)
                b64_chunks = base64.b64encode(csv_chunks.encode()).decode()
                href_chunks = f'<a href="data:file/csv;base64,{b64_chunks}" download="chunks.csv">Download Chunks CSV File</a>'
                st.markdown(href_chunks, unsafe_allow_html=True)

        # ベクトルデータのダウンロードボタンの追加
        if st.session_state.vectorstore is not None:
            if st.button("Download Vector Data"):
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # ベクトルストアを一時ディレクトリに保存
                        st.session_state.vectorstore.save_local(tmp_dir)
                        # ZIPファイルに圧縮
                        zip_path = os.path.join(tmp_dir, "vectorstore.zip")
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for root, dirs, files in os.walk(tmp_dir):
                                for file in files:
                                    if file != "vectorstore.zip":
                                        file_path = os.path.join(root, file)
                                        zipf.write(file_path, arcname=file)
                        # ZIPファイルを読み込み
                        with open(zip_path, 'rb') as f:
                            zip_bytes = f.read()
                        b64_zip = base64.b64encode(zip_bytes).decode()
                        href_zip = f'<a href="data:application/zip;base64,{b64_zip}" download="vectorstore.zip">Download Vector Data ZIP File</a>'
                        st.markdown(href_zip, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred while downloading vector data: {str(e)}")
    else:
        st.info("No data available to download.")

    st.markdown("---")

    # データのアップロード
    uploaded_metadata = st.file_uploader("Upload Metadata CSV", type="csv", key="upload_metadata")
    uploaded_chunks = st.file_uploader("Upload Chunks CSV", type="csv", key="upload_chunks")
    uploaded_vector = st.file_uploader("Upload Vector Data (ZIP)", type="zip", key="upload_vector")  # ベクトルデータのアップロード追加

    if (uploaded_metadata is not None and uploaded_chunks is not None) or uploaded_vector is not None:
        if st.button("Load Uploaded Data"):
            try:
                # メタデータの読み込み
                if uploaded_metadata is not None:
                    metadata_df = pd.read_csv(uploaded_metadata)
                    st.session_state.metadata_df = pd.concat([st.session_state.metadata_df, metadata_df], ignore_index=True)

                # チャンクデータの読み込み
                if uploaded_chunks is not None:
                    chunks_df = pd.read_csv(uploaded_chunks)
                    split_docs = {}
                    current_files = chunks_df['Source'].unique().tolist()
                    for file in current_files:
                        file_chunks = chunks_df[chunks_df['Source'] == file]
                        docs = [
                            Document(page_content=row['Content'], metadata={"source": row['Source'], "chunk_id": row['Chunk ID']})
                            for _, row in file_chunks.iterrows()
                        ]
                        split_docs[file] = docs
                    st.session_state.split_docs.update(split_docs)
                    st.session_state.current_files = list(set(st.session_state.current_files + current_files))

                # ベクトルデータの読み込み
                if uploaded_vector is not None:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        vector_zip_path = os.path.join(tmp_dir, "uploaded_vectorstore.zip")
                        with open(vector_zip_path, "wb") as f:
                            f.write(uploaded_vector.getvalue())
                        with zipfile.ZipFile(vector_zip_path, 'r') as zip_ref:
                            zip_ref.extractall(tmp_dir)
                        # ベクトルストアをロード
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 同じモデルを使用
                        uploaded_vectorstore = FAISS.load_local(tmp_dir, embeddings, allow_dangerous_deserialization=True)
                        if st.session_state.vectorstore is None:
                            st.session_state.vectorstore = uploaded_vectorstore
                        else:
                            # FAISSのマージメソッドが存在しない場合、手動でマージ
                            st.session_state.vectorstore.merge(uploaded_vectorstore)
                # ベクトルストアの再構築（必要に応じて）
                if st.session_state.vectorstore is not None and (uploaded_chunks is not None or uploaded_vector is not None):
                    if uploaded_vector is None:
                        # 新しく追加されたチャンクをベクトルストアに追加
                        new_docs = []
                        for file in current_files:
                            new_docs.extend(st.session_state.split_docs[file])
                        st.session_state.vectorstore.add_documents(new_docs)

                # データのロードが成功したらフラグを設定
                st.session_state.data_loaded = True
                st.session_state.document_processed = True  # 追加: メインページの表示条件に対応
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred while loading the data: {str(e)}")

# メインページ
# モード切り替えボタン
mode = st.radio("Select mode:", ('RAG', 'Simple Chat'))
st.session_state.mode = mode

if st.session_state.mode == 'RAG' and (st.session_state.document_processed or st.session_state.data_loaded):
    # 2列レイアウトの作成
    col1, col2 = st.columns(2)

    # 左側のカラム（チャット画面）
    with col1:
        st.header("RAG Chat")
        # クエリ入力
        query = st.text_input("Enter your question:")
        if st.button("Ask (RAG)"):
            with st.spinner("Generating answer..."):
                try:
                    functions = [
                        {
                            "name": "extract_keywords",
                            "description": "キーワード検索を行うための単語をリスト化します。",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "抽出された単語のリスト"}
                                },
                                "required": ["keywords"]
                            }
                        }
                    ]

                    # ファイル名の判定
                    specified_files = []
                    current_files = st.session_state.current_files  # 現在のファイルリスト
                    # st.write(f"Current Files: {', '.join(current_files)}")

                    # ファイル名から拡張子を除いた名前を使用してクエリに含まれているかを確認
                    for file_name in current_files:
                        file_name_without_extension = file_name.replace(".txt", "")  # 拡張子を除去
                        # ファイル名（拡張子なし）がクエリに含まれているか確認
                        if file_name_without_extension in query:
                            specified_files.append(file_name)

                    # クエリから固有名詞・特徴的な単語を抽出
                    extraction_prompt = (
                        f"#質問\n{query}"
                    )
                    llm = ChatOpenAI(
                            temperature=0,
                            model="gpt-4o",
                            functions=functions,
                            function_call={"name": "extract_keywords"}
                        )
                    extraction_response = llm([
                        {"role": "system", "content": """小説を見分けることが目的です。次の質問から固有名詞または特徴的な単語を、複合語に注意して抽出してください。
                         表現のゆれ（活用）で完全一致が難しいキーワードは分割します。「...」など記号も除きます。小説特徴付ける単語以外は除きます。
                         例えば、「研究グループ」は意味の独立した単語の複合語なので「研究」と「グループ」に分けますが、「鶴の一声」のような慣用句については分けると意味が損なわれるので「鶴の一声」のままとします。
                         また、「メンバー」「科」などの一般的な名詞が含まれている場合は、他の名詞と分割します。"""},
                        {"role": "user", "content": extraction_prompt}
                    ])

                    # GPTによる抽出結果を処理
                    extracted_keywords = []
                    if extraction_response.additional_kwargs.get("function_call"):
                        function_call = extraction_response.additional_kwargs["function_call"]
                        arguments = json.loads(function_call["arguments"])
                        extracted_keywords = arguments.get("keywords", [])

                    # st.write(f"抽出されたキーワード: {extracted_keywords}")
                    filtered_keywords = [kw for kw in extracted_keywords if kw not in [file_name_without_extension for file_name in current_files]]

                    functions = [
                        {
                            "name": "get_short_answer_with_quote",
                            "description": "クエリに対する短い回答を作成し、ドキュメントから引用文を提供します。",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string", "description": "クエリに対する簡潔な回答。"},
                                    "quote": {"type": "string", "description": "ドキュメントからの引用文。"}
                                },
                                "required": ["answer", "quote"]
                            }
                        },
                        {
                            "name": "multiply",
                            "description": "計算問題に解答します。質問から計算方法を判別し、ドキュメントから変数と引用文を提供します。",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "string", "description": "最初の変数 (x)."},
                                    "y": {"type": "string", "description": "2つ目の変数 (y)."},
                                    "quote": {"type": "string", "description": "ドキュメントからの引用文。"}
                                },
                                "required": ["x", "y", "quote"]
                            }
                        }
                    ]
                    
                    # ChatOpenAIインスタンスの作成（Function Callingを有効にする）
                    llm = ChatOpenAI(
                        temperature=0,
                        model="gpt-4o",
                        functions=functions,
                        function_call="auto"
                    )

                    # 指定ファイルがある場合
                    if specified_files:
                        st.write(f"クエリに含まれていた指定ファイル: {', '.join(specified_files)}")
                        selected_file = specified_files[0]  # 最初の一致したファイルを使用
                        st.write(f"選択されたファイル: {selected_file}")

                        # 選択されたファイルのチャンクを取得
                        matched_chunks = st.session_state.split_docs.get(selected_file, [])

                        # キーワードヒットのカウント
                        keyword_hits = {keyword: 0 for keyword in extracted_keywords}

                        for doc in matched_chunks:
                            doc_content_normalized = normalize_text(doc.page_content)
                            for keyword in extracted_keywords:
                                keyword_normalized = normalize_text(keyword)
                                keyword_count = doc_content_normalized.count(keyword_normalized)
                                if keyword_count > 0:
                                    keyword_hits[keyword] += keyword_count

                        st.write(f"キーワードのヒット回数: {keyword_hits}")
                        keyword_idf = {}

                    else:
                        st.write("ファイル名がクエリに含まれていません。")

                        # 抽出したキーワードがある場合、キーワード検索を行う
                        if filtered_keywords:
                            # キーワードが含まれるファイルを完全一致で検索し、ヒット回数をカウント
                            keyword_hits = {}  # 各ファイルごとのキーワードヒット数を記録
                            matched_chunks = {}  # 各ファイルごとのヒットしたチャンクを記録

                            # キーワードが含まれるファイルを完全一致で検索
                            keyword_query = " ".join(extracted_keywords)
                            st.write(f"検索キーワード: {keyword_query}")

                            total_documents = len(st.session_state.split_docs)  # 総ドキュメント数

                            # ドキュメント頻度（DF）の計算
                            keyword_document_frequency = {keyword: 0 for keyword in extracted_keywords}

                            for file, docs in st.session_state.split_docs.items():
                                file_keyword_hits = {keyword: 0 for keyword in extracted_keywords}
                                matched_docs = []
                                for doc in docs:
                                    # ドキュメントテキストを正規化
                                    doc_content_normalized = normalize_text(doc.page_content)
                                    keyword_found_in_doc = False
                                    for keyword in extracted_keywords:
                                        # キーワードを正規化
                                        keyword_normalized = normalize_text(keyword)
                                        # キーワードの出現回数をカウント
                                        keyword_count = doc_content_normalized.count(keyword_normalized)
                                        if keyword_count > 0:
                                            file_keyword_hits[keyword] += keyword_count
                                            keyword_found_in_doc = True
                                    if keyword_found_in_doc:
                                        matched_docs.append(doc)
                                # ファイル内でキーワードが見つかった場合
                                if any(count > 0 for count in file_keyword_hits.values()):
                                    keyword_hits[file] = file_keyword_hits
                                    matched_chunks[file] = matched_docs
                                    for keyword in extracted_keywords:
                                        if file_keyword_hits[keyword] > 0:
                                            keyword_document_frequency[keyword] += 1

                            # IDFの計算
                            keyword_idf = {}
                            for keyword in extracted_keywords:
                                df = keyword_document_frequency[keyword]
                                if df > 0:
                                    keyword_idf[keyword] = math.log(total_documents / df)
                                else:
                                    keyword_idf[keyword] = 0.0

                            # 各ファイルのTF-IDFスコアを計算
                            file_scores = {}
                            for file, hits in keyword_hits.items():
                                tf_idf_score = 0.0
                                found_keywords_count = 0  # 登場したキーワード数のカウント
                                for keyword in extracted_keywords:
                                    tf = 1 + math.log(hits[keyword]) if hits[keyword] > 0 else 0
                                    idf_weight = 1.5  # IDFの影響を強める場合
                                    idf = idf_weight * math.log(total_documents / (1 + df))
                                    tf_idf_score += tf * idf
                                    if hits[keyword] > 0:
                                        found_keywords_count += 1  # キーワードが登場したらカウントアップ

                                # キーワードが登場する数に応じてボーナスを与える
                                coverage_bonus = found_keywords_count / len(extracted_keywords)
                                tf_idf_score += coverage_bonus * idf_rate  # ボーナスの強さは調整可能

                                file_scores[file] = tf_idf_score

                            # スコアの高い順にファイルをソート
                            sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

                            # 最もスコアの高いファイルを選択
                            if sorted_files:
                                selected_file = sorted_files[0][0]
                                st.write(f"TF-IDFスコアに基づいて選択されたファイル: {selected_file}")
                                st.write(f"ファイルごとのTF-IDFスコア: {file_scores}")

                                # 選択されたファイルのヒットしたチャンクを取得
                                matched_chunks = matched_chunks[selected_file]
                                keyword_hits = keyword_hits[selected_file]
                            else:
                                st.error("キーワードに一致するファイルが見つかりませんでした。")
                                continue_processing = False
                        else:
                            st.error("キーワードが抽出できなかったため、検索を実行できませんでした。")
                            keyword_idf = {}
                            continue_processing = False

                    # メタデータベースの回答生成
                    if 'selected_file' in locals():
                        neo4j_data = fetch_neo4j_data(selected_file, extracted_keywords)
                        st.write("Neo4jから取得したデータ:", neo4j_data)
                        
                        messages = [
                                {"role": "system", "content": """
                                質問を入念に読み解きます。グラフDBの情報から、質問に関連する情報を抜き出して整理します。
                                なぜ、関連していると言えるのかも記載します。"""},
                                {"role": "user", "content": f"質問: {query}\n\n#グラフDB:{neo4j_data}"}
                            ]

                        llm = ChatOpenAI(
                            temperature=0,
                            model="gpt-4o"
                        )
                        neo4j_data = llm(messages)
                        neo4j_data = neo4j_data.content
                        st.write(neo4j_data)

                        metadata_response_json = metadata_based_gpt_response(query, st.session_state.metadata_df, selected_file, llm)

                        st.subheader("Metadata-based Answer:")
                        st.write(metadata_response_json["Metadata_based_Answer"])

                        # メタデータからの回答が不十分な場合、RAGでの回答処理を行う
                        if metadata_response_json["Sufficiency"] == "不十分":
                            st.write("メタデータからの回答が不十分でした。RAGを実行して詳細な情報を検索します。")

                            # キーワードが抽出できた場合、キーワード検索と類似度検索を行う
                            if filtered_keywords and any(keyword_hits.values()):
                                # キーワード検索でヒットしたチャンクを検索対象にして、類似度検索を実行
                                st.write("キーワード検索でヒットしたチャンクを検索対象にして、類似度検索を実行します。")

                                # クエリのベクトルを取得
                                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                                query_vector = embeddings.embed_query(query)

                                # ヒットしたチャンクのベクトルとキーワードTF-IDFスコアを計算
                                chunk_vectors = []
                                keyword_tf_idf_scores = []
                                for doc in matched_chunks:
                                    # チャンクのベクトルを計算
                                    chunk_vector = embeddings.embed_query(doc.page_content)
                                    chunk_vectors.append(chunk_vector)

                                    # チャンク内のキーワード出現頻度（TF）を計算
                                    doc_content = normalize_text(doc.page_content)
                                    tf_scores = {}
                                    for keyword in extracted_keywords:
                                        keyword_normalized = normalize_text(keyword)
                                        count = doc_content.count(keyword_normalized)
                                        if count > 0:
                                            tf_scores[keyword] = 1 + math.log(count)
                                        else:
                                            tf_scores[keyword] = 0.0

                                    # IDFを計算（既に計算済みのkeyword_idfを使用）
                                    tf_idf = 0.0
                                    for keyword, tf in tf_scores.items():
                                        idf = keyword_idf.get(keyword, 0.0)
                                        tf_idf += tf * idf

                                    keyword_tf_idf_scores.append(tf_idf)

                                # ベクトル類似度スコアを計算
                                vector_scores = [np.dot(query_vector, chunk_vector) for chunk_vector in chunk_vectors]

                                # スコアの正規化
                                epsilon = 1e-8  # ゼロ除算を防ぐための小さな値
                                vector_scores = np.array(vector_scores)
                                keyword_tf_idf_scores = np.array(keyword_tf_idf_scores)

                                # ベクトルスコアの正規化（0から1の範囲）
                                vector_scores_norm = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + epsilon)
                                # キーワードTF-IDFスコアの正規化（0から1の範囲）
                                keyword_scores_norm = (keyword_tf_idf_scores - keyword_tf_idf_scores.min()) / (keyword_tf_idf_scores.max() - keyword_tf_idf_scores.min() + epsilon)

                                # ベクトルスコアとキーワードTF-IDFスコアを組み合わせて総合スコアを計算
                                combined_scores = alpha * vector_scores_norm + beta * keyword_scores_norm

                                # チャンクと総合スコアを組み合わせる
                                chunk_scores = list(zip(matched_chunks, combined_scores))

                                # 総合スコアに基づいてチャンクをソート
                                sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)  # 降順にソート

                                # 上位のチャンクを取得
                                docs = [doc for doc, score in sorted_chunks][:4]
                                st.write(docs)

                            # キーワードが抽出されなかった場合、ベクトル検索のみ実行
                            else:
                                st.write("キーワードが抽出されなかったため、通常のベクトル検索結果を使用します。")
                                retrieved_docs = st.session_state.vectorstore.similarity_search(query, k=10)
                                docs = [doc for doc in retrieved_docs][:3]

                            # 回答の生成（Function Calling）
                            context = "\n".join([doc.page_content for doc in docs])

                            # キーワードとその登場回数を含めた文字列
                            keyword_hits_str = "\n".join([f"{keyword}: {count}" for keyword, count in keyword_hits.items()])

                            # 10回以下のキーワードを抽出
                            low_frequency_keywords = {keyword: count for keyword, count in keyword_hits.items() if count <= 20}

                            # 10回以下のキーワードを別の文字列として出力
                            low_frequency_keywords_str = "\n".join([f"{keyword}: {count}" for keyword, count in low_frequency_keywords.items()])

                            # 必要に応じて、出力部分に追加
                            st.write(f"10回以下のキーワード:\n{low_frequency_keywords_str}")

                            # 辞書を使用してキーワードごとの文を格納
                            low_freq_keyword_sentences = {}

                            for keyword in low_frequency_keywords:
                                sentences_set = set()  # 重複を排除するためにセットを使用
                                keyword_normalized = normalize_text(keyword)
                                # matched_chunksがリストの場合、ファイルごとに分ける処理が不要なので、単純なループに変更
                                for doc in matched_chunks:
                                    # 文単位で分割
                                    sentences = doc.page_content.split('。')
                                    for sentence in sentences:
                                        sentence_normalized = normalize_text(sentence)
                                        if keyword_normalized in sentence_normalized:
                                            sentences_set.add(sentence.strip())  # 前後の空白を削除して追加
                                low_freq_keyword_sentences[keyword] = list(sentences_set)  # セットをリストに変換

                            # contextを文ごとに分割してリストに格納
                            context_sentences = context.split('。')

                            # 低頻度キーワードとその文の出力
                            low_freq_keyword_sentences_str = ""

                            for keyword, sentences in low_freq_keyword_sentences.items():
                                # 各文がcontextに含まれていない場合のみ追加
                                for sentence in sentences:
                                    if sentence not in context_sentences:  # contextに含まれている文を除外
                                        low_freq_keyword_sentences_str += f"- {sentence}。\n"
                                
                                low_freq_keyword_sentences_str += "\n"

                            # 出力部分に追加
                            st.write(f"10回以下のキーワードに含まれる文（contextに含まれないもの）:\n{low_freq_keyword_sentences_str}")
                            
                            # keyword_hitsが定義されていて、かつ空でない場合のみ処理する
                            if 'keyword_hits' in locals() and keyword_hits is not None and any(keyword_hits.values()):
                                keyword_hits_str = "\n".join([f"{keyword}: {count}" for keyword, count in keyword_hits.items()])
                                # メッセージにキーワードとその登場回数を含める
                                messages = [
                                    {"role": "system", "content": """
                                    ①「総額」や「いくら？」などの計算に関する質問の場合、自分で計算を行わなわず、必ずmultiply関数を呼び出してください。
                                    ②その他の質問の場合、get_short_answer_with_quote関数を呼び出し、回答を作成し、その根拠を引用してください。必ず、引用は”本文”から文章を"そのまま"抜き出します。
                                    - 回答は、質問に対して厳密（”～以上”などの細かい表現に注意する）かつ簡潔（50字以内）にします。
                                    - 文書全体を調べた結果判明したグラフDBを参考にし、その内容を本文と合わせて理解します。
                                    - 論理的に明らかに質問の前提に誤りがあると判断される場合は「質問誤り」と答えます。"""},
                                    {"role": "user", "content": f"質問: {query}\n\n#本文:\n{context}\n\n{low_freq_keyword_sentences_str}\n\n#グラフDB（参考）:\n{neo4j_data}"}
                                ]
                            else:  # keyword_hitsが存在しないか、空の場合
                                # メッセージにキーワードとその登場回数を含めない
                                messages = [
                                    {"role": "system", "content": """
                                     質問に対して厳密（”～以上”などの細かい表現に注意する）かつ簡潔な回答（50字以内）を作成し、その根拠を引用してください。
                                     文脈から考えて質問の前提に誤りがあると判断される場合のみ「質問誤り」と答えます。"""},
                                    {"role": "user", "content": f"質問: {query}\n文脈: {context}"}
                                ]

                            llm = ChatOpenAI(
                                temperature=0,
                                model="gpt-4o",
                                functions=functions,
                                function_call="auto"
                            )
                            response = llm(messages)
                            st.write(messages)
                            st.write(response)

                            # Check for function call response
                            if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                                function_call = response.additional_kwargs['function_call']
                                if function_call['name'] == "multiply":
                                    arguments = json.loads(function_call['arguments'])
                                    x_args = arguments.get('x')
                                    y_args = arguments.get('y')
                                    quote_args = arguments.get('quote')

                                    # 数値に変換する部分を追加
                                    try:
                                        x_value = float(x_args)
                                        y_value = float(y_args)
                                    except ValueError:
                                        z_multiply = f"Invalid input: x={x_args}, y={y_args} must be numbers"
                                    else:
                                        z_multiply = multiply_fun(x_value, y_value)  # Call the multiply function

                                    # Create a new message with the calculation result
                                    follow_up_message = [
                                        {"role": "system", "content": "あなたは数学者です。慎重に質問を読み解いて、変数を用いて計算します。質問に対して簡潔な回答を作成してください。計算結果を元に根拠を示してください。"},
                                        {"role": "user", "content": f"質問: {query}\n\n変数: {x_value}, {y_value}\n\n#文脈:\n{quote_args}"}
                                    ]
                                    st.write(follow_up_message)

                                    llm = ChatOpenAI(
                                        temperature=0,
                                        model="gpt-4o",
                                        functions=functions,
                                        function_call={"name": "get_short_answer_with_quote"}
                                    )

                                    # Generate the final response
                                    response = llm(follow_up_message)
                                    st.write(response)

                            # 回答と引用文の取得
                            if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                                function_call = response.additional_kwargs['function_call']
                                arguments = json.loads(function_call['arguments'])
                                short_answer = arguments.get('answer', '分かりません')
                                supporting_quote = arguments.get('quote', '分かりません')
                                
                                # contextだけでなく、low_freq_keyword_sentencesにも含まれているかどうかをチェック
                                context_sentences = context.split('。')
                                in_context = supporting_quote in context_sentences
                                
                                in_low_freq_keywords = any(supporting_quote in sentences for sentences in low_freq_keyword_sentences.values())
                                
                                if not in_context and not in_low_freq_keywords:
                                    st.write(supporting_quote)
                                    supporting_quote = "分かりません"
                                
                                st.write(supporting_quote)
                                
                            else:
                                short_answer = response.content
                                supporting_quote = "分かりません"
                                st.write(supporting_quote)

                            # 関連チャンクの番号と先頭100文字を表示
                            st.subheader("Related Chunks:")
                            for doc in docs:
                                chunk_id = doc.metadata.get('chunk_id', '不明')
                                chunk_content = doc.page_content[:100]
                                st.write(f"**Chunk ID:** {chunk_id}")
                                st.write(f"**Content:** {chunk_content}...")
                                st.markdown("---")

                            # 回答表示
                            st.subheader("RAG-based Answer:")
                            st.write(short_answer)
                            st.subheader("Supporting Quote:")
                            st.write(supporting_quote)

                    else:
                        st.error("指定されたファイルが見つかりませんでした。")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # 右側のカラム（メタデータ、チャンク確認）
    with col2:
        st.header("Document Metadata")

        if not st.session_state.metadata_df.empty:
            # Synopsis Displayカラムを追加して表示
            if 'Synopsis Display' in st.session_state.metadata_df.columns:
                st.session_state.metadata_df['Synopsis Display'] = st.session_state.metadata_df['Synopsis Display'].apply(lambda x: x if isinstance(x, str) else '')
            else:
                st.session_state.metadata_df['Synopsis Display'] = ''

            # Charactersカラムを整形
            if 'Characters' in st.session_state.metadata_df.columns:
                st.session_state.metadata_df['Characters'] = st.session_state.metadata_df['Characters'].apply(
                    lambda x: format_characters(x) if isinstance(x, list) else str(x)
                )

            # 不要なSynopsisカラムの表示を避けるため、必要に応じて削除または非表示に設定
            display_df = st.session_state.metadata_df.copy()
            if 'Synopsis' in display_df.columns:
                display_df = display_df.drop(columns=['Synopsis'])
            display_df = display_df.rename(columns={'Synopsis Display': 'Synopsis'})

            # すべてのカラムを文字列型に変換
            display_df = display_df.astype(str)

            # データフレームを表示
            st.dataframe(display_df)
        else:
            st.info("No metadata available.")

        st.header("Document Chunks")
        st.write(f"Current Files: {', '.join(st.session_state.current_files)}")

        if st.session_state.split_docs:
            # チャンクデータをDataFrameに変換
            chunk_data = []
            for file, docs in st.session_state.split_docs.items():
                for doc in docs:
                    chunk_data.append({
                        "Source": doc.metadata['source'],
                        "Chunk ID": doc.metadata['chunk_id'],
                        "Content": doc.page_content[:100] + "..."  # 最初の100文字を表示
                    })
            df = pd.DataFrame(chunk_data)

            # データが存在する場合のみ表示
            if not df.empty:
                st.dataframe(df)
            else:
                st.info("No document chunks available.")
        else:
            st.info("No document chunks available.")

elif st.session_state.mode == 'RAG' and not st.session_state.document_processed:
    st.info("Please upload and process a document first to use RAG mode.")

else:  # Simple Chat mode
    st.header("Simple Chat")
    query = st.text_input("Enter your question:")
    if st.button("Ask (Simple)"):
        with st.spinner("Generating answer..."):
            try:
                llm = OpenAI(temperature=0, model="gpt-4o")
                response = llm(query)
                st.subheader("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {str(e)}")

# 一時ファイルの削除
if 'tmp_file_path' in locals():
    os.unlink(tmp_file_path)