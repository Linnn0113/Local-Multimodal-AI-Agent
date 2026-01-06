import argparse
import os
import chromadb
from PIL import Image
from model_loader import EmbeddingModel
from utils import extract_text_with_page_numbers, move_file_to_category
import numpy as np

# --- 初始化 ---
client = chromadb.PersistentClient(path="./db")
paper_collection = client.get_or_create_collection(name="papers")
image_collection = client.get_or_create_collection(name="images")
model_handler = EmbeddingModel()

# --- 核心功能函数 ---

def add_paper(file_path, topics=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Processing {file_path}...")
    
    # 1. 按页提取文本 (获取 chunks)
    chunks = extract_text_with_page_numbers(file_path)
    if not chunks:
        print("No readable text found in PDF.")
        return

    # 2. 确定分类 (用前3页的内容来做整体分类)
    summary_text = " ".join([c["text"] for c in chunks[:3]])
    summary_embedding = model_handler.get_text_embedding(summary_text)

    best_topic = "Uncategorized"
    final_path = file_path

    if topics:
        topic_list = topics.split(',')
        topic_embeddings = model_handler.text_model.encode(topic_list)
        
        # 计算相似度
        similarities = np.dot(topic_embeddings, summary_embedding) / (
            np.linalg.norm(topic_embeddings, axis=1) * np.linalg.norm(summary_embedding)
        )
        best_topic = topic_list[np.argmax(similarities)]
        print(f"Detected Topic: {best_topic}")
        
        # 移动文件
        final_path = move_file_to_category(file_path, best_topic)

    # 3. 存入数据库
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    filename = os.path.basename(final_path)

    print(f"Indexing {len(chunks)} pages...")
    for chunk in chunks:
        page_id = f"{filename}_p{chunk['page']}"
        
        ids.append(page_id)
        documents.append(chunk["text"]) 
        embeddings.append(model_handler.get_text_embedding(chunk["text"]))
        
        metadatas.append({
            "path": final_path,
            "topic": best_topic,
            "page": chunk["page"]
        })

    paper_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print("Paper indexed successfully.")

def search_paper(query):
    print(f"Searching for: {query}")
    query_vec = model_handler.get_text_embedding(query)
    
    results = paper_collection.query(
        query_embeddings=[query_vec],
        n_results=3 
    )
    
    print("\n" + "="*50)
    print(f" Search Results for: '{query}'")
    print("="*50)
    
    if not results['ids'][0]:
        print("No results found.")
        return

    for i, doc_id in enumerate(results['ids'][0]):
        meta = results['metadatas'][0][i]
        snippet = results['documents'][0][i]
        
        print(f"\n📄 Result {i+1}")
        print(f"• File: {os.path.basename(meta['path'])}")
        print(f"• Page: {meta['page']}")
        print(f"• Topic: {meta['topic']}")
        print("-" * 30)
        preview = snippet.replace('\n', ' ')[:300] 
        print(f"💡 Snippet: \"...{preview}...\"")
        print("-" * 30)

def add_image(image_dir):
    """批量索引目录下的图片"""
    valid_exts = ['.jpg', '.jpeg', '.png']
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_exts):
                full_path = os.path.join(root, file)
                try:
                    img = Image.open(full_path)
                    vec = model_handler.get_image_embedding(img)
                    image_collection.upsert(
                        ids=[file],
                        embeddings=[vec],
                        metadatas=[{"path": full_path}]
                    )
                    print(f"Indexed image: {file}")
                except Exception as e:
                    print(f"Failed to index {file}: {e}")

def search_image(query):
    """以文搜图"""
    print(f"Searching image for: {query}")
    query_vec = model_handler.get_text_for_image_embedding(query)
    
    results = image_collection.query(
        query_embeddings=[query_vec],
        n_results=3
    )
    
    print("\n--- Image Results ---")
    if not results['ids'][0]:
        print("No images found.")
        return

    for i, doc_id in enumerate(results['ids'][0]):
        meta = results['metadatas'][0][i]
        print(f"[{i+1}] {doc_id} (Path: {meta['path']})")

# --- CLI 入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent")
    subparsers = parser.add_subparsers(dest="command")

    # Command: add_paper
    parser_add = subparsers.add_parser("add_paper", help="Add and classify a paper")
    parser_add.add_argument("path", type=str, help="Path to the PDF file")
    parser_add.add_argument("--topics", type=str, help="Comma separated topics")

    # Command: search_paper
    parser_search = subparsers.add_parser("search_paper", help="Semantic search for papers")
    parser_search.add_argument("query", type=str, help="Search query")

    # Command: index_images
    parser_idx_img = subparsers.add_parser("index_images", help="Index a folder of images")
    parser_idx_img.add_argument("path", type=str, help="Folder path containing images")

    # Command: search_image
    parser_img_search = subparsers.add_parser("search_image", help="Text-to-Image search")
    parser_img_search.add_argument("query", type=str, help="Description of the image")

    args = parser.parse_args()

    if args.command == "add_paper":
        add_paper(args.path, args.topics)
    elif args.command == "search_paper":
        search_paper(args.query)
    elif args.command == "index_images":
        add_image(args.path)
    elif args.command == "search_image":
        search_image(args.query)
    else:
        parser.print_help()