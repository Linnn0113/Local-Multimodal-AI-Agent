import gradio as gr
import os
import chromadb
from PIL import Image
import numpy as np
import shutil

# 导入你现有的后端模块
# 确保目录下有 model_loader.py 和 utils.py
from model_loader import EmbeddingModel
from utils import extract_text_with_page_numbers, move_file_to_category

# --- 全局资源加载 ---
print("正在初始化模型和数据库...")
try:
    # 加载模型
    model_handler = EmbeddingModel()
    
    # 连接数据库
    client = chromadb.PersistentClient(path="./db")
    paper_collection = client.get_or_create_collection(name="papers")
    image_collection = client.get_or_create_collection(name="images")
    print("模型与数据库加载完毕！")
except Exception as e:
    print(f"初始化失败: {e}")

# --- 功能函数定义 ---

def process_upload(file_obj, topics_str):
    """处理论文上传"""
    if file_obj is None:
        return "请先上传文件"
    
    # Gradio 传入的 file_obj.name 就是临时文件路径
    # 兼容不同版本的 Gradio (3.x vs 4.x)
    tmp_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    
    # 1. 提取文本
    chunks = extract_text_with_page_numbers(tmp_path)
    if not chunks:
        return "无法提取文本，请检查 PDF 文件。"
    
    # 2. 确定分类
    # 取前3页做分类
    summary_text = " ".join([c["text"] for c in chunks[:3]])
    summary_vec = model_handler.get_text_embedding(summary_text)

    best_topic = "Uncategorized"
    if topics_str:
        t_list = topics_str.split(',')
        t_vecs = model_handler.text_model.encode(t_list)
        # 计算相似度
        scores = [np.dot(t_v, summary_vec) for t_v in t_vecs]
        best_topic = t_list[np.argmax(scores)]

    # 3. 保存文件
    target_dir = os.path.join("data", best_topic)
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取原始文件名 (Gradio 会重命名临时文件，我们尽量还原)
    original_name = os.path.basename(tmp_path)
    if hasattr(file_obj, 'orig_name'): # 某些 Gradio 版本
        original_name = file_obj.orig_name
        
    final_path = os.path.join(target_dir, original_name)
    shutil.copy(tmp_path, final_path)

    # 4. 存入数据库
    ids, docs, vecs, metas = [], [], [], []
    for chunk in chunks:
        page_id = f"{original_name}_p{chunk['page']}"
        ids.append(page_id)
        docs.append(chunk["text"])
        vecs.append(model_handler.get_text_embedding(chunk["text"]))
        metas.append({
            "path": final_path,
            "topic": best_topic,
            "page": chunk["page"]
        })

    paper_collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
    
    return f"✅ 成功！归类为: {best_topic}\n已索引 {len(chunks)} 页。\n保存路径: {final_path}"

def search_docs(query, top_k):
    """语义搜索"""
    if not query: return "请输入问题"
    
    query_vec = model_handler.get_text_embedding(query)
    results = paper_collection.query(query_embeddings=[query_vec], n_results=int(top_k))

    if not results['ids'] or not results['ids'][0]:
        return "未找到相关结果"
    
    output = ""
    for i, _ in enumerate(results['ids'][0]):
        meta = results['metadatas'][0][i]
        snippet = results['documents'][0][i]
        path = meta.get('path', 'Unknown')
        topic = meta.get('topic', 'N/A')
        page = meta.get('page', 'N/A')
        
        output += f"### 📄 结果 {i+1}: {os.path.basename(path)}\n"
        output += f"**Topic**: {topic} | **Page**: {page}\n\n"
        output += f"> ...{snippet[:300]}...\n"
        output += "---\n"
    return output

def index_local_images():
    """索引 data 目录图片"""
    image_dir = "data"
    valid_exts = ['.jpg', '.jpeg', '.png']
    count = 0
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_exts):
                full_path = os.path.join(root, file)
                try:
                    img = Image.open(full_path)
                    vec = model_handler.get_image_embedding(img)
                    image_collection.upsert(ids=[file], embeddings=[vec], metadatas=[{"path": full_path}])
                    count += 1
                except: pass
    return f"✅ 重建索引完成！共索引 {count} 张图片。"

def search_imgs(query):
    """以文搜图"""
    if not query: return []
    
    query_vec = model_handler.get_text_for_image_embedding(query)
    results = image_collection.query(query_embeddings=[query_vec], n_results=4)
    
    images = []
    if results['ids'] and results['ids'][0]:
        for i, _ in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            img_path = meta['path']
            if os.path.exists(img_path):
                # Gradio Gallery 接受 (image_path, caption) 的元组列表
                images.append((img_path, f"Result {i+1}"))
    return images

# --- 构建 UI ---
with gr.Blocks(title="多模态 AI 助手") as demo:
    gr.Markdown("# 🤖 本地多模态 AI 智能助手")
    
    with gr.Tab("📄 论文上传与分类"):
        gr.Markdown("上传 PDF 文件，系统将自动分析语义并归档。")
        with gr.Row():
            file_input = gr.File(label="上传 PDF", file_types=[".pdf"])
            topics_input = gr.Textbox(label="分类主题 (逗号分隔)", value="CV,NLP,Agent,RL")
        upload_btn = gr.Button("开始处理", variant="primary")
        upload_output = gr.Textbox(label="处理结果")
        
        upload_btn.click(process_upload, inputs=[file_input, topics_input], outputs=upload_output)

    with gr.Tab("🔍 语义文献搜索"):
        gr.Markdown("输入自然语言问题，搜索相关论文片段。")
        search_input = gr.Textbox(label="输入问题", placeholder="例如: How does transformer work?")
        top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="返回结果数量")
        search_btn = gr.Button("搜索")
        search_output = gr.Markdown(label="搜索结果")
        
        search_btn.click(search_docs, inputs=[search_input, top_k_slider], outputs=search_output)

    with gr.Tab("🖼️ 以文搜图"):
        gr.Markdown("输入描述搜索本地图片。请确保 data 目录下有图片。")
        with gr.Row():
            idx_btn = gr.Button("🔄 重建图片索引 (扫描 data/ 目录)")
            idx_output = gr.Textbox(label="索引状态", show_label=False)
        
        idx_btn.click(index_local_images, outputs=idx_output)
        
        img_query = gr.Textbox(label="图片描述", placeholder="例如: A diagram of neural network")
        img_btn = gr.Button("搜图")
        gallery = gr.Gallery(label="搜索结果", columns=2, height="auto")
        
        img_btn.click(search_imgs, inputs=img_query, outputs=gallery)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)