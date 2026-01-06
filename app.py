import streamlit as st
import os
import chromadb
from PIL import Image
import numpy as np
import tempfile
import shutil

from model_loader import EmbeddingModel
from utils import extract_text_with_page_numbers, move_file_to_category

# --- 页面配置 ---
st.set_page_config(page_title="多模态 AI 助手", layout="wide", page_icon="🤖")

# --- 核心资源加载---
@st.cache_resource
def load_models():
    """加载模型，只执行一次"""
    return EmbeddingModel()

@st.cache_resource
def load_db():
    """连接数据库，只执行一次"""
    client = chromadb.PersistentClient(path="./db")
    paper_collection = client.get_or_create_collection(name="papers")
    image_collection = client.get_or_create_collection(name="images")
    return paper_collection, image_collection

# 初始化加载
try:
    with st.spinner('正在加载 AI 模型 (MiniLM & CLIP)... 请稍候'):
        model_handler = load_models()
        paper_collection, image_collection = load_db()
    st.success("模型与数据库加载完毕！")
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# --- 侧边栏 ---
st.sidebar.title("🤖 AI Agent 控制台")
app_mode = st.sidebar.radio("选择功能", ["📄 论文上传与分类", "🔍 语义文献搜索", "🖼️ 以文搜图"])

# --- 功能 1: 论文上传与分类 ---
if app_mode == "📄 论文上传与分类":
    st.title("📄 智能论文归档")
    st.markdown("上传 PDF，系统将自动分析内容、分类并建立语义索引。")

    # 1. 话题设置
    topics_input = st.text_input("设置分类主题 (用逗号分隔)", "CV,NLP,Agent,RL,Backbone")
    
    # 2. 文件上传
    uploaded_files = st.file_uploader("上传 PDF 论文", type=["pdf"], accept_multiple_files=True)

    if st.button("开始处理") and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"正在处理: {uploaded_file.name}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # 1. 提取文本
            chunks = extract_text_with_page_numbers(tmp_path)
            
            if not chunks:
                st.warning(f"文件 {uploaded_file.name} 无法提取文本。")
                os.unlink(tmp_path)
                continue

            # 2. 确定分类
            summary_text = " ".join([c["text"] for c in chunks[:3]])
            summary_vec = model_handler.get_text_embedding(summary_text)

            best_topic = "Uncategorized"
            if topics_input:
                t_list = topics_input.split(',')
                t_vecs = model_handler.text_model.encode(t_list)
                scores = [np.dot(t_v, summary_vec) for t_v in t_vecs]
                best_topic = t_list[np.argmax(scores)]

            # 3. 移动文件到真实的数据目录
            target_dir = os.path.join("data", best_topic)
            os.makedirs(target_dir, exist_ok=True)
            final_path = os.path.join(target_dir, uploaded_file.name)

            shutil.copy(tmp_path, final_path)

            # 4. 存入数据库
            ids, docs, vecs, metas = [], [], [], []
            for chunk in chunks:
                page_id = f"{uploaded_file.name}_p{chunk['page']}"
                ids.append(page_id)
                docs.append(chunk["text"])
                vecs.append(model_handler.get_text_embedding(chunk["text"]))
                metas.append({
                    "path": final_path,
                    "topic": best_topic,
                    "page": chunk["page"]
                })

            paper_collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
            st.success(f"✅ {uploaded_file.name} -> 归类为 **{best_topic}** (索引了 {len(chunks)} 页)")

            os.unlink(tmp_path)
            progress_bar.progress((idx + 1) / len(uploaded_files))

# --- 功能 2: 语义文献搜索 ---
elif app_mode == "🔍 语义文献搜索":
    st.title("🔍 深度语义搜索")
    query = st.text_input("请输入问题或关键词", "How does self-attention work?")
    top_k = st.slider("返回结果数量", 1, 10, 3)

    if st.button("搜索") or query:
        if not query:
            st.warning("请输入查询内容")
        else:
            query_vec = model_handler.get_text_embedding(query)
            results = paper_collection.query(query_embeddings=[query_vec], n_results=top_k)

            if not results['ids'] or not results['ids'][0]:
                st.info("没有找到相关结果。")
            else:
                for i, _ in enumerate(results['ids'][0]):
                    meta = results['metadatas'][0][i]
                    snippet = results['documents'][0][i]
                    # score = results['distances'][0][i] 
                    
                    with st.container():
                        st.markdown(f"### 📄 结果 {i+1}: {os.path.basename(meta.get('path', 'Unknown'))}")
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.info(f"**Topic**: {meta.get('topic', 'N/A')}\n\n**Page**: {meta.get('page', 'N/A')}")
                        with col2:
                            st.markdown(f"> ...{snippet[:500]}...")
                        st.divider()

# --- 功能 3: 以文搜图 ---
elif app_mode == "🖼️ 以文搜图":
    st.title("🖼️ 智能图片检索")

    if st.sidebar.button("重建图片索引 (扫描 data/ 目录)"):
        with st.spinner("正在扫描图片..."):
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
            st.sidebar.success(f"成功索引 {count} 张图片！")

    query = st.text_input("描述你想找的图片", "A diagram of transformer architecture")
    
    if query:
        query_vec = model_handler.get_text_for_image_embedding(query)
        results = image_collection.query(query_embeddings=[query_vec], n_results=4)

        if not results['ids'] or not results['ids'][0]:
            st.info("没有找到图片。请确保 data 目录下有图片并已点击左侧'重建索引'。")
        else:
            cols = st.columns(2)
            for i, doc_id in enumerate(results['ids'][0]):
                meta = results['metadatas'][0][i]
                img_path = meta['path']
                
                with cols[i % 2]:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"{doc_id} (Path: {img_path})", use_container_width=True)
                    else:
                        st.error(f"图片丢失: {img_path}")