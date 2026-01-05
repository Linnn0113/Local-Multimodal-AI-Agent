import os
import shutil
from pypdf import PdfReader

def extract_text_with_page_numbers(pdf_path):
    """读取 PDF，按页提取文本，返回列表"""
    chunks = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or len(text.strip()) < 50: continue
            chunks.append({
                "text": text.strip(),
                "page": page_num + 1
            })
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return chunks

def move_file_to_category(file_path, category):
    """文件移动逻辑"""
    target_dir = os.path.join(os.path.dirname(file_path), category)
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.basename(file_path)
    target_path = os.path.join(target_dir, filename)
    if os.path.abspath(file_path) != os.path.abspath(target_path):
        shutil.move(file_path, target_path)
    return target_path