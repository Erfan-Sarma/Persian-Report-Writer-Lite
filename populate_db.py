import os
import glob
import shutil
from docx import Document as DocxReader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def extract_text_from_docx(file_path):
    """خواندن متن از فایل Word و پاکسازی اطلاعات زائد انتهای گزارش"""
    try:
        doc = DocxReader(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        
        text = "\n".join(full_text)
        
        # یک قانون برای حذف بخش‌های زائد انتهای گزارش (مثل کدها یا بخش هزینه‌ها)
        # اگر متنی بعد از اعلام قیمت یا امضا باشد، آن را قطع می‌کنیم.
        if "تعیین و اعلام می‌گردد" in text:
            # متن را تا اتمام اعلام قیمت نگه می‌داریم و بقیه (کدها/هزینه‌ها) را حذف می‌کنیم
            parts = text.split("تعیین و اعلام می‌گردد")
            text = parts[0] + "تعیین و اعلام می‌گردد."
            
        return text
    except Exception as e:
        print(f"⚠️ خطا در خواندن فایل {file_path}: {e}")
        return None

def initialize_vector_db():
    print("🧹 پاکسازی دیتابیس قدیمی...")
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        
    print("🧠 راه‌اندازی مدل Nomic Embeddings از Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # پیدا کردن تمام فایل‌های docx در پوشه جاری
    docx_files = glob.glob("/run/media/erfan/62B696FDB696D141/WorkStation/SJ/Reports/*.docx")
    if not docx_files:
        print("❌ هیچ فایل .docx در این پوشه پیدا نشد! لطفاً فایل‌های گزارش را کنار اسکریپت قرار دهید.")
        return
        
    documents = []
    print(f"📂 در حال پردازش {len(docx_files)} فایل گزارش قدیمی...")
    
    for file_path in docx_files:
        content = extract_text_from_docx(file_path)
        if content:
            # ذخیره کردن متن تمیز شده به عنوان ساختار مرجع در دیتابیس
            doc = Document(
                page_content=content,
                metadata={"file_name": file_path, "type": "justice_report"}
            )
            documents.append(doc)
            
    print("💾 در حال ذخیره‌سازی داده‌ها در ChromaDB لوکال...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ بانک اطلاعاتی گزارش‌های کارشناسی با موفقیت ایجاد و به روز شد!")

if __name__ == "__main__":
    initialize_vector_db()