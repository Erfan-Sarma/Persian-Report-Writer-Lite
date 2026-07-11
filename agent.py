import json
import os
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# =====================================================================
# 1. INITIALIZATION 
# =====================================================================
# Temperature set to 0.0 to force absolute strictness
llm = ChatOllama(model="llama3.1:latest", temperature=0.0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# =====================================================================
# 2. STATE DEFINITION
# =====================================================================
class ReportState(BaseModel):
    user_raw_input: str = ""
    extracted_keywords: str = ""
    matched_template_layout: str = ""
    
    intro_section: str = ""
    documents_section: str = ""
    property_specs_section: str = ""
    valuation_section: str = ""
    
    current_draft: str = ""
    is_approved: bool = False
    feedback: str = ""
    revision_count: int = 0
    exported_file_path: str = ""

# =====================================================================
# 3. MODULAR AGENT NODES
# =====================================================================

def data_analyzer_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🤖 [Node 1]: Data Analyzer & RAG Layout Matcher ---")
    
    messages = [
        SystemMessage(content="Generate a 3-word Persian search query based on the input text. ONLY output the 3 words."),
        HumanMessage(content=state.user_raw_input)
    ]
    search_query = llm.invoke(messages).content.strip()
    
    results = vector_store.similarity_search(search_query, k=1)
    matched_layout = results[0].page_content if results else "No DB Match Found."
    
    # VISUAL VERIFICATION FOR THE USER: See exactly what the database is doing!
    print("\n=======================================================")
    print(f"🔍 [RAG DATABASE FETCH] Based on query: '{search_query}'")
    print("Here is a snippet of what ChromaDB pulled from your DOCX files:")
    print(f"   -> \"{matched_layout}\"")
    print("=======================================================\n")
    
    return {"extracted_keywords": search_query, "matched_template_layout": matched_layout}


def intro_writer_node(state: ReportState) -> Dict[str, Any]:
    print("--- ✍️ [Node 2-A]: Intro Writer Node (Strictly No Metrics) ---")
    
    system_prompt = (
        "You are an expert Iranian court scribe. Write ONLY the introduction paragraph.\n"
        "ABSOLUTE RULES:\n"
        "1. Start directly with 'ریاست محترم...'. Do not say hello or add conversational text.\n"
        "2. Include ONLY the court branch, notification number, date, plaintiff/defendant names, and the property address.\n"
        "3. FATAL ERROR: You are strictly FORBIDDEN from using the words 'متر', 'ریال', 'تومان', 'اسکلت', 'بتن', 'طبقه', or 'سوله'.\n"
        "4. FATAL ERROR: Do NOT mention any money, measurements, or physical building specs. If you do, the report is invalid."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    return {"intro_section": llm.invoke(messages).content.strip()}


def documents_writer_node(state: ReportState) -> Dict[str, Any]:
    print("--- ✍️ [Node 2-B]: Documents Writer Node (Deeds Only) ---")
    
    system_prompt = (
        "You are a registry scribe. Write the 'اسناد و مدارک' section.\n"
        "ABSOLUTE RULES:\n"
        "1. Start directly with the text (e.g., 'به موجب اصل سند...'). No conversational filler.\n"
        "2. Include ONLY registry IDs, title deed numbers (شماره چاپی/پلاک ثبتی), and postal codes.\n"
        "3. FATAL ERROR: You are strictly FORBIDDEN from mentioning physical building materials, skeletons, or financial values (ریال/تومان)."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    return {"documents_section": llm.invoke(messages).content.strip()}


def property_specs_writer_node(state: ReportState) -> Dict[str, Any]:
    print("--- ✍️ [Node 2-C]: Specs Writer Node (Architecture Only) ---")
    
    system_prompt = (
        "You are an engineering scribe. Write the 'مشخصات فنی و معماری' section in a dense, formal continuous paragraph.\n"
        "ABSOLUTE RULES:\n"
        "1. Write about the land area, building frame (اسکلت), roof, floors, facade, and unit count.\n"
        "2. FATAL ERROR: Do NOT mention court names, plaintiff/defendant names, or financial valuations (ریال/تومان)."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    return {"property_specs_section": llm.invoke(messages).content.strip()}


def valuation_writer_node(state: ReportState) -> Dict[str, Any]:
    print("--- ✍️ [Node 2-D]: Valuation Node (Boilerplate Injection) ---")
    
    # Here we inject YOUR exact requested boilerplate!
    system_prompt = (
        "You are the Chief Financial Valuator. Write the final evaluation paragraph.\n"
        "You MUST use the exact structure and phrasing provided below, simply filling in the specific context and numbers from the user's prompt.\n\n"
        "REQUIRED TEMPLATE FORMAT:\n"
        "با توجه به بازدید میدانی و بررسی های به عمل آمده، با عنایت به محدوده وقوع و موقعیت املاک، مساحت عرصه، دسترسی و توجه به نرخ عادلانه روز و از طرفی صرف نظر از اصالت، صحت و سقم اسناد و مدارک و صرف نظر از سوابق ثبتی و بدون در نظرگرفتن تعهدات و دیونی که ممکن است به افراد حقیقی و حقوقی، بانکها، شهرداری، مراجع قضایی یا سایر سازمان ها و ارگان های دولتی و خصوصی وجود داشته باشد، ارزش [INSERT PROPERTY TYPE/SHARES HERE] مطابق آدرس فوق جمعا به مبلغ [INSERT RIAL VALUE HERE] ریال معادل [INSERT TOMAN VALUE HERE] تومان تعیین و اعلام می گردد.\n\n"
        "ABSOLUTE RULE: Do not add any conversational filler. Output only the finalized paragraph based on the template."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    return {"valuation_section": llm.invoke(messages).content.strip()}


def report_compiler_node(state: ReportState) -> Dict[str, Any]:
    print("--- 🧩 [Node 3]: Compiler (Using User's Professional Layout) ---")
    
    # Using your cleaner, professional layout
    compiled_text = (
        f"{state.intro_section}\n\n"
        f"اسناد و مدارک :\n"
        f"{state.documents_section}\n\n"
        f"مشخصات فنی، معماری و توصیف اعیان ملک:\n"
        f"{state.property_specs_section}\n\n"
        f"نتیجه کارشناسی و ارزیابی مالی:\n"
        f"{state.valuation_section}"
    )
    return {"current_draft": compiled_text}


def file_exporter_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 💾 [Node 4]: File Exporter Node ---")
    output_filename = "generated_reprots/detailed_judicial_report.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(state.current_draft)
        return {"exported_file_path": output_filename}
    except Exception as e:
        return {"exported_file_path": f"ERROR: {e}"}

# =====================================================================
# 4. TOPOLOGY (Bypassed Critic for stability during prompt testing)
# =====================================================================
workflow = StateGraph(ReportState)

workflow.add_node("data_analyzer", data_analyzer_node)
workflow.add_node("intro_writer", intro_writer_node)
workflow.add_node("documents_writer", documents_writer_node)
workflow.add_node("property_specs_writer", property_specs_writer_node)
workflow.add_node("valuation_writer", valuation_writer_node)
workflow.add_node("report_compiler", report_compiler_node)
workflow.add_node("file_exporter", file_exporter_node)

workflow.add_edge(START, "data_analyzer")
workflow.add_edge("data_analyzer", "intro_writer")
workflow.add_edge("intro_writer", "documents_writer")
workflow.add_edge("documents_writer", "property_specs_writer")
workflow.add_edge("property_specs_writer", "valuation_writer")
workflow.add_edge("valuation_writer", "report_compiler")
workflow.add_edge("report_compiler", "file_exporter")
workflow.add_edge("file_exporter", END)

compiled_pipeline = workflow.compile()

# =====================================================================
# 5. TEST RUN
# =====================================================================
if __name__ == "__main__":
    industrial_case_input = (
        "سلام، وقت بخیر. یک گزارش کارشناسی برای شعبه پنجم دادگاه حقوقی مرودشت باید تنظیم کنیم. "
        "شماره ابلاغیه 1405999222333444 هست به تاریخ 1405/02/10. موضوع دعوی مربوط به بانک ملی ایران (خواهان) و "
        "شرکت تولیدی آریان (خوانده) هست. آدرس ملک: مرودشت، شهرک صنعتی، فاز ۲، خیابان تلاش، پلاک ۱۱۴. "
        "ملک یک سوله صنعتی هست. مساحت کل زمین (عرصه) ۱۲۰۰ متر مربع هست. یک سوله با اسکلت فلزی و سقف ورق گالوانیزه "
        "به مساحت ۸۰۰ متر مربع ساخته شده که کف آن بتن‌ریزی صنعتی است. یک بخش اداری هم در دو طبقه با اسکلت بتنی "
        "به مساحت مجموعاً ۱۰۰ متر مربع (هر طبقه ۵۰ متر) جلوی سوله قرار داره که نمای آن آجر نسوز هست. "
        "سند تک برگی داره به شماره چاپی 112233 سری الف سال ۹۸. پلاک ثبتی ۱۱۴ فرعی از ۸۸ اصلی در بخش ثبتی ۵ مرودشت. "
        "ملک پایان کار شهرداری ندارد و فعلاً متروکه است. ارزش کل عرصه و اعیان توسط کارشناس ۸۵،۰۰۰،۰۰۰،۰۰۰ ریال "
        "برآورد شده. ضمناً نماینده بانک ۲۰ دقیقه دیر به محل بازدید رسید و هزینه کارشناسی هم به مبلغ ۵ میلیون ریال "
        "به حساب واریز شد که نیازی نیست این دو مورد آخر تو گزارش نهایی بیاد."
    )
    
    test_state = ReportState(user_raw_input=industrial_case_input)
    print("🚀 Initializing Pipeline...")
    execution_result = compiled_pipeline.invoke(test_state)
    
    print("\n======================================================================")
    print(f"🏁 Execution Finished. File saved to: {execution_result.get('exported_file_path', 'Unknown')}")
    print("======================================================================")