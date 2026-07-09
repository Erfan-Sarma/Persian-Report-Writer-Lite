import json
import os
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# =====================================================================
# 1. INITIALIZATION (Models & Vector Database)
# =====================================================================
# Initialize local LLM and embedding model
llm = ChatOllama(model="llama3.1:latest", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize Vector Store for Dynamic Template Matching (RAG)
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# =====================================================================
# 2. ADVANCED AGENT STATE DEFINITION
# =====================================================================
class ReportState(BaseModel):
    """Tracks the state of the report compilation across all modular nodes."""
    user_raw_input: str = ""
    extracted_keywords: str = ""
    matched_template_layout: str = ""
    
    # Granular segments to ensure no technical detail or metric is lost
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
# 3. MODULAR AGENT NODES (English Code & System Prompts)
# =====================================================================

def data_analyzer_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🤖 [Node 1]: Data Analyzer & RAG Layout Matcher ---")
    
    # Generate a tight search query from the raw inputs to find matching layouts
    messages = [
        SystemMessage(
            content="Generate a single concise Farsi search query (e.g., 'ارزیابی ارث دادگاه') "
                    "to locate a matching legal report template configuration from the database. "
                    "Respond with ONLY the query phrase."
        ),
        HumanMessage(content=state.user_raw_input)
    ]
    response = llm.invoke(messages)
    search_query = response.content.strip()
    
    # Retrieve the closest structural layout format from the Vector DB
    results = vector_store.similarity_search(search_query, k=1)
    matched_layout = results[0].page_content if results else "Default Layout: Intro -> Docs -> Specs -> Evaluation"
    
    print(f"-> Successfully retrieved relevant legal layout configuration based on input query.")
    return {"extracted_keywords": search_query, "matched_template_layout": matched_layout}


def intro_writer_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- ✍️ [Node 2-A]: Specialized Intro Writer Node ---")
    
    system_prompt = (
        "You are an expert legal report editor for the Iranian judiciary system.\n"
        "Your sole task is to extract and write the 'Introduction and Case Background' section "
        "of a court expert report.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Write exclusively in highly formal, prestigious, and bureaucratic Persian (ادبیات رسمی دادگستری).\n"
        "2. Extract and accurately preserve all case metadata: Court Name, Branch Number, Notification/Case ID, "
        "Dates, Plaintiff/Defendant names, and the exact property address.\n"
        "3. Do NOT include property specifications, metrics, asset calculations, or final financial valuations here."
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    response = llm.invoke(messages)
    return {"intro_section": response.content.strip()}


def documents_writer_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- ✍️ [Node 2-B]: Specialized Legal Documents & Titles Writer Node ---")
    
    system_prompt = (
        "You are an expert asset validation agent for the court.\n"
        "Your sole task is to compile the 'Documents and Evidence' (اسناد و مدارک) section of the report.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Write exclusively in highly formal, structured Persian.\n"
        "2. Extract every single legal identifier: Single-page title deed numbers (سند تک‌برگ), "
        "printing serials, main/sub-registry plot numbers (پلاک اصلی و فرعی), registration sectors, "
        "postal codes, or historical probate certification numbers (گواهی انحصار وراثت) from dispute resolutions.\n"
        "3. If the input specifies that the property lacks standard registry documents or construction permits, "
        "detail that exact status explicitly with legal terminology (فاقد سند رسمی ثبتی و فاقد پروانه ساختمانی).\n"
        "4. Do NOT drop any document serial numbers or registration IDs."
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    response = llm.invoke(messages)
    return {"documents_section": response.content.strip()}


def property_specs_writer_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- ✍️ [Node 2-C]: Specialized Structural & Technical Specs Node ---")
    
    system_prompt = (
        "You are a Senior Structural Engineer and Court Valuation Expert.\n"
        "Your sole task is to generate the 'Property Specifications & Technical Details' (مشخصات ساختمان) section.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Write in highly detailed, professional engineering/real-estate Persian.\n"
        "2. Extract and layout EVERY single physical specification: Structural skeleton frame type (concrete, masonry, ties), "
        "roofing style (joist & foam/block), number of stories, exact land area (عرصه), exact floor-by-floor infrastructure "
        "built-up areas (اعیانی), staircase setups, balconies with their exact individual square footage, and commercial storefront dimensions.\n"
        "3. Include interior finishes (plaster work, mosaic floorings, tile height measurements, kitchen cabinetry status), "
        "utility connections, exterior facade descriptions, and current occupancy or possessory statuses of each unit.\n"
        "4. WARNING: Do NOT summarize. Every square meter, balcony metric, or material mention from the source data must be retained."
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    response = llm.invoke(messages)
    return {"property_specs_section": response.content.strip()}


def valuation_writer_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- ✍️ [Node 2-D]: Specialized Financial Valuation & Summary Node ---")
    
    system_prompt = (
        "You are a Court Forensic Financial Valuator.\n"
        "Your sole task is to generate the 'Final Conclusion and Financial Evaluation' (نتیجه گزارش و ارزیابی مالی) section.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Write in precise, ironclad legal-financial Persian.\n"
        "2. Break down all expert evaluations individually: Land value (ارزش عرصه), base floor building value, "
        "upper floor values, and separate residential units.\n"
        "3. Present the total combined valuation clearly in both numeric Rial format and written out text (both in Rial and Toman if necessary).\n"
        "4. If inheritance shares are present, output a structured list mapping each heir to their precise fraction/percentage "
        "based on the source information.\n"
        "5. Conclude with standard legal disclaimers protecting the appraisal against external debts, mortgages, or registry claims "
        "(صرف‌نظر از هرگونه بدهی، دیون، معارض و سوابق ثبتی)."
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_raw_input)]
    response = llm.invoke(messages)
    return {"valuation_section": response.content.strip()}


def report_compiler_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🧩 [Node 3]: Comprehensive Report Compiler & Aggregator ---")
    
    # Aggregate all individual Persian blocks into a professionally formatted document structure
    compiled_text = (
        f"{state.intro_section}\n\n"
        f"📋 اسناد و مدارک ابرازی:\n"
        f"----------------------------------------------------------------------\n"
        f"{state.documents_section}\n\n"
        f"🏢 مشخصات فنی، معماری و توصیف اعیان ملک:\n"
        f"----------------------------------------------------------------------\n"
        f"{state.property_specs_section}\n\n"
        f"⚖️ نتیجه کارشناسی و ارزیابی مالی:\n"
        f"----------------------------------------------------------------------\n"
        f"{state.valuation_section}"
    )
    return {"current_draft": compiled_text}


def supervisor_critic_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🧐 [Node 4]: Quality Supervisor & Cross-Reference Critic ---")
    
    # Circuit breaker to prevent endless loops in automation environment
    if state.revision_count >= 2:
        print("-> Max revision limit reached. Approving draft to prevent infinite loop.")
        return {"is_approved": True, "feedback": ""}
        
    system_prompt = (
        "You are the Chief Judiciary Quality Inspector. Analyze the assembled report draft.\n"
        "Ensure no critical numeric values, specific dimensions, or structural details from an expert perspective are missing.\n"
        "You MUST return a pure JSON object with exactly two keys: 'approved' (boolean) and 'feedback' (string).\n"
        "Do not include markdown tags like ```json or any conversational filler text. Just raw JSON."
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.current_draft)]
    response = llm.invoke(messages)
    
    try:
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        result = json.loads(clean_content)
        return {"is_approved": result.get("approved", True), "feedback": result.get("feedback", "")}
    except Exception as e:
        print(f"-> Failed to parse critic JSON: {e}. Defaulting to auto-approve.")
        return {"is_approved": True, "feedback": ""}


def file_exporter_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 💾 [Node 5]: Persistent File Exporter Node ---")
    output_filename = "detailed_judicial_report.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(state.current_draft)
        print(f"🚀 Success! Comprehensive Persian expert report safely compiled into '{output_filename}'.")
        return {"exported_file_path": output_filename}
    except Exception as e:
        print(f"-> File writing error encountered: {e}")
        return {"exported_file_path": f"ERROR: {e}"}

# =====================================================================
# 4. TOPOLOGY & WORKFLOW GRAPH CONSTRUCTION
# =====================================================================
def routing_governor(state: ReportState) -> Literal["intro_writer", "file_exporter"]:
    """Routes state forward based on structural verification criteria."""
    if state.is_approved:
        return "file_exporter"
    print(f"-> Draft rejected by supervisor. Loop back for optimization. Reason: {state.feedback}")
    return "intro_writer"

# Building the computational state graph execution sequence
workflow = StateGraph(ReportState)

workflow.add_node("data_analyzer", data_analyzer_node)
workflow.add_node("intro_writer", intro_writer_node)
workflow.add_node("documents_writer", documents_writer_node)
workflow.add_node("property_specs_writer", property_specs_writer_node)
workflow.add_node("valuation_writer", valuation_writer_node)
workflow.add_node("report_compiler", report_compiler_node)
workflow.add_node("supervisor_critic", supervisor_critic_node)
workflow.add_node("file_exporter", file_exporter_node)

# Set up chronological execution bounds
workflow.add_edge(START, "data_analyzer")
workflow.add_edge("data_analyzer", "intro_writer")
workflow.add_edge("intro_writer", "documents_writer")
workflow.add_edge("documents_writer", "property_specs_writer")
workflow.add_edge("property_specs_writer", "valuation_writer")
workflow.add_edge("valuation_writer", "report_compiler")
workflow.add_edge("report_compiler", "supervisor_critic")

# Dynamic conditional validation gate
workflow.add_conditional_edges("supervisor_critic", routing_governor)
workflow.add_edge("file_exporter", END)

# Compile agent system
compiled_pipeline = workflow.compile()

# =====================================================================
# 5. PIPELINE TEST RUN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Injecting your second highly detailed real-world apartment context payload to verify structural preservation
    sample_judicial_input = (
        "ریاست محترم شعبه اجرای احکام کیفری شهرستان زرقان. بازگشت به ابلاغیه شماره 1404121000574639070 مورخ 1404/08/21 "
        "در خصوص دعوی آقای سعید دریس و مجتبی پارسافر. از ملک ارائه شده به آدرس: شهرک مهر امام رضا - خیابان پاسارگاد ۳ - "
        "تعاونی ۲ - بلوک رز ۹ - طبقه اول - واحد ۶ بازدید شد. اسناد: به موجب اصل سند تک برگی شماره چاپی ۲۴۹۵۱۲ سری ب سال ۹۵ "
        "تحت پلاک ثبتی شماره ۳۹۰۰ فرعی از ۵۲۳۳ اصلی واقع در بخش ثبتی ۳ زرقان به مساحت ۷۲/۰۴ متر مربع دارای کد پستی 7341193628 "
        "شش دانگ اعیان آپارتمان به نام آقای شکراله دریس مورخ ۱۳۹۶/۱۲/۱۳ ثبت شده است. مشخصات ساختمان: یک واحد آپارتمان "
        "دو خوابه با اسکلت بتنی و سقف تیرچه بلوک واقع در طبقه اول یک بلوک ۴ طبقه ۱۹ واحدی به سمت جنوب غربی، دارای سالن، "
        "آشپزخانه، حمام، سرویس، تراس، سرمایش کولر آبی و گرمایش پکیج و رادیاتور با نمای سیمان رنگی. ارزش کل شش دانگ پلاک ثبتی "
        "جمعاً به مبلغ ۱۴،۰۰۰،۰۰۰،۰۰۰ ریال برابر با یک میلیارد و چهارصد میلیون تومان تعیین می‌گردد."
    )
    
    initial_state = ReportState(user_raw_input=sample_judicial_input)
    print("🚀 Initializing Persian Report Generation Pipeline under Modular English Architecture...")
    execution_result = compiled_pipeline.invoke(initial_state)
    
    print("\n======================================================================")
    print(f"🏁 Execution Finished. File saved to: {execution_result['exported_file_path']}")
    print("======================================================================")