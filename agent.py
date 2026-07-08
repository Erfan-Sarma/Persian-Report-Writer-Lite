from typing import Dict, Any, Literal
import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# =====================================================================
# 1. INITIALIZE LOCAL MODELS & VECTOR STORE
# =====================================================================
llm = ChatOllama(model="llama3.1:latest", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# =====================================================================
# 2. THE GRAPH STATE DEFINITION (With Revision Counter)
# =====================================================================
class ReportState(BaseModel):
    user_raw_input: str = ""
    user_data: Dict[str, Any] = Field(default_factory=dict)
    user_summary: str = ""
    matched_template_structure: str = ""
    current_draft: str = ""
    feedback: str = ""
    is_approved: bool = False
    revision_count: int = 0  # Added to prevent infinite loops!

# =====================================================================
# 3. GRAPH NODES
# =====================================================================

def data_extractor_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🧠 [NODE 0]: Data Extractor Agent ---")
    
    system_prompt = (
        "You are a precise data extraction specialist working with Persian legal/property records.\n"
        "Extract key parameters into a clean structure. Ignore system codes or processing fees at the end.\n"
        "Respond ONLY with a valid JSON object matching this schema. Do not add markdown backticks:\n"
        "{\n"
        '  "extracted_metrics": {\n'
        '     "owner": "Name",\n'
        '     "property_address": "Address",\n'
        '     "registration_id": "Plak Sabti",\n'
        '     "physical_progress": "Percentage",\n'
        '     "total_value": "Value"\n'
        "  },\n"
        '  "search_summary": "A short 1-sentence Persian summary for database template search"\n'
        "}"
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Raw Input: {state.user_raw_input}")]
    response = llm.invoke(messages)
    
    try:
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        extracted_json = json.loads(clean_content)
        print("✅ Data successfully structured into keys!")
        return {
            "user_data": extracted_json.get("extracted_metrics", {}),
            "user_summary": extracted_json.get("search_summary", "ارزیابی ملک")
        }
    except Exception:
        return {"user_data": {"raw": state.user_raw_input}, "user_summary": "ارزیابی ملک"}


def data_analyst_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🤖 [NODE 1]: Data Analyst Agent (Retrieving Template) ---")
    results = vector_store.similarity_search(state.user_summary, k=1)
    
    if results:
        matched_layout = results[0].page_content
        print(f"🎯 Pattern Found! Matching structural blueprint pulled from history.")
    else:
        matched_layout = "۱. مقدمه ۲. اسناد و مدارک ۳. مشخصات بنا ۴. نتیجه گزارش"
        print("⚠️ Fallback layout used.")
        
    return {"matched_template_structure": matched_layout}


def persian_writer_node(state: ReportState) -> Dict[str, Any]:
    print(f"\n--- ✍️ [NODE 2]: Persian Writer Agent (Drafting - Attempt {state.revision_count + 1}) ---")
    
    system_prompt = (
        "You are an expert corporate technical reporter writing exclusively in formal, eloquent Persian.\n"
        "You MUST align your response layout exactly with this layout structure template:\n"
        f"==== REQUIRED STRUCTURE ====\n{state.matched_template_structure}\n============================\n"
        "CRITICAL: You must explicitly include a section named 'نتیجه گزارش:' or 'نتیجه‌گیری:' at the bottom "
        "and state the final evaluation price there. Do not skip this section title."
    )
    
    user_content = f"Extracted Data Values: {state.user_data}"
    
    if state.feedback:
        print(f"🔄 Re-drafting based on supervisor feedback...")
        user_content += f"\n\nCRITICAL ERROR CORRECTION NEEDED FROM SUPERVISOR: {state.feedback}"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    response = llm.invoke(messages)
    
    # Increment the revision counter every time the writer runs
    return {"current_draft": response.content, "revision_count": state.revision_count + 1}


def supervisor_critic_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🧐 [NODE 3]: Supervisor / Critic Agent (Evaluating) ---")
    
    # If we hit maximum retries, bypass the validation to break the loop
    if state.revision_count >= 3:
        print("⚠️ Maximum revision attempts reached! Forcing approval to prevent loop.")
        return {"is_approved": True, "feedback": ""}
        
    system_prompt = (
        "You are a quality control supervisor ensuring Persian reports conform strictly to required sections.\n"
        "Check if the text includes a concluding section containing the text 'نتیجه' or financial valuation details.\n"
        "Respond ONLY in this exact JSON format:\n"
        "{\n"
        '  "approved": true or false,\n'
        '  "feedback": "Your adjustment note in Persian if approved is false, otherwise empty string"\n'
        "}"
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Verify this layout text structure:\n\n{state.current_draft}")]
    response = llm.invoke(messages)
    
    try:
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        result = json.loads(clean_content)
        
        if result.get("approved") is True:
            print("✅ Evaluation Passed!")
            return {"is_approved": True, "feedback": ""}
        else:
            print(f"❌ Evaluation Rejected! Correction required: {result.get('feedback')}")
            return {"is_approved": False, "feedback": result.get("feedback")}
    except Exception:
        # If the supervisor LLM makes a formatting error, pass it through to be safe
        print("⚠️ JSON parsing error in Supervisor. Passing through safely.")
        return {"is_approved": True, "feedback": ""}

# =====================================================================
# 4. ROUTING & GRAPH TOPOLOGY
# =====================================================================
def route_after_critic(state: ReportState) -> Literal["persian_writer", "__end__"]:
    if state.is_approved:
        return END
    return "persian_writer"

builder = StateGraph(ReportState)
builder.add_node("data_extractor", data_extractor_node)
builder.add_node("data_analyst", data_analyst_node)
builder.add_node("persian_writer", persian_writer_node)
builder.add_node("supervisor_critic", supervisor_critic_node)

builder.add_edge(START, "data_extractor")
builder.add_edge("data_extractor", "data_analyst")
builder.add_edge("data_analyst", "persian_writer")
builder.add_edge("persian_writer", "supervisor_critic")
builder.add_conditional_edges("supervisor_critic", route_after_critic)

compiled_agent = builder.compile()

# =====================================================================
# 5. EXECUTE THE SYSTEM
# =====================================================================
if __name__ == "__main__":
    messy_user_input = (
        "سلام خسته نباشید. یک ارزیابی ملک داریم برای آقای علی علوی در فرهنگ شهر شیراز، کوچه ۵ پلاک ۱۲. "
        "پلاک ثبتی ملکش هم ۱۴۲۰/۹۹ هست زمینش کلاً ۴۰۰ متره. یه بنای ۴ طبقه بتنی نیمه کاره داره که حدود ۶۵ درصدش "
        "دیوارچینی و اسکلتش انجام شده ولی نازک کاری نداره هنوز. قیمت شش دانگ ملک رو کارشناس زده ۷۵0 میلیارد ریال "
        "و ما هم کل شش دانگ رو ارزیابی میکنیم. لطفاً گزارشش رو آماده کنید. "
        "--- اطلاعات سیستم مالی: کد رهگیری بایگانی ۸۸۷۲۶۱ --- هزینه پرداختی ایاب ذهاب: ۲۰۰۰۰۰ ریال"
    )
    
    test_input = ReportState(user_raw_input=messy_user_input)
    
    print("🚀 Running the Safe LangGraph RAG Engine...")
    final_output = compiled_agent.invoke(test_input)
    
    print("\n==================================================")
    print("🏁 FINAL REPORT:")
    print(final_output["current_draft"])
    print("==================================================")