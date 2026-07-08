from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. THE STATE DEFINITION
# This is the single source of truth that moves through the graph.
# =====================================================================
class ReportState(BaseModel):
    user_data: Dict[str, Any] = Field(default_factory=dict, description="Raw input numbers/metrics provided by user.")
    user_summary: str = ""                                 # The brief summary or goal prompt
    matched_template_structure: str = ""                  # Extracted report layout from older examples
    current_draft: str = ""                               # The Persian report text draft
    feedback: str = ""                                    # Any criticism or revision notes if structure fails
    is_approved: bool = False                             # Flag determining if report matches rules


# =====================================================================
# 2. NODES (The Worker Functions)
# Each function takes the current state, does work, and returns updates.
# =====================================================================

def data_analyst_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🤖 [NODE 1]: Data Analyst Agent ---")
    print(f"Analyzing user metrics: {state.user_data}")
    
    # In a later step, this will query ChromaDB/FAISS for an old report structure
    # For now, we simulate matching a specific structural layout.
    simulated_template = "Structure: [Header] -> [Data Analysis Table] -> [Persian Conclusion]"
    print(f"Found matching historical report structure.")
    
    return {"matched_template_structure": simulated_template}


def persian_writer_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- ✍️ [NODE 2]: Persian Writer Agent ---")
    print(f"Drafting report using layout: {state.matched_template_structure}")
    
    # Simulating LLM drafting text in Persian based on whether feedback was given
    if state.feedback:
        print(f"Applying supervisor feedback: '{state.feedback}'")
        draft = "گزارش نهایی اصلاح شده: مقادیر ورودی با ساختار کاملاً هماهنگ هستند و بخش نتیجه‌گیری اضافه شد."
    else:
        # A basic draft missing a mandatory section on purpose to test our Critic loop!
        draft = "گزارش اولیه: مقادیر سیستم تحلیل شدند."
        
    return {"current_draft": draft}


def supervisor_critic_node(state: ReportState) -> Dict[str, Any]:
    print("\n--- 🧐 [NODE 3]: Supervisor / Critic Agent ---")
    print("Checking report structure against historical rules...")
    
    # Let's write rules simulating validation
    # If it doesn't contain 'نتیجه‌گیری' (Conclusion), reject it once to show a loop
    if "نتیجه‌گیری" in state.current_draft:
        print("✅ Report passes all structural requirements!")
        return {"is_approved": True, "feedback": ""}
    else:
        print("❌ Critique: The report lacks a Persian Conclusion section.")
        return {"is_approved": False, "feedback": "بخش نتیجه‌گیری نهایی را به گزارش اضافه کنید."}


# =====================================================================
# 3. CONDITIONAL EDGES (Routing Logic)
# Determines whether to finish or route backwards into a cycle loop.
# =====================================================================
def route_after_critic(state: ReportState) -> Literal["persian_writer", "__end__"]:
    if state.is_approved:
        return END  # Finished!
    else:
        print("🔄 State routed backward to Persian Writer for revisions...")
        return "persian_writer"  # Send back to writer node


# =====================================================================
# 4. BUILDING THE GRAPH ARCHITECTURE
# =====================================================================
# Create the graph builder object linked to our State schema
builder = StateGraph(ReportState)

# Define our steps (Nodes)
builder.add_node("data_analyst", data_analyst_node)
builder.add_node("persian_writer", persian_writer_node)
builder.add_node("supervisor_critic", supervisor_critic_node)

# Set up the execution flow connections (Edges)
builder.add_edge(START, "data_analyst")          # Start here
builder.add_edge("data_analyst", "persian_writer") # Then go to writer
builder.add_edge("persian_writer", "supervisor_critic") # Then evaluate

# Add the conditional check after evaluation
builder.add_conditional_edges("supervisor_critic", route_after_critic)

# Compile everything into an executable application
compiled_agent = builder.compile()


# =====================================================================
# 5. EXECUTION EXAMPLE
# =====================================================================
if __name__ == "__main__":
    # Simulate a user providing input data and a brief summary
    initial_input = ReportState(
        user_data={"temperature": 42.5, "pressure": 101.3, "status": "Critical"},
        user_summary="گزارش خطای سیستم در بخش تولید"
    )
    
    print("🚀 Running the LangGraph Agent System...")
    final_output = compiled_agent.invoke(initial_input)
    
    print("\n==================================================")
    print("🏁 FINAL OUTPUT GENERATED BY GRAPH:")
    print(final_output["current_draft"])
    print("==================================================")