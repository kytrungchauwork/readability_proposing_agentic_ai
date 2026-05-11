import streamlit as st
import time
import json
from phobert_singleton import phobert_model
from main import run_loop
from analyst import Analyst
from critic import Critic
from planning import Planner
from proposer import Proposer
from reviewer import Reviewer
from llm import get_llm

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Readability Agent Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL TECHNICAL UI (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #0066FF;
        --bg-slate: #F8FAFC;
        --border-color: #E2E8F0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-slate);
    }

    /* Monospace for technical data */
    .stCode, .stJson, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Professional Card System */
    .work-card {
        background: white;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 20px;
    }

    /* Level Badges */
    .badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-0 { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
    .badge-1 { background: #FEF9C3; color: #854D0E; border: 1px solid #FEF08A; }
    .badge-2 { background: #F3E8FF; color: #6B21A8; border: 1px solid #E9D5FF; }

    /* Buttons Style */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        text-transform: none;
        transition: all 0.2s;
    }
    
    .stButton button:disabled {
        background-color: #F1F5F9 !important;
        color: #94A3B8 !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* Technical Results Container */
    .result-box {
        background: #0F172A;
        color: #F8FAFC;
        padding: 25px;
        border-radius: 8px;
        line-height: 1.6;
        font-size: 1.1rem;
        border-left: 4px solid var(--primary-blue);
    }
    </style>
    """, unsafe_allow_html=True)

# --- COMPONENT INITIALIZATION ---
@st.cache_resource
def init_backend():
    llm = get_llm()
    return {
        "analyst": Analyst(),
        "critic": Critic(llm),
        "planner": Planner(llm),
        "proposer": Proposer(llm),
        "reviewer": Reviewer(phobert_model)
    }

cp = init_backend()

# --- SESSION MANAGEMENT ---
if 'target_level' not in st.session_state: st.session_state.target_level = None
if 'final_text' not in st.session_state: st.session_state.final_text = ""

# --- SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    st.markdown("### 🛠️ ENGINE CONTROL")
    st.caption("Cấu hình tham số Agentic")
    max_iters = st.slider("Max Loop Iterations", 5, 30, 20)
    
    st.markdown("---")
    st.markdown("### 📊 MODEL INFO")
    st.markdown("""
    - **Language Model:** Llama 3.1 8B
    - **Classifier:** PhoBERT-Base
    - **Task:** Readability Adaptation
    """)
    
    if st.button("RESET SESSION", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- MAIN INTERFACE ---
st.title("Vietnamese Readability Agent")
st.markdown("Hệ thống hiệu chỉnh cấp độ văn bản tự động sử dụng trí tuệ nhân tạo đa tác nhân.")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("#### 📥 INPUT STREAM")
    input_text = st.text_area("Source Text", height=280, 
                              placeholder="Nhập nội dung văn bản cần phân tích và biên tập...",
                              label_visibility="collapsed")
    
    # Token counter with progress bar
    tokens = phobert_model.tokenizer.encode(input_text)
    token_count = len(tokens)
    prog_color = "normal" if token_count <= 250 else "exception"
    st.progress(min(token_count / 256, 1.0))
    st.caption(f"Payload Size: {token_count} / 256 tokens (PhoBERT Context Limit)")

with right_col:
    st.markdown("#### ⚙️ ANALYSIS & CONTROL")
    if input_text and token_count <= 256:
        analysis = cp['analyst'].run(input_text)
        cur_lvl = float(analysis['label'])
        
        # Display Current Status Card
        b_class = f"badge-{int(cur_lvl)}"
        l_name = "TIỂU HỌC" if cur_lvl == 0.0 else "THCS" if cur_lvl == 1.0 else "THPT"
        
        st.markdown(f"""
            <div class="work-card">
                <span class="badge {b_class}">{l_name} (LEVEL {cur_lvl})</span>
                <p style="margin-top:15px; color:#475569;">Trạng thái hiện tại được xác định bởi Reviewer Agent.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎯 TARGET DEFINITION")
        st.caption("Chọn cấp độ mục tiêu để Agent bắt đầu quá trình lập kế hoạch:")
        
        btn_cols = st.columns(3)
        if btn_cols[0].button("TIỂU HỌC (0.0)", disabled=(cur_lvl == 0.0), use_container_width=True):
            st.session_state.target_level = 0.0
        if btn_cols[1].button("THCS (1.0)", disabled=(cur_lvl == 1.0), use_container_width=True):
            st.session_state.target_level = 1.0
        if btn_cols[2].button("THPT (2.0)", disabled=(cur_lvl == 2.0), use_container_width=True):
            st.session_state.target_level = 2.0

        if st.session_state.target_level is not None:
            st.markdown(f"Selected Target: **LEVEL {st.session_state.target_level}**")
            
            if st.button("EXECUTE ADAPTATION", type="primary", use_container_width=True):
                trace = []
                with st.status("🚀 Agentic Loop in progress...", expanded=True) as status:
                    # Initial Critique
                    st.write("Analyzing linguistic features...")
                    critique = cp['critic'].run(input_text, analysis)
                    
                    # Execution
                    final_result = run_loop(
                        text=input_text,
                        proposer=cp['proposer'],
                        reviewer=cp['reviewer'],
                        planner=cp['planner'],
                        current_level=cur_lvl,
                        target_level=st.session_state.target_level,
                        critique=critique,
                        trace=trace,
                        max_iter=max_iters
                    )
                    
                    # Final Validation
                    final_pred = phobert_model.predict(final_result)
                    if final_pred['label'] == st.session_state.target_level:
                        st.session_state.final_text = final_result
                        status.update(label="Process Completed Successfully", state="complete")
                    else:
                        st.session_state.final_text = final_result
                        status.update(label="Max Iterations Reached (Partial Success)", state="error")

# --- OUTPUT SECTION ---
if st.session_state.final_text:
    st.markdown("---")
    st.markdown("#### 📤 ADAPTED OUTPUT")
    st.markdown(f"""
        <div class="result-box">
            {st.session_state.final_text}
        </div>
    """, unsafe_allow_html=True)
    
    col_act1, col_act2 = st.columns([1, 5])
    if col_act1.button("COPY TEXT"):
        st.toast("Copied to clipboard")
    
    # Trace Log with Key Fix
    with st.expander("🛠️ SYSTEM LOGS (TRACE)"):
        for step in trace:
            s_name = str(step.get('step', 'N/A')).upper()
            it = step.get('iter', '-')
            st.markdown(f"**Step:** `{s_name}` | **Iteration:** `{it}`")
            
            if 'output' in step:
                st.json(step['output'])
            elif 'reason' in step:
                st.info(f"Terminated: {step['reason']}")
            else:
                st.write(step)

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Core System: Agentic Readability Transformation Engine | Developer Edition")