# ⚡ Vietnamese Readability Agent Pro (VRAP)

A Multi-Agent System for Automated Vietnamese Readability Transformation.

VRAP is an advanced Agentic AI system designed to adapt the readability level of Vietnamese text. By leveraging a hybrid architecture of Large Language Models (Llama 3.1 via Groq) and specialized Deep Learning models (PhoBERT), it transforms text into three standard academic levels: Elementary (0.0), Junior High (1.0), and High School (2.0).

---

## 🚀 Key Features

- Multi-Agent Architecture: Orchestrates specialized agents (Planner, Proposer, and Reviewer) to handle complex linguistic tasks.
- Semantic Guardian: A strict planning constraint system that ensures the rewritten text preserves 100% of the original meaning and core entities.
- Self-Correction Loop: An automated iterative engine that runs up to 20 iterations to satisfy the objective validation of the PhoBERT classifier.
- Technical UI: A professional Streamlit-based dashboard featuring real-time token tracking and detailed execution "Trace Logs."

---

## 🛠️ System Architecture

The workflow follows a rigorous linguistic pipeline:
1. Analyst Agent (PhoBERT): Provides the initial readability baseline.
2. Critic Agent (Llama 3.1): Extracts linguistic evidence (lexical, syntax, semantic) justifying the current level.
3. Planner Agent (Llama 3.1): The "Chief Editor" that creates a detailed blueprint, mapping vocabulary changes and structural adjustments while locking core entities.
4. Proposer Agent (Llama 3.1): Executes the rewrite based on the Planner's instructions.
5. Reviewer Agent (PhoBERT): Acts as an objective judge to validate if the target level has been reached.

---

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher.
- CUDA-enabled GPU (Optional, PhoBERT runs on CPU but GPU is recommended for speed).

### 2. Install Dependencies
Run the following command to install all required libraries:

bash
pip install torch torchvision transformers langchain langchain-openai streamlit sentencepiece


### 3. API Configuration
Open llm.py and insert your Groq API Key:

python
def get_llm():
 return ChatOpenAI(
 openai_api_key="YOUR_GROQ_API_KEY",
 base_url="https://api.groq.com/openai/v1",
 model="llama-3.1-8b-instant",
 temperature=0.2
 )


---

## 🖥️ How to Run

### Web Interface (Recommended)
Launch the professional dashboard using Streamlit:

bash
streamlit run app.py

The application will be available at http://localhost:8501.

### Command Line Interface (CLI)
For quick testing via terminal:

bash
python main.py


---

## 📖 Usage Guide

1. Input Stream: Paste your text into the "Source Text" area. Note the 256-token limit (enforced by PhoBERT's context window).
2. Analyze: The system automatically detects the current level.
3. Target Definition: Select your desired output level (Elementary, Junior High, or High School).
4. Execution: Click EXECUTE ADAPTATION. The status bar will show the Agent's real-time thinking process.
5. Review: Once successful, the result appears in the black output box. You can inspect the "System Logs" to see the reasoning behind each iteration.

---

## ⚠️ Technical Notes

- Token Constraints: PhoBERT is optimized for a 256-token context. Input exceeding this will be flagged to prevent inaccurate classification.
- Rate Limiting: To comply with Groq API's free tier limits, the system includes hardcoded time.sleep() intervals. Do not remove these to avoid 429 RateLimitErrors.
- First Run: On the first execution, the system will download the phobert-readability-scale model (~600MB) from HuggingFace.

---

## 🤝 Contribution
This project is developed for research into Multi-Agent NLP applications for the Vietnamese language. Contributions to the Planning logic or Semantic Guardrail prompts are welcome.

---
VRAP Core System | Developer Edition v1.0