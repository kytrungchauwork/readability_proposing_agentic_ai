# 🚀 Agentic Readability AI (Vietnamese)

A **multi-agent AI system** for transforming Vietnamese text across readability levels (Tiểu học, THCS, THPT), powered by a **PhoBERT-based classifier** and an **iterative rewrite–validation loop**.

---

## 🧠 Overview

This project builds an **agentic NLP pipeline** where multiple agents collaborate to:
- Analyze text readability
- Explain why a level is assigned
- Rewrite text to a target level
- Validate the result using a model
- Iterate until convergence

👉 The system enforces **semantic preservation**, meaning the rewritten text keeps the original meaning while changing only readability.

---

## 🏗️ Architecture

```
Input Text
   ↓
🧩 Analyst (PhoBERT)
   → Predict readability level
   ↓
🔍 Critic (LLM)
   → Extract linguistic signals (lexical, syntax, semantic)
   ↓
✍️ Proposer (LLM)
   → Rewrite text toward target level
   ↓
✅ Reviewer (PhoBERT)
   → Validate readability level
   ↓
🔁 Loop until convergence
```

---

## 🤖 Agents

### 1. Analyst
- Uses PhoBERT readability model
- Outputs:
  - `label` (0.0 / 1.0 / 2.0)
  - `confidence`

### 2. Critic
- Explains why a readability level is assigned
- Extracts signals:
  - Lexical (word difficulty)
  - Syntax (sentence complexity)
  - Semantic (abstraction level)

### 3. Proposer
- Rewrites text based on:
  - Current level
  - Target level
  - Critic signals
- Enforces:
  - ✅ Preserve meaning
  - ❌ No entity change (AI ≠ trẻ em)
  - ❌ No hallucination

### 4. Reviewer
- Uses PhoBERT to:
  - Validate rewritten text
  - Check if target level is reached

---

## 🔁 Loop Strategy

- Dynamic direction:
```python
direction = target_level - current_level
```

- Soft match condition:
  - Accept if:
    - Level difference ≤ 0.3
    - Confidence > 0.6

- Early stopping:
  - No change in rewrite
  - Max iterations reached

---

## ✨ Features

- 🔄 Multi-agent iterative refinement  
- 🧠 PhoBERT-based evaluation  
- 🔒 Semantic-preserving rewriting  
- ⚠️ Robust handling of invalid JSON from LLM  
- 🧾 Full trace logging for debugging  
- 🎯 Soft convergence instead of strict matching  

---

## 📦 Model (PhoBERT)

This project uses a fine-tuned PhoBERT model for readability classification:

👉 https://huggingface.co/VMSR-Lab/phobert-readability-scale-with-small-range-dataset-classification

### Labels:
- `0.0` → Tiểu học  
- `1.0` → THCS  
- `2.0` → THPT  

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/agentic-readability-ai.git
cd agentic-readability-ai

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

Example:

Input:
```
AI đang thay đổi cách con người học tập và làm việc hiệu quả hơn.
```

Target:
```
0.0 (Tiểu học)
```

Output:
```
Máy tính giúp con học và làm việc tốt hơn.
```

---

## 📊 Example Trace

```json
{
  "iteration": 2,
  "proposer": "AI đang giúp con người học và làm việc tốt hơn.",
  "reviewer": {
    "detected_level": 1.0,
    "confidence": 0.63
  }
}
```

---

## 📁 Project Structure

```
agentic-readability-ai/
│
├── main.py
├── analyst.py
├── critic.py
├── proposer.py
├── reviewer.py
├── utils.py
├── llm.py
├── phobert_singleton.py
│
├── README.md
```


---

## ⚠️ Limitations

- LLM may output invalid JSON → handled with fallback  
- Hard to reach extreme levels perfectly (0.0 ↔ 2.0)  
- Semantic constraint limits aggressive rewriting  

---

## 🚀 Future Work

- Deploy API (FastAPI)  
- Build UI (Gradio / Hugging Face Space)  
- Improve semantic similarity scoring  
- Fine-tune proposer model  

---

## 👨‍💻 Author

Built as an **Agentic AI system** combining:
- LLM reasoning  
- PhoBERT classification  
- Iterative feedback loop  

---

## ⭐ If you find this useful

Give it a ⭐ on GitHub!
