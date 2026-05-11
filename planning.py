# planning.py
from langchain_core.prompts import ChatPromptTemplate
from utils import safe_json_loads
import json

class Planner:
    def __init__(self, llm):
        self.llm = llm

    def run(self, text, current_level, target_level, critique, history):
        # 1. NHẬN DIỆN TRẠNG THÁI BẾ TẮC
        stuck_count = history.count(f"Level {current_level}")
        
        stagnation_level = "NORMAL"
        if stuck_count >= 2: stagnation_level = "HIGH"
        if stuck_count >= 5: stagnation_level = "EXTREME"

        # 2. THIẾT LẬP RÀO CHẮN NGỮ NGHĨA (SEMANTIC CONSTRAINTS)
        semantic_constraints = """
        - CORE MEANING PRESERVATION: The rewritten text must convey 100% of the original information.
        - ENTITY LOCK: Do not change, remove, or hallucinate subjects, objects, or key entities.
        - TONE CONSISTENCY: Maintain the original intent (e.g., if it's a factual statement, keep it factual).
        - PARAPHRASE ONLY: Change the complexity of the words, NOT the meaning of the message.
        """

        # 3. XÁC ĐỊNH HƯỚNG ĐI CHI TIẾT
        if float(target_level) < float(current_level):
            direction_hint = """
            - GOAL: SIMPLIFY vocabulary while locking the core meaning.
            - Strategy: Find native Vietnamese synonyms for Sino-Vietnamese terms.
            - Structure: Shorten sentences by removing redundant adjectives, NOT by removing core facts.
            """
        else:
            direction_hint = """
            - GOAL: ELEVATE vocabulary while locking the core meaning.
            - Strategy: Use formal Sino-Vietnamese terminology that matches the exact context.
            - Structure: Combine sentences using formal conjunctions without adding new information.
            """

        prompt = ChatPromptTemplate.from_template("""
You are a VIETNAMESE LINGUISTIC MASTERMIND & SEMANTIC GUARDIAN. 
Your mission: Create a plan to reach Level {target} while ensuring the text remains 100% FAITHFUL to the original meaning.

### SEMANTIC RULES:
{semantic_constraints}

### CONTEXT:
- Original Text: "{text}"
- Current Measured Level: {current}
- Target Level: {target}
- Critique (Technical reasons for {current}): {critique}
- Failure History: {history}
- Stagnation Level: {stagnation_level}

### STRATEGY GUIDELINE:
{direction_hint}

### TASK:
1. **Identify Core Entities**: List the parts of the text that MUST NOT be changed.
2. **Vocabulary Overhaul**: Suggest synonyms that change complexity but NOT meaning.
3. **Structural Blueprint**: Plan a new structure that preserves all original facts.

### STRICT OUTPUT FORMAT (JSON ONLY):
{{
  "strategy_name": "string",
  "core_meaning_to_preserve": "Summary of the facts that must stay the same",
  "word_mappings": [
    {{
      "original": "string", 
      "replacement": "string", 
      "linguistic_reason": "string",
      "meaning_check": "How this replacement preserves the exact original meaning"
    }}
  ],
  "structural_instructions": "Detailed description of structural changes while keeping facts intact",
  "forbidden_elements": ["words/styles that failed before or that change the meaning"],
  "step_by_step_guidance": [
    "Step 1: Replace X with Y (ensure meaning is kept)",
    "Step 2: ..."
  ],
  "reasoning": "Technical explanation of the readability-meaning balance"
}}
""")
        
        chain = prompt | self.llm
        res = chain.invoke({
            "text": text,
            "current": str(current_level),
            "target": str(target_level),
            "critique": json.dumps(critique, ensure_ascii=False),
            "history": history,
            "direction_hint": direction_hint,
            "stagnation_level": stagnation_level,
            "semantic_constraints": semantic_constraints
        })
        
        return safe_json_loads(res.content)