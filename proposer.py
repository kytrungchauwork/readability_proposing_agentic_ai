import json
from langchain_core.prompts import ChatPromptTemplate
from utils import safe_json_loads

class Proposer:
    def __init__(self, llm):
        self.llm = llm

    def run(self, text, current_level, target_level, critique, direction, plan, history):
        # 🔥 CLEAN CRITIQUE & PLAN
        critique_str = json.dumps(critique, ensure_ascii=False)
        plan_str = json.dumps(plan, ensure_ascii=False) if isinstance(plan, dict) else str(plan)

        # 🔥 MAP DIRECTION → TEXT
        if direction < 0:
            direction_rule = "SIMPLIFY AGGRESSIVELY. Replace formal Hán-Việt words with common native Vietnamese words. Break down abstract concepts."
        elif direction > 0:
            direction_rule = "INCREASE COMPLEXITY. Use more academic, formal, and Sino-Vietnamese terminology. Use longer, nested sentence structures."
        else:
            direction_rule = "Keep similar complexity"

        prompt = ChatPromptTemplate.from_template("""
You are an expert Vietnamese Readability Editor.

TASK:
Rewrite the text to reach the EXACT target level.

LEVEL DEFINITIONS & EXAMPLES:
- 0.0 (TIỂU HỌC): Very simple. Uses only common daily words. No abstract nouns.
  *Example: "Máy tính thông minh giúp mọi người làm việc nhanh hơn."*
- 1.0 (THCS): Medium difficulty. Familiar topics but uses some formal words. 
  *Example: "Công nghệ mới đang làm thay đổi cách con người mua bán và làm việc trên thế giới."*
- 2.0 (THPT): Academic/Formal. High density of Sino-Vietnamese (Hán-Việt) words. Complex structures.
  *Example: "Hệ thống trí tuệ nhân tạo đang tác động sâu sắc đến cấu trúc kinh tế toàn cầu."*

STRATEGY PLAN FROM PLANNER (MANDATORY):
{plan}

HISTORY OF ATTEMPTS (DO NOT REPEAT):
{history}

DIRECTION: {direction_rule}

SEMANTIC CONSTRAINTS:
- Preserve original meaning.
- DO NOT change the subject.
- DO NOT hallucinate new info.
- ONLY change wording and sentence structure.

FORMAT RULES:
- Output ONLY valid JSON.
- No markdown, no explanation.

OUTPUT FORMAT:
{{
  "rewrite": "string",
  "changes": ["list of specific word replacements made"]
}}

INPUT:
Text: {text}
Current Level: {current}
Target Level: {target}
""")

        chain = prompt | self.llm

        res = chain.invoke({
            "text": text,
            "current": str(current_level),
            "target": str(target_level),
            "direction_rule": direction_rule,
            "critique": critique_str,
            "plan": plan_str,
            "history": history
        })

        # =========================
        # 🔥 SAFE PARSE + FALLBACK
        # =========================
        data = safe_json_loads(res.content)

        if not isinstance(data, dict) or "rewrite" not in data:
            return {
                "rewrite": text,
                "changes": ["fallback: invalid LLM output"]
            }

        if not data["rewrite"].strip():
            return {
                "rewrite": text,
                "changes": ["fallback: empty rewrite result"]
            }

        return data