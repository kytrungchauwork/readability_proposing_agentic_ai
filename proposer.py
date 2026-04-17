from langchain_core.prompts import ChatPromptTemplate
from utils import safe_json_loads
import json


class Proposer:
    def __init__(self, llm):
        self.llm = llm

    def run(self, text, current_level, target_level, critique, direction):

        # 🔥 CLEAN CRITIQUE (tránh phá prompt)
        critique_str = json.dumps(critique, ensure_ascii=False)

        # 🔥 MAP DIRECTION → TEXT (LLM hiểu tốt hơn)
        if direction < 0:
            direction_rule = "Simplify aggressively (shorter sentence, simpler words, more concrete)"
        elif direction > 0:
            direction_rule = "Increase complexity (more abstract, more formal, longer structure)"
        else:
            direction_rule = "Keep similar complexity"

        prompt = ChatPromptTemplate.from_template("""
You are a readability transformation engine.

TASK:
Rewrite text from current level to target level.

LEVELS:
- 0.0 = TIỂU HỌC (very simple, short, concrete)
- 1.0 = THCS (medium)
- 2.0 = THPT (academic, abstract)

READABILITY SIGNALS (for guidance, not instruction):
{critique}

DIRECTION:
{direction_rule}

SEMANTIC CONSTRAINTS (VERY IMPORTANT):
- MUST preserve original meaning
- DO NOT change subject (e.g., AI must remain AI or equivalent)
- DO NOT introduce new entities
- DO NOT hallucinate
- DO NOT remove key information
- You can ONLY rewrite wording, NOT meaning

BAD EXAMPLE:
AI → trẻ em ❌

GOOD EXAMPLE:
AI → trí tuệ nhân tạo ✅

FORMAT RULES (STRICT):
- Output MUST be valid JSON
- DO NOT output code
- DO NOT output markdown
- DO NOT explain anything
- ALL strings must be valid JSON (no unescaped quotes)

OUTPUT FORMAT:
{{
  "rewrite": "string",
  "changes": ["string", "string"]
}}

INPUT:
Text: {text}
Current: {current}
Target: {target}
""")

        chain = prompt | self.llm

        res = chain.invoke({
            "text": text,
            "current": str(current_level),
            "target": str(target_level),
            "direction_rule": direction_rule,
            "critique": critique_str
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

        # 🔥 EXTRA GUARD: tránh rewrite rỗng
        if not data["rewrite"].strip():
            return {
                "rewrite": text,
                "changes": ["fallback: empty rewrite"]
            }

        return data