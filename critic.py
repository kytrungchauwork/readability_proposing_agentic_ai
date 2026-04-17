from langchain_core.prompts import ChatPromptTemplate
from utils import safe_json_loads

class Critic:
    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template("""
You are a READABILITY FEATURE EXTRACTOR.

TASK:
Extract ONLY evidence that justifies the predicted readability level.

DO NOT:
- evaluate correctness
- suggest improvements
- add opinions

LEVELS:
- 0.0 = TIỂU HỌC
- 1.0 = THCS
- 2.0 = THPT

INPUT:
Text: {text}
Predicted level: {label}
Confidence: {confidence}

STRICT OUTPUT RULES:
- Output MUST be valid JSON
- NO markdown
- NO extra text

OUTPUT FORMAT:
{{
  "level": "{label}",
  "signals": [
    {{
      "type": "lexical",
      "description": "string",
      "strength": 0.0
    }},
    {{
      "type": "syntax",
      "description": "string",
      "strength": 0.0
    }},
    {{
      "type": "semantic",
      "description": "string",
      "strength": 0.0
    }}
  ],
  "summary": "string"
}}
""")

    def run(self, text, analysis):
        chain = self.prompt | self.llm

        res = chain.invoke({
            "text": text,
            "label": str(analysis["label"]),
            "confidence": str(analysis["confidence"])
        })

        return safe_json_loads(res.content)