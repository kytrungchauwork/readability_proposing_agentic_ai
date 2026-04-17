import json
import re

def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except:
        # extract JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    return {
        "error": "invalid_json",
        "raw": text
    }