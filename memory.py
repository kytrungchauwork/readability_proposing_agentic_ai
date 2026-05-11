# memory.py
import json

class Memory:
    def __init__(self):
        self.history = []

    def add_iteration(self, iteration, text, detected_level, confidence, changes):
        self.history.append({
            "iter": iteration,
            "text": text,
            "level": detected_level,
            "conf": confidence,
            "changes": changes
        })

    def get_formatted_history(self):
        if not self.history:
            return "No previous attempts."
        
        formatted = []
        for h in self.history:
            # Lấy danh sách changes và đảm bảo nó là list
            raw_changes = h.get('changes', [])
            if not isinstance(raw_changes, list):
                raw_changes = [str(raw_changes)]
            
            # Chuyển đổi mọi phần tử trong changes thành chuỗi (để tránh lỗi join)
            safe_changes = []
            for c in raw_changes:
                if isinstance(c, dict):
                    # Nếu là dict, chuyển thành string JSON (ví dụ: {"old": "A", "new": "B"})
                    safe_changes.append(json.dumps(c, ensure_ascii=False))
                else:
                    # Nếu là các kiểu khác, ép kiểu về string
                    safe_changes.append(str(c))

            formatted.append(
                f"- Iter {h['iter']}: Level {h['level']} (Conf: {h['conf']:.2f}). "
                f"Text: '{h['text']}'. "
                f"Changes made: {', '.join(safe_changes)}"
            )
        return "\n".join(formatted)

    def is_stuck(self, current_text):
        # Kiểm tra nếu text mới trùng với bất kỳ text nào trong quá khứ (loại bỏ khoảng trắng)
        return any(h['text'].strip() == current_text.strip() for h in self.history)