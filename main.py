# main.py
import json
import time
from llm import get_llm
from analyst import Analyst
from critic import Critic
from proposer import Proposer
from reviewer import Reviewer
from planning import Planner
from memory import Memory
from phobert_singleton import phobert_model

# =========================
# LOOP ENGINE (AGENTIC VERSION)
# =========================
def run_loop(
    text,
    proposer,
    reviewer,
    planner,
    current_level,
    target_level,
    critique,
    trace,
    max_iter=20 # <--- Mặc định tăng lên 20 lần
):
    # Khởi tạo bộ nhớ cho Agent trong lượt chạy này
    memory = Memory()
    current_text = text
    
    for i in range(max_iter):
        iteration = i + 1
        print(f"\n🔁 ITERATION {iteration}/{max_iter}")

        # 1. Lấy lịch sử từ Memory
        full_history = memory.get_formatted_history()
        
        # --- TỐI ƯU HÓA HISTORY (SLIDING WINDOW) ---
        # Nếu lịch sử quá dài (trên 7 dòng), chỉ lấy 7 dòng cuối cùng để tiết kiệm Token và tránh làm loãng context
        history_lines = full_history.split('\n')
        if len(history_lines) > 7:
            history_context = "--- (Older history truncated) ---\n" + "\n".join(history_lines[-7:])
        else:
            history_context = full_history

        # 2. PLANNING: Agent lập kế hoạch
        plan = planner.run(
            text=current_text,
            current_level=current_level,
            target_level=target_level,
            critique=critique,
            history=history_context # Gửi history đã cắt tỉa
        )
        
        print(f"[PLANNER]: {plan.get('strategy_name', 'No strategy name')}")
        mappings = plan.get('word_mappings', [])
        if mappings:
            print(f"   Mapping: {', '.join([m['original'] + ' -> ' + m['replacement'] for m in mappings])}")
            
        trace.append({
            "step": "planner",
            "iter": iteration,
            "output": plan
        })

        # 🔥 NGHỈ 3 GIÂY ĐỂ TRANH RATE LIMIT
        time.sleep(3)

        # 3. PROPOSER: Thực hiện rewrite dựa trên Plan và Memory
        direction = float(target_level) - float(current_level)
        
        proposal = proposer.run(
            text=current_text,
            current_level=current_level,
            target_level=target_level,
            critique=critique,
            direction=direction,
            plan=plan,
            history=history_context # Gửi history đã cắt tỉa
        )

        rewritten = proposal.get("rewrite", current_text)
        print(f"[PROPOSER]: {rewritten}")

        trace.append({
            "step": "proposer",
            "iter": iteration,
            "output": proposal
        })

        # 4. STUCK CHECK
        if memory.is_stuck(rewritten):
            print("⚠️ WARNING: Agent lặp lại văn bản cũ. Đang chờ Planner đổi chiến thuật...")

        # 5. REVIEWER (PhoBERT local)
        review = reviewer.run(rewritten, target_level)
        detected = float(review.get("detected_level", 999))
        confidence = float(review.get("confidence", 0))

        print(f"[REVIEW]: Detected Level {detected} (Target: {target_level}) | Conf: {confidence:.4f}")

        trace.append({
            "step": "reviewer",
            "iter": iteration,
            "output": review
        })

        # 6. CẬP NHẬT BỘ NHỚ (Vẫn lưu đầy đủ vào RAM, chỉ cắt tỉa khi gửi cho LLM)
        memory.add_iteration(
            iteration=iteration,
            text=rewritten,
            detected_level=detected,
            confidence=confidence,
            changes=proposal.get("changes", [])
        )

        # 7. ĐIỀU KIỆN DỪNG
        if detected == float(target_level):
            print(f"\n✅ SUCCESS: Target level {target_level} đạt được ở lần lặp thứ {iteration}!")
            trace.append({"step": "stop", "iter": iteration, "reason": "target_reached"})
            return rewritten

        # Cập nhật trạng thái
        current_level = detected
        current_text = rewritten
        
        # 🔥 NGHỈ 2 GIÂY TRƯỚC KHI SANG VÒNG TIẾP THEO
        time.sleep(2)

    print(f"\n❌ FAILED: Sau {max_iter} lần thử vẫn không thể đạt nhãn mục tiêu.")
    trace.append({"step": "max_iter_reached", "final_level": current_level})
    return current_text


# =========================
# MAIN SYSTEM
# =========================
def run_system(text, target_level=None):
    llm = get_llm()

    analyst = Analyst()
    critic = Critic(llm)
    planner = Planner(llm)
    proposer = Proposer(llm)
    reviewer = Reviewer(phobert_model)

    trace = []

    print("\n=== STARTING SYSTEM ===")
    analysis = analyst.run(text)
    print(f"[ANALYST]: Mức độ ban đầu: {analysis['label']}")
    trace.append({"step": "analyst", "output": analysis})

    critique = critic.run(text, analysis)
    trace.append({"step": "critic", "output": critique})
    time.sleep(2)

    if target_level is None:
        print("\n👉 Chọn mức độ mục tiêu (0.0: Tiểu học, 1.0: THCS, 2.0: THPT)")
        try:
            target_level = float(input("Nhập mức độ: "))
        except ValueError:
            target_level = 1.0
        
    trace.append({"step": "target_selected", "output": target_level})

    # Chạy vòng lặp tối đa 20 lần
    final_text = run_loop(
        text=text,
        proposer=proposer,
        reviewer=reviewer,
        planner=planner,
        current_level=analysis["label"],
        target_level=target_level,
        critique=critique,
        trace=trace,
        max_iter=20 # <--- Chốt 20 lần
    )

    final_pred = phobert_model.predict(final_text)
    
    result = {
        "initial_level": analysis["label"],
        "target_level": target_level,
        "final_text": final_text,
        "final_level": final_pred["label"],
        "status": "success" if final_pred["label"] == target_level else "failed",
        "trace": trace
    }

    print("\n=== FINAL SUMMARY ===")
    print(f"Path: {analysis['label']} ➔ {result['final_level']} (Target: {target_level})")
    print(f"Final Text: {final_text}")

    return result


if __name__ == "__main__":
    sample_text = "Hệ thống trí tuệ nhân tạo đang thay đổi bộ mặt của nền kinh tế toàn cầu."
    run_system(sample_text)