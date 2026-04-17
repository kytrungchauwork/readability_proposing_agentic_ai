import json
from llm import get_llm
from analyst import Analyst
from critic import Critic
from proposer import Proposer
from reviewer import Reviewer
from phobert_singleton import phobert_model


# =========================
# LOOP ENGINE (FINAL)
# =========================
def run_loop(
    text,
    proposer,
    reviewer,
    current_level,
    target_level,
    critique,
    trace,
    max_iter=5
):

    prev_text = None
    stuck_count = 0

    for i in range(max_iter):
        print(f"\n🔁 ITERATION {i+1}")

        trace.append({
            "step": "iteration_input",
            "iter": i + 1,
            "text": text,
            "current_level": current_level
        })

        # ======================
        # DIRECTION
        # ======================
        direction = float(target_level) - float(current_level)

        # ======================
        # PROPOSER
        # ======================
        proposal = proposer.run(
            text=text,
            current_level=current_level,
            target_level=target_level,
            critique=critique,
            direction=direction
        )

        rewritten = proposal.get("rewrite", text)

        print("\n[PROPOSER]")
        print(rewritten)

        trace.append({
            "step": "proposer",
            "iter": i + 1,
            "output": proposal
        })

        # ======================
        # NO CHANGE CHECK
        # ======================
        if rewritten == prev_text:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= 2:
            print("\n⚠️ STUCK → cannot improve further")

            trace.append({
                "step": "stuck_stop",
                "iter": i + 1,
                "reason": "no_progress_multiple_times"
            })

            return rewritten

        prev_text = rewritten

        # ======================
        # REVIEWER
        # ======================
        review = reviewer.run(rewritten, target_level)

        print("\n[REVIEW]")
        print(review)

        trace.append({
            "step": "reviewer",
            "iter": i + 1,
            "output": review
        })

        detected = float(review.get("detected_level", 999))
        confidence = float(review.get("confidence", 0))

        # ======================
        # SOFT MATCH
        # ======================
        if abs(detected - float(target_level)) <= 0.3 and confidence > 0.6:
            print("\n✅ MATCH (SOFT)")

            trace.append({
                "step": "stop",
                "iter": i + 1,
                "reason": "soft_match"
            })

            return rewritten

        # ======================
        # UPDATE STATE
        # ======================
        current_level = detected
        text = rewritten

    print("\n⚠️ MAX ITER REACHED")

    trace.append({
        "step": "max_iter_reached"
    })

    return text


# =========================
# MAIN SYSTEM
# =========================
def run_system(text):

    llm = get_llm()

    analyst = Analyst()
    critic = Critic(llm)
    proposer = Proposer(llm)
    reviewer = Reviewer(phobert_model)

    trace = []

    print("\n=== INPUT ===")
    print(text)

    trace.append({
        "step": "input",
        "text": text
    })

    # =========================
    # ANALYST
    # =========================
    analysis = analyst.run(text)

    print("\n[ANALYST]")
    print(analysis)

    trace.append({
        "step": "analyst",
        "output": analysis
    })

    # =========================
    # CRITIC
    # =========================
    critique = critic.run(text, analysis)

    print("\n[CRITIC]")
    print(critique)

    trace.append({
        "step": "critic",
        "output": critique
    })

    # =========================
    # USER TARGET
    # =========================
    print("\n👉 CRITIC DONE")
    print("Select target level:")
    print("0.0 = TIỂU HỌC")
    print("1.0 = THCS")
    print("2.0 = THPT")

    target_level = float(input("Enter target level: "))

    print("\n[TARGET LEVEL]")
    print(target_level)

    trace.append({
        "step": "target_selected",
        "output": target_level
    })

    # =========================
    # LOOP
    # =========================
    final_text = run_loop(
        text=text,
        proposer=proposer,
        reviewer=reviewer,
        current_level=analysis["label"],
        target_level=target_level,
        critique=critique,
        trace=trace,
        max_iter=5
    )

    # =========================
    # FINAL EVALUATION
    # =========================
    final_pred = phobert_model.predict(final_text)

    final_level = final_pred["label"]
    final_conf = final_pred["confidence"]

    print("\n=== FINAL SUMMARY ===")
    print(f"Final text: {final_text}")
    print(f"Final level: {final_level} (confidence={final_conf:.4f})")
    print(f"Target level: {target_level}")
    print(f"Progress: {analysis['label']} → {final_level}")

    if abs(final_level - float(target_level)) <= 0.3:
        print("✅ SUCCESS: đạt gần target level")
        status = "success"
    else:
        print("⚠️ FAILED: không đạt target level")
        status = "failed"

    # =========================
    # FINAL OUTPUT
    # =========================
    result = {
        "analysis": analysis,
        "critique": critique,
        "target_level": target_level,
        "final_text": final_text,
        "final_level": final_level,
        "final_confidence": final_conf,
        "status": status,
        "trace": trace
    }

    return result


# =========================
# RUN TEST
# =========================
if __name__ == "__main__":

    text = "Em đi học mỗi ngày."

    result = run_system(text)

    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))