"""
Groq LLM RAG Protocol Briefing Engine
Grounds clinical load management briefings in fresh medical-protocol.md and briefing-template.md.
Features primary/backup model cascade, exponential backoff for rate limits,
clinical guardrail verification, and deterministic rule fallbacks.
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Tuple
from guardrails.validator import verify_clinical_scope, check_prompt_injection

PRIMARY_MODEL = "llama-3.3-70b-versatile"
BACKUP_MODEL = "llama-3.1-8b-instant"

PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "medical-protocol.md")
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "briefing-template.md")


def load_fresh_protocols() -> Tuple[str, str]:
    """Reads medical-protocol.md and briefing-template.md fresh from disk on every evaluation."""
    protocol_text = ""
    template_text = ""
    
    if os.path.exists(PROTOCOL_PATH):
        with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
            protocol_text = f.read()
            
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_text = f.read()
            
    return protocol_text, template_text


def generate_fallback_briefing(
    player: Dict[str, Any],
    ml_risk_class: str,
    risk_prob: float
) -> Dict[str, Any]:
    """
    Deterministic sports-medicine clinical rule engine grounded directly in medical-protocol.md.
    Used when Groq API is unavailable, rate-limited, or for offline evaluation.
    """
    protocol_text, template_text = load_fresh_protocols()
    
    acwr = float(player.get("acwr", 1.0))
    sleep = float(player.get("avg_sleep_hours", 8.0))
    wellness = float(player.get("wellness_score", 8.0))
    soreness = float(player.get("soreness", 2.0))
    fatigue = float(player.get("fatigue", 2.0))
    hsr_spike = float(player.get("hsr_spike_pct", 0.0))
    days_since_injury = float(player.get("days_since_last_injury", 365))
    prev_injuries = int(player.get("previous_injuries_count", 0))
    last_injury = player.get("last_injury_type", "None")
    
    triggered_clauses = []
    recommended_status = "Cleared"
    pitch_restrictions = "Unrestricted team training and competitive fixture selection."
    gym_restrictions = "Standard maintenance resistance and core strength programming."
    recovery_modalities = "Routine hydration, cold plunge post-session, and dynamic stretching."
    rationale_sentence = "Telemetry within optimal physiological boundaries; cleared for full competition."
    pathophysiology = "All cardiovascular, sleep, and biomechanical telemetry fall within safe homeostatic zones."
    
    # Clause 1: ACWR Evaluation
    if acwr > 1.50:
        triggered_clauses.append("Clause 1: Critical ACWR Spike (> 1.50) - Exponential soft-tissue injury risk.")
        recommended_status = "Rest Mandated"
        pitch_restrictions = "Pitch training completely suspended for 48 hours; off-feet cardiovascular work only (wattbike/pool)."
        gym_restrictions = "Isometric non-weight-bearing exercises; no eccentric overload."
        recovery_modalities = "Cryotherapy, contrast baths, manual myofascial release with Lead Physiotherapist."
        rationale_sentence = f"Critical ACWR spike ({acwr:.2f} > 1.50) triggers mandatory 48-hour off-pitch recovery protocol."
        pathophysiology = f"Acute load significantly exceeds chronic fitness base (ACWR {acwr:.2f}), creating severe collagen microtrauma vulnerability."
    elif acwr > 1.30:
        triggered_clauses.append("Clause 1: Elevated ACWR Caution Zone (1.30 - 1.50) - Elevated strain probability.")
        if recommended_status != "Rest Mandated":
            recommended_status = "Modified Training"
            pitch_restrictions = "Reduce session volume by 25%; strictly restrict maximal sprint acceleration drills (>25 km/h)."
            gym_restrictions = "Reduce eccentric hamstring and groin loads by 30%."
            rationale_sentence = f"Elevated ACWR ({acwr:.2f}) mandates 25% load curtailment and high-speed running restrictions."
            pathophysiology = f"Workload in cautionary accumulation zone ({acwr:.2f}); neuromuscular fatigue impairs tendon stiffness."
    elif acwr < 0.80:
        triggered_clauses.append("Clause 1: Under-training / Deconditioning Zone (ACWR < 0.80).")
        rationale_sentence = f"Sub-optimal ACWR ({acwr:.2f}); supplementary conditioning indicated to avert detraining."

    # Clause 2: Sleep Evaluation
    if sleep < 6.0:
        triggered_clauses.append("Clause 2: Severe Sleep Deprivation (< 6.0h) - Neuro-mechanical impairment and 1.7x injury vulnerability.")
        if recommended_status == "Cleared":
            recommended_status = "Modified Training"
        pitch_restrictions += " Plyometrics and collision contact barred."
        recovery_modalities += " Supplementary 45-minute afternoon sleep protocol and magnesium supplementation."
        pathophysiology += f" Critical sleep debt ({sleep}h) elevates systemic cortisol and degrades neuromuscular motor firing."
    elif sleep < 7.0:
        triggered_clauses.append("Clause 2: Moderate Sleep Debt (6.0 - 7.0h) - Compromised glycogen resynthesis.")

    # Clause 3: Wellness & Soreness Triggers
    if soreness >= 7.0:
        triggered_clauses.append(f"Clause 3: Acute Focal Muscle Soreness ({soreness}/10) - High risk of myofascial strain.")
        if recommended_status == "Cleared":
            recommended_status = "Modified Training"
        pitch_restrictions = "Restricted to non-impact tactical positioning drills."
        recovery_modalities += " Mandatory ultrasound screening of sore muscle group."
        rationale_sentence = f"Acute muscle soreness ({soreness}/10) requires localized physio examination and drill modification."
    if fatigue >= 7.0:
        triggered_clauses.append(f"Clause 3: High Neuromuscular Fatigue ({fatigue}/10) - Autonomic exhaustion.")

    # Clause 4: HSR GPS Spike
    if hsr_spike > 20.0:
        triggered_clauses.append(f"Clause 4: GPS High-Speed Running Spike (+{hsr_spike:.1f}% week-on-week) - Hamstring strain alert.")
        if recommended_status == "Cleared":
            recommended_status = "Modified Training"

    # Clause 5: Prior Injury Multipliers
    if days_since_injury < 90 and prev_injuries > 0 and acwr > 1.25:
        triggered_clauses.append(f"Clause 5: Prior Injury Multiplier - History of {last_injury} within {int(days_since_injury)} days elevates risk.")
        if recommended_status != "Rest Mandated":
            recommended_status = "Modified Training"
            rationale_sentence = f"Prior {last_injury} ({int(days_since_injury)} days ago) combined with elevated load ({acwr:.2f}) necessitates modified training."

    if not triggered_clauses:
        triggered_clauses.append("All metrics satisfy Tier 1 Clearance requirements under Medical Protocol Section 6.")

    # Format into template
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    briefing_md = template_text
    briefing_md = briefing_md.replace("{{PLAYER_NAME}}", player.get("name", "Unknown Player"))
    briefing_md = briefing_md.replace("{{PLAYER_ID}}", player.get("player_id", "PLY-000"))
    briefing_md = briefing_md.replace("{{POSITION}}", player.get("position", "Unknown"))
    briefing_md = briefing_md.replace("{{EVALUATION_DATE}}", now_str)
    briefing_md = briefing_md.replace("{{ACWR}}", f"{acwr:.2f}")
    briefing_md = briefing_md.replace("{{SLEEP_HOURS}}", f"{sleep:.1f}")
    briefing_md = briefing_md.replace("{{WELLNESS_SCORE}}", f"{wellness:.1f}")
    briefing_md = briefing_md.replace("{{FATIGUE}}", f"{fatigue:.1f}")
    briefing_md = briefing_md.replace("{{SORENESS}}", f"{soreness:.1f}")
    briefing_md = briefing_md.replace("{{STRESS}}", f"{player.get('stress', 3.0):.1f}")
    briefing_md = briefing_md.replace("{{MOOD}}", f"{player.get('mood', 8.0):.1f}")
    briefing_md = briefing_md.replace("{{HSR_METERS}}", f"{player.get('high_speed_running_m', 450.0):.0f}")
    briefing_md = briefing_md.replace("{{HSR_VARIANCE}}", f"{hsr_spike:+.1f}")
    briefing_md = briefing_md.replace("{{INJURY_HISTORY_SUMMARY}}", f"{prev_injuries} previous events; last: {last_injury} ({int(days_since_injury)}d ago)")
    briefing_md = briefing_md.replace("{{ML_RISK_CLASS}}", ml_risk_class)
    briefing_md = briefing_md.replace("{{RISK_PROBABILITY}}", f"{risk_prob * 100:.1f}")
    briefing_md = briefing_md.replace("{{RISK_VECTORS}}", f"ACWR={acwr:.2f}, Sleep={sleep}h, Soreness={soreness}/10, HSR Variance={hsr_spike:+.1f}%")
    briefing_md = briefing_md.replace("{{TRIGGERED_CLAUSES}}", "\n".join([f"- {c}" for c in triggered_clauses]))
    briefing_md = briefing_md.replace("{{PATHOPHYSIOLOGY_RATIONALE}}", pathophysiology)
    briefing_md = briefing_md.replace("{{RECOMMENDED_STATUS}}", recommended_status)
    briefing_md = briefing_md.replace("{{PITCH_RESTRICTIONS}}", pitch_restrictions)
    briefing_md = briefing_md.replace("{{GYM_RESTRICTIONS}}", gym_restrictions)
    briefing_md = briefing_md.replace("{{RECOVERY_MODALITIES}}", recovery_modalities)
    briefing_md = briefing_md.replace("{{ONE_SENTENCE_DIRECTIVE}}", rationale_sentence)

    return {
        "player_id": player.get("player_id"),
        "player_name": player.get("name"),
        "ml_risk_class": ml_risk_class,
        "risk_probability": risk_prob,
        "recommended_status": recommended_status,
        "protocol_rationale": rationale_sentence,
        "full_briefing_markdown": briefing_md,
        "decided_by": "Fallback Rule",
        "model_used": "Deterministic Rule Engine (Grounded in medical-protocol.md)",
        "triggered_clauses": triggered_clauses
    }


def query_groq_rag(
    player: Dict[str, Any],
    ml_risk_class: str,
    risk_prob: float,
    api_key: str
) -> Dict[str, Any]:
    """
    Executes protocol-grounded RAG via Groq API.
    Handles rate-limiting with exponential backoff (2s, 4s, 8s), model cascade
    (llama-3.3-70b-versatile -> llama-3.1-8b-instant), guardrail checks,
    and falls back gracefully to the deterministic rule engine.
    """
    if not api_key or not isinstance(api_key, str) or not api_key.strip().startswith("gsk_"):
        # No valid API key provided -> use fallback rule engine
        res = generate_fallback_briefing(player, ml_risk_class, risk_prob)
        res["decided_by"] = "Fallback Rule"
        return res

    try:
        from groq import Groq
    except ImportError:
        res = generate_fallback_briefing(player, ml_risk_class, risk_prob)
        res["decided_by"] = "Fallback Rule"
        return res

    protocol_text, template_text = load_fresh_protocols()
    
    system_prompt = f"""You are the Elite Sports Medicine Protocol Briefing AI for a professional football club.
You assist the Club Doctor and Lead Physiotherapist in evaluating player telemetry and formulating load management plans.

NON-NEGOTIABLE GUARDRAIL RULES:
1. Ground all recommendations strictly in the official club medical protocol provided below.
2. DO NOT formulate definitive imaging or anatomical diagnoses (e.g. do not say 'confirmed ACL tear'). Only evaluate statistical risk probability and physical tolerance.
3. DO NOT prescribe prescription pharmaceuticals. Recommend only training load adjustments, physical therapy, and restorative modalities.
4. Output MUST conform to JSON with keys:
   - "recommended_status": "Cleared" | "Modified Training" | "Rest Mandated"
   - "protocol_rationale": "One concise clinical sentence explaining the decision."
   - "triggered_clauses": ["Clause X...", "Clause Y..."]
   - "pitch_restrictions": "Specific pitch-based drill restrictions"
   - "gym_restrictions": "Specific gym / lifting restrictions"
   - "recovery_modalities": "Prescribed physical therapy / recovery"
   - "pathophysiology_summary": "Brief physiological explanation"

OFFICIAL MEDICAL PROTOCOL:
{protocol_text}
"""

    user_payload = {
        "player_id": player.get("player_id"),
        "name": player.get("name"),
        "position": player.get("position"),
        "acwr": player.get("acwr"),
        "avg_sleep_hours": player.get("avg_sleep_hours"),
        "wellness_score": player.get("wellness_score"),
        "fatigue": player.get("fatigue"),
        "soreness": player.get("soreness"),
        "stress": player.get("stress"),
        "mood": player.get("mood"),
        "acute_load_au": player.get("acute_load_au"),
        "chronic_load_au": player.get("chronic_load_au"),
        "high_speed_running_m": player.get("high_speed_running_m"),
        "hsr_spike_pct": player.get("hsr_spike_pct"),
        "previous_injuries_count": player.get("previous_injuries_count"),
        "last_injury_type": player.get("last_injury_type"),
        "days_since_last_injury": player.get("days_since_last_injury"),
        "ml_risk_class": ml_risk_class,
        "ml_risk_probability": f"{risk_prob * 100:.1f}%"
    }

    user_prompt = f"""Evaluate this player's telemetry against the medical protocol and output the clinical briefing JSON:
{json.dumps(user_payload, indent=2)}"""

    # Retry backoff schedule for 429 rate limit
    backoffs = [2, 4, 8]
    models_to_try = [PRIMARY_MODEL, BACKUP_MODEL]
    
    client = Groq(api_key=api_key.strip())
    
    last_error = None
    for model_name in models_to_try:
        for attempt, delay in enumerate([0] + backoffs):
            if delay > 0:
                time.sleep(delay)
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model_name,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=900
                )
                
                content = chat_completion.choices[0].message.content
                data = json.loads(content)
                
                # Clinical scope guardrail check
                is_safe, violations = verify_clinical_scope(content)
                if not is_safe:
                    # Guardrail caught overreach -> fall back to deterministic safety rule
                    res = generate_fallback_briefing(player, ml_risk_class, risk_prob)
                    res["decided_by"] = "Fallback Rule (Guardrail Tripped)"
                    res["guardrail_violations"] = violations
                    return res
                
                rec_status = data.get("recommended_status", "Modified Training" if ml_risk_class == "High Risk" else "Cleared")
                if rec_status not in ["Cleared", "Modified Training", "Rest Mandated"]:
                    rec_status = "Modified Training" if ml_risk_class == "High Risk" else "Cleared"
                    
                rationale = data.get("protocol_rationale", f"Evaluated under protocol: {ml_risk_class} profile.")
                triggered = data.get("triggered_clauses", [])
                pitch_restr = data.get("pitch_restrictions", "Standard load")
                gym_restr = data.get("gym_restrictions", "Standard load")
                rec_mod = data.get("recovery_modalities", "Standard recovery")
                patho = data.get("pathophysiology_summary", "Telemetry analyzed against clinical baselines.")
                
                # Render full briefing according to briefing-template.md
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                briefing_md = template_text
                briefing_md = briefing_md.replace("{{PLAYER_NAME}}", player.get("name", "Unknown Player"))
                briefing_md = briefing_md.replace("{{PLAYER_ID}}", player.get("player_id", "PLY-000"))
                briefing_md = briefing_md.replace("{{POSITION}}", player.get("position", "Unknown"))
                briefing_md = briefing_md.replace("{{EVALUATION_DATE}}", now_str)
                briefing_md = briefing_md.replace("{{ACWR}}", f"{float(player.get('acwr', 1.0)):.2f}")
                briefing_md = briefing_md.replace("{{SLEEP_HOURS}}", f"{float(player.get('avg_sleep_hours', 8.0)):.1f}")
                briefing_md = briefing_md.replace("{{WELLNESS_SCORE}}", f"{float(player.get('wellness_score', 8.0)):.1f}")
                briefing_md = briefing_md.replace("{{FATIGUE}}", f"{float(player.get('fatigue', 2.0)):.1f}")
                briefing_md = briefing_md.replace("{{SORENESS}}", f"{float(player.get('soreness', 2.0)):.1f}")
                briefing_md = briefing_md.replace("{{STRESS}}", f"{float(player.get('stress', 3.0)):.1f}")
                briefing_md = briefing_md.replace("{{MOOD}}", f"{float(player.get('mood', 8.0)):.1f}")
                briefing_md = briefing_md.replace("{{HSR_METERS}}", f"{float(player.get('high_speed_running_m', 450.0)):.0f}")
                briefing_md = briefing_md.replace("{{HSR_VARIANCE}}", f"{float(player.get('hsr_spike_pct', 0.0)):+.1f}")
                briefing_md = briefing_md.replace("{{INJURY_HISTORY_SUMMARY}}", f"{player.get('previous_injuries_count', 0)} events; last: {player.get('last_injury_type', 'None')}")
                briefing_md = briefing_md.replace("{{ML_RISK_CLASS}}", ml_risk_class)
                briefing_md = briefing_md.replace("{{RISK_PROBABILITY}}", f"{risk_prob * 100:.1f}")
                briefing_md = briefing_md.replace("{{RISK_VECTORS}}", f"ACWR={player.get('acwr')}, Sleep={player.get('avg_sleep_hours')}h, Soreness={player.get('soreness')}/10")
                briefing_md = briefing_md.replace("{{TRIGGERED_CLAUSES}}", "\n".join([f"- {c}" for c in triggered]))
                briefing_md = briefing_md.replace("{{PATHOPHYSIOLOGY_RATIONALE}}", patho)
                briefing_md = briefing_md.replace("{{RECOMMENDED_STATUS}}", rec_status)
                briefing_md = briefing_md.replace("{{PITCH_RESTRICTIONS}}", pitch_restr)
                briefing_md = briefing_md.replace("{{GYM_RESTRICTIONS}}", gym_restr)
                briefing_md = briefing_md.replace("{{RECOVERY_MODALITIES}}", rec_mod)
                briefing_md = briefing_md.replace("{{ONE_SENTENCE_DIRECTIVE}}", rationale)

                return {
                    "player_id": player.get("player_id"),
                    "player_name": player.get("name"),
                    "ml_risk_class": ml_risk_class,
                    "risk_probability": risk_prob,
                    "recommended_status": rec_status,
                    "protocol_rationale": rationale,
                    "full_briefing_markdown": briefing_md,
                    "decided_by": "ML Model + LLM RAG",
                    "model_used": model_name,
                    "triggered_clauses": triggered
                }

            except Exception as e:
                err_str = str(e)
                last_error = err_str
                # If rate-limited (429), continue exponential backoff loop
                if "429" in err_str or "rate_limit" in err_str.lower():
                    continue
                # If invalid auth or invalid key, break out to fallback
                if "invalid_api_key" in err_str.lower() or "401" in err_str:
                    break
                # Otherwise break attempt loop and try next model
                break

    # If all models and retries exhausted, fall back seamlessly
    res = generate_fallback_briefing(player, ml_risk_class, risk_prob)
    res["decided_by"] = "Fallback Rule"
    res["error_notice"] = f"LLM API limit or network unavailable ({last_error}). Evaluated via grounded rule fallback."
    return res

