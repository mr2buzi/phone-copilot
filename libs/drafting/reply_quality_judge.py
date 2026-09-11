from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ReplyQualityJudgement:
    obligation_satisfied: bool = True
    quality_gate_penalty: float = 0.0
    quality_gate_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def judge_reply_obligation(analysis: dict[str, object]) -> ReplyQualityJudgement:
    reasons: list[str] = []
    scene_type = str(analysis.get("scene_type") or "")
    agenda_state = str(analysis.get("agenda_state") or "")
    conversation_job = str(analysis.get("conversation_job") or "")

    if conversation_job != "acknowledge_repeated_reply" and float(analysis.get("ignored_question_debt_penalty") or 0.0) >= 0.7:
        reasons.append("unanswered_question_debt")
    if conversation_job != "acknowledge_repeated_reply" and float(analysis.get("asked_new_question_before_answering_penalty") or 0.0) >= 0.7:
        reasons.append("new_question_before_answering_debt")
    if bool(analysis.get("policy_violation")):
        reasons.append("policy_violation")
    if not bool(analysis.get("policy_must_satisfied", True)) and conversation_job != "normal_reply":
        reasons.append("policy_obligation_missing")
    if bool(analysis.get("forbidden_move_violated")):
        reasons.append("scene_forbidden_move")
    if (
        not bool(analysis.get("required_move_satisfied", True))
        and scene_type not in {"normal", "opening", "dead_conversation"}
    ):
        reasons.append("scene_required_move_missing")
    if bool(analysis.get("forbidden_dialogue_move_violated")):
        reasons.append("agenda_forbidden_move")
    if float(analysis.get("generic_prompt_penalty") or 0.0) >= 0.7:
        reasons.append("generic_prompt_loop")
    if conversation_job != "answer_reciprocal_activity" and float(analysis.get("stale_template_penalty") or 0.0) >= 0.7:
        reasons.append("stale_template")
    if float(analysis.get("semantic_contamination_penalty") or 0.0) >= 0.7:
        reasons.append("semantic_contamination")
    if float(analysis.get("ignored_identity_question_penalty") or 0.0) >= 0.7:
        reasons.append("ignored_identity_question")
    if float(analysis.get("affection_missed_penalty") or 0.0) >= 0.7:
        reasons.append("missed_affection")
    if agenda_state in {"topic_selection_needed", "dead_conversation_recovery"} and float(analysis.get("agenda_fit_score") or 0.0) <= 0.1:
        reasons.append("agenda_progression_missing")

    reasons = sorted(dict.fromkeys(reasons))
    return ReplyQualityJudgement(
        obligation_satisfied=not reasons,
        quality_gate_penalty=1.0 if reasons else 0.0,
        quality_gate_reasons=reasons,
    )
