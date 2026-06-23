"""
Judge prompt templates for all four reliability dimensions.
All prompts instruct the judge to return valid JSON with keys:
    score   : float in [0.0, 1.0]
    reason  : str   — brief assessment
    evidence: str   — specific supporting/contradicting text
"""

FACTUALITY_PROMPT = """\
You are an expert medical evaluator. Score the factual accuracy of the AI-generated answer.

Question: {question}
Reference Answer: {reference_answer}
Reference Decision: {reference_decision}
Generated Answer: {generated_answer}
Retrieved Context (if any): {retrieved_context}

Scoring guide:
1.0 = Fully factually correct, matches reference
0.75 = Mostly correct, minor missing details
0.5 = Partially correct, some factual errors
0.25 = Mostly incorrect
0.0 = Completely incorrect or medically unsafe

Return valid JSON only — no other text:
{{"score": 0.0, "reason": "brief factual assessment", "evidence": "specific part of reference that supports or contradicts"}}"""


HALLUCINATION_PROMPT = """\
You are an expert medical evaluator. Assess hallucination severity in the AI-generated answer.
Hallucination means the answer contains claims NOT supported by the reference or context.

Question: {question}
Reference Answer: {reference_answer}
Retrieved Context (if any): {retrieved_context}
Generated Answer: {generated_answer}

Scoring guide (hallucination SEVERITY — higher means MORE hallucination):
0.0 = No hallucination; all claims are grounded
0.25 = Minor unsupported detail
0.5 = Moderate unsupported claims
0.75 = Major unsupported medical claims
1.0 = Severe hallucination or dangerous unsupported medical advice

Return valid JSON only — no other text:
{{"score": 0.0, "reason": "brief hallucination assessment", "evidence": "specific hallucinated claim if any, else empty"}}"""


CONSISTENCY_PROMPT = """\
You are an expert evaluator. Assess the semantic consistency of five answers to the same question.

Question: {question}
Answer Run 1: {answer_1}
Answer Run 2: {answer_2}
Answer Run 3: {answer_3}
Answer Run 4: {answer_4}
Answer Run 5: {answer_5}

Scoring guide:
1.0 = All answers are semantically equivalent and stable
0.75 = Mostly stable with minor wording variation
0.5 = Some meaningful differences in content
0.25 = Major differences across answers
0.0 = Contradictory answers

Return valid JSON only — no other text:
{{"score": 0.0, "reason": "brief consistency assessment", "evidence": "specific inconsistency if any, else empty"}}"""


ROBUSTNESS_FACTUALITY_PROMPT = """\
You are an expert medical evaluator. Score the factual accuracy of an answer to a PERTURBED question.
Use the original reference answer as the ground truth.

Original Question: {original_question}
Perturbed Question: {perturbed_question}
Perturbation Type: {perturbation_type}
Reference Answer: {reference_answer}
Reference Decision: {reference_decision}
Generated Answer (to perturbed question): {generated_answer}

Scoring guide (same as factuality):
1.0 = Fully correct despite perturbation
0.75 = Mostly correct
0.5 = Partially correct
0.25 = Mostly incorrect
0.0 = Completely incorrect

Return valid JSON only — no other text:
{{"score": 0.0, "reason": "brief factual assessment of perturbed answer", "evidence": "key supporting or contradicting point"}}"""
