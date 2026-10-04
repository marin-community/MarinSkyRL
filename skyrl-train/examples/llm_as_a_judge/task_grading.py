"""One-turn math grading through a configured judge endpoint."""

import re

from rolloutengine.contracts import ModelTurn, Transition
from taskcompendium.grading import GradeResult, Outcome

from skyrl_gym.answer_tasks import ground_truth
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge

PROMPT = """
You are a strict math evaluation assistant.

Compare the following **gold** and **predicted** math solutions. Your job is to determine if the predicted solution is mathematically correct and if the predicted solution ends with a line of the form:

#### <number>

You must only give a score of "1" if:
- The final line of the predicted solution **ends with `#### <number>`**, and
- The number **matches the final answer in the gold solution** exactly.

Instructions:
- You may provide internal reasoning or explanation before giving your final judgment.
- Your final judgment must appear as a separate line at the end of your response, in the format:

### Final Score: 1

or

### Final Score: 0

Do not include any explanation after the final score.
"""


def grade_judged_answer(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    judge = OpenAIJudge(base_url=config["base_url"], model=config["model"], api_key_env="OPENAI_API_KEY")
    message = PROMPT + f"\n\nGOLD SOLUTION:\n{ground_truth(extras)}\n\nPREDICTED SOLUTION:\n{turn.text}\n\nAnswer:"
    reply = judge.generate([{"role": "user", "content": message}]).strip()
    scores = re.findall(r"### Final Score:\s*([01](?:\.0)?)", reply)
    if scores or reply in {"1", "0"}:
        reward = float(scores[-1] if scores else reply)
        grade = GradeResult(Outcome.GRADED, reward)
    else:
        reward = 0.0
        grade = GradeResult(Outcome.INFRA_ERROR, None, "The judge returned no score", diagnostics={"reply": reply})
    return Transition(done=True, reward=reward, grade=grade)
