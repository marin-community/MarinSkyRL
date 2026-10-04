"""Explicit factories for the task implementations supplied by SkyRL."""

from collections.abc import Callable
from concurrent.futures import Executor
from functools import partial

from rolloutengine.contracts import TaskSession
from shellbox.machine import Machine
from taskcompendium.models import TaskSpec

from skyrl_gym.answer_tasks import (
    grade_aime,
    grade_cat_count,
    grade_gsm8k,
    grade_ifeval,
    grade_mcq,
    grade_nupa,
    grade_prompt_only,
    grade_reasoning_gym,
)
from skyrl_gym.task_sessions import (
    AnswerTaskSession,
    CodeTaskSession,
    MathTaskSession,
    SearchCodeTaskSession,
    SearchTaskSession,
)
from skyrl_gym.nemotron_tasks import NemotronTaskSession
from skyrl_gym.sql_tasks import SQLTaskSession, SeededSQLTaskSession


def session_factories(
    *, max_turns: int, executor: Executor | None = None
) -> dict[str, Callable[[TaskSpec, Machine | None], TaskSession]]:
    """Return fresh-session factories with the worker's turn limit and verifier executor."""
    answer_graders = {
        "aime": grade_aime,
        "cat_count": grade_cat_count,
        "gsm8k": grade_gsm8k,
        "ifeval": grade_ifeval,
        "mcq": grade_mcq,
        "nupa": grade_nupa,
        "preference": grade_prompt_only,
        "prompt_only": grade_prompt_only,
        "reasoning_gym": grade_reasoning_gym,
    }
    return {
        **{
            name: partial(AnswerTaskSession, grader=grader, executor=executor)
            for name, grader in answer_graders.items()
        },
        "gsm8k_multi_turn": partial(MathTaskSession, max_turns=max_turns),
        "search": partial(SearchTaskSession, max_turns=max_turns, executor=executor),
        "searchcode": partial(SearchCodeTaskSession, max_turns=max_turns, executor=executor),
        "lcb": CodeTaskSession,
        "nemotron_ultra": partial(NemotronTaskSession, executor=executor),
        "text2sql": partial(SQLTaskSession, max_turns=max_turns, executor=executor),
        "text_to_sql": partial(SeededSQLTaskSession, executor=executor),
    }
