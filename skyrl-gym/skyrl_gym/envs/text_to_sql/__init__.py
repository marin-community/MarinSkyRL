"""Single-turn text-to-SQL RLVR environment.

The model is given a natural-language question and a SQLite ``CREATE TABLE``
schema and must return one ``SELECT`` (inside ``<solution></solution>``). Reward
is result-set equivalence against a reference query, graded on the seeded
database and on a copy with every third row deleted.

The env is registered by entry point in ``skyrl_gym.envs.__init__``; nothing
imports ``TextToSQLEnv`` from this package directly.
"""
