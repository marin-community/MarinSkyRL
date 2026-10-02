from func_timeout import func_timeout, FunctionTimedOut
from skyrl_gym.tools.core import tool, ToolGroup
import pandas as pd
import sqlite3
import sys
import os


class SQLCodeExecutorToolGroup(ToolGroup):
    def __init__(self, db_file_path: str, verifyit_enabled: bool = False):
        self.db_path = db_file_path
        self.verifyit_enabled = verifyit_enabled
        super().__init__(name="SQLCodeExecutorToolGroup")

    @tool
    def sql(self, db_id, sql, turns_left, timeout=5) -> str:
        def _execute_sql(db_file, sql):
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                conn.execute("BEGIN TRANSACTION;")
                cursor.execute(sql)
                execution_res = frozenset(cursor.fetchall())
                conn.rollback()
                conn.close()
                return execution_res
            except Exception as e:
                conn.rollback()
                conn.close()
                return f"Error executing SQL: {str(e)}, db file: {db_file}"

        def _execute_sql_wrapper(db_file, sql, timeout=5) -> str:
            try:
                res = func_timeout(timeout, _execute_sql, args=(db_file, sql))
                if isinstance(res, frozenset):
                    df = pd.DataFrame(res)
                    res = df.to_string(index=False)
                    # NOTE: observation too long, just truncate
                    if len(res) > 9000:
                        # just truncate
                        truncated_df = df.head(50)
                        res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                            index=False
                        )  # or index=True if you want row numbers
                else:
                    res = str(res)

            except KeyboardInterrupt:
                sys.exit(0)
            except FunctionTimedOut:
                res = f"SQL Timeout:\n{sql}"
            except Exception as e:
                res = str(e)

            return res

        # TODO (erictang000): move this logic up into the text2sql env, since this is more specific logic
        reminder_text = f"<reminder>You have {turns_left} turns left to complete the task.</reminder>"
        if sql is None:
            obs = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
        else:
            db_file = os.path.join(self.db_path, db_id, db_id + ".sqlite")
            if self.verifyit_enabled:
                from verifyit.bounded import call_bounded

                obs = call_bounded(execute_bounded_sql_tool, db_file, sql, timeout=timeout)
            else:
                obs = _execute_sql_wrapper(db_file, sql, timeout)

        return f"\n\n<observation>{obs}\n{reminder_text}</observation>\n\n"


def execute_bounded_sql_tool(database: str, statement: str) -> str:
    """Read-only, 100000-row interactive observation policy, bounded by its caller."""
    from pathlib import Path
    from skyrl_gym.envs.text_to_sql import scoring
    from skyrl_gym.envs.sqlite_verifyit import _query_error

    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            _, rows = scoring._run_query(connection, statement, read_only=True)
        except sqlite3.Error as error:
            if not _query_error(error):
                raise
            return "Error executing SQL: query rejected"
        if len(rows) > scoring._MAX_RESULT_ROWS:
            return "Error executing SQL: result exceeds 100000 rows"
        frame = pd.DataFrame(frozenset(rows))
        observation = frame.to_string(index=False)
        if len(observation) > 9000:
            observation = "Truncated to 50 lines since returned response too long: " + frame.head(50).to_string(
                index=False
            )
        return observation
    finally:
        connection.close()
