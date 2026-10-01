"""Build-time checks against the sandbox image's actual stateful execution service."""

import concurrent.futures
import unittest
import uuid

from main import execute_ipython_session, shell_manager


class SessionRetentionTests(unittest.TestCase):
    def test_hard_kill_is_an_infrastructure_failure_even_on_the_first_call(self):
        session = str(uuid.uuid4())
        try:
            result = execute_ipython_session(
                "import signal; signal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True: pass", session, timeout=5
            )
            self.assertEqual(result["error_type"], "VerifierRuntimeError")
            self.assertTrue(result["new_session_created"])
        finally:
            shell_manager.stop_shell(session)

    def test_python_error_is_tool_feedback(self):
        session = str(uuid.uuid4())
        try:
            result = execute_ipython_session("raise ValueError('candidate error')", session, timeout=10)
            self.assertEqual(result["process_status"], "error")
            self.assertNotIn("error_type", result)
            self.assertIn("ValueError", result["stdout"] + result["stderr"])
        finally:
            shell_manager.stop_shell(session)

    def test_concurrent_sessions_survive_interrupts(self):
        sessions = [str(uuid.uuid4()) for _ in range(4)]
        try:
            # Fork shells before creating threads, matching the single-threaded HTTP workers.
            for value, session in enumerate(sessions):
                initial = execute_ipython_session(f"x = {value}", session, timeout=10)
                self.assertEqual(initial["process_status"], "completed")

            def trajectory(value):
                session = sessions[value]
                interrupted = execute_ipython_session("while True: pass", session, timeout=0.25)
                self.assertEqual(interrupted["process_status"], "timeout")
                self.assertFalse(interrupted["new_session_created"])
                continuation = execute_ipython_session("print(x + 1)", session, timeout=10)
                self.assertEqual(continuation["process_status"], "completed")
                self.assertFalse(continuation["new_session_created"])
                self.assertEqual(continuation["stdout"].strip(), str(value + 1))

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(trajectory, range(4)))
        finally:
            for session in sessions:
                shell_manager.stop_shell(session)


if __name__ == "__main__":
    unittest.main()
