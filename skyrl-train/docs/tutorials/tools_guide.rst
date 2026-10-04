Execute task tools with Shellbox
================================

A task session receives its prepared Shellbox ``Machine``.
The engine creates the machine from the task's ``EnvironmentSpec``
and closes the session before the machine.

The machine provides command execution and file transfer:

.. code-block:: python

   from shellbox.machine import Command

   result = await machine.run(
       Command(("python", "-c", "print(6 * 7)"), timeout=10.0)
   )

``Result`` contains stdout, stderr, an exit code, a timeout reason, and truncation flags.
Candidate errors can become model-visible tool observations.
Machine failures are infrastructure errors and must not become incorrect-answer grades.
For example, a Python ``NameError`` returns tool feedback while the interpreter retains its state.
A lost interpreter or an invalid execution response produces an infrastructure failure with no verdict.
An exit code or timeout alone does not distinguish those cases.

Stateful Python
---------------

``skyrl_gym.python_execution.PythonKernel`` keeps one interpreter per task.
Start the kernel during session preparation. Execute each tool call in the same namespace.
Close the kernel during session cleanup.

.. code-block:: python

   from skyrl_gym.python_execution import PythonKernel

   kernel = PythonKernel(machine)
   await kernel.start()
   try:
       await kernel.execute("value = 6 * 7", timeout=10.0)
       result = await kernel.execute("print(value)", timeout=10.0)
   finally:
       await kernel.close()

The task image must contain IPython and the benchmark's Python dependencies.
Per-call timeouts retain the interpreter namespace when the kernel survives.
A lost kernel is an infrastructure failure. Do not restart it and silently discard state.

Code and SQL verification
-------------------------

``skyrl_gym.code_execution.execute_code`` sends the candidate and test inputs
to the machine. It compares outputs with private expected answers on the worker.
Dataset preparation uses ``validate_code_example`` with a supplied machine for
positive and negative candidate checks. The data contract only normalizes code-test inputs.

SQL sessions install a read-only snapshot of the public database in the machine.
Private expected query results stay on the worker.
Each rollout receives a fresh machine, so one task cannot change another task's database.

Lean
----

``skyrl_gym.lean_execution.compile_lean`` executes ``lake env lean``
in the task's ``environment.workdir``. The task image must contain the toolchain and cached dependencies.
The function returns the same ``shellbox.machine.Result`` record as other task commands.
The compiler uses bounded diagnostics and a process deadline.
Lean refinement returns compiler feedback and a fresh conversation for the next attempt.

External services
-----------------

Search and judge clients execute HTTP requests on the worker.
Their endpoints and credentials belong to the task's private verifier configuration.
OpenEnv starts its HTTP server inside the supplied machine.
It does not create a second Docker container.

See :doc:`new_env` for session factories and :doc:`task_rollouts` for rollout ownership.
