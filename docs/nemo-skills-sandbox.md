# Stateful NeMo Skills sandbox

Build the service image from the repository root:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.nemo-skills-sandbox -t nemo-skills-sandbox:session-retention docker
```

The image applies a patch to the pinned service runtime and runs four concurrent trajectories during the build.
Each trajectory assigns a variable, exceeds a tool timeout, and reads the variable again. The build fails if an
interrupt discards state. Publish the image through the normal registry workflow and record its immutable digest
before an authorized deployment update. Updating the shared deployment restarts its pods and discards existing
sessions; drain callers before rollout.

The Python shell must install `signal.default_int_handler` after constructing `TerminalInteractiveShell`.
`signal.SIG_DFL` terminates the shell on SIGINT. A tool execution timeout sends SIGINT, so that setting destroys
variables even when the pod and HTTP worker remain healthy. Python's default interrupt handler raises
`KeyboardInterrupt`, allowing the shell to return a timeout observation and retain its namespace. Code that
ignores interrupts can still require a hard kill; state lost in that case is an infrastructure failure.

Sandbox transport failures, malformed responses, worker errors, and unexpected session recreation produce
`VerificationStatus.ERROR` with no score or pass/fail verdict. Diagnostics use the existing `VerifierRuntimeError`
identifier and `infrastructure` category; the original transport exception is retained as `cause_error_type`.
SkyRL masks these trajectories from both loss and group baselines. Python exceptions and timeouts that preserve
the session remain tool observations so the model can continue. Do not retry state-mutating calls in a new shell.

Inspect worker logs as well as pod restart counts when state disappears. A worker log reporting a shell dying
"during interrupt" distinguishes this timeout failure from pod OOM. The service memory limit must cover every
retained session, including kernels idle between tool calls. A request semaphore only bounds active execution.
The interrupt fix does not bound accumulated kernels or establish a safe memory budget for arbitrary programs.
After deployment, validate concurrent workload retention, memory use, and cleanup before accepting affected
`math_tir` attempts for quality or difficulty ratings. Incident evidence: [Echo](https://echo.oa.dev/wiki/564).
