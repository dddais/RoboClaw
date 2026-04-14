---
name: long-horizon-execution
description: "Long-horizon robot task execution workflow for multi-step manipulation tasks. Use this skill when one user goal must be decomposed into ordered subtasks, each subtask needs explicit success checks, retries, or recovery, and every prompt-driven policy rollout should be executed through $monitored-subtask-execution instead of calling raw MCP robot tools directly."
---

# Long Horizon Execution

## Overview

Use this skill to execute a multi-step robot task as an ordered sequence of verifiable subtasks. Build or refine the subtask plan, execute one policy rollout at a time through `$monitored-subtask-execution`, verify each result, and update the remaining plan until the full task is complete.

## Inputs

- `global_goal`: one-sentence description of the overall task
- `subtask_plan`: ordered subtask list; each item should include `subtask_id`, `prompt`, `success_check`, `reset_after`, and `max_retries`
- `completion_criteria`: observable conditions that define overall task completion
- `run_dir`: directory for progress logs and final run summary
- `policy host/port`: optional policy server passed through to `$monitored-subtask-execution`
- `step_interval`: optional PolicyTask step interval passed through to `$monitored-subtask-execution`
- `timeout_s` and `poll_interval_s`: rollout timeout and poll interval used by `$monitored-subtask-execution`; recommended `poll_interval_s` is `5.0` to `10.0` to avoid API rate limits

## Defaulting Rules

If the user gives only a high-level execution goal, infer a minimal one-pass plan instead of blocking on every missing field.

- If `subtask_plan` is missing, derive a short ordered plan from `global_goal`.
- If `run_dir` is missing, create a timestamped default such as `runs/long-horizon/<goal-slug>-<timestamp>`.
- If timeout or polling values are missing, choose conservative defaults suitable for one monitored rollout.
- Ask follow-up questions only when the scene state, success checks, or safety boundary are too ambiguous to execute responsibly.

## Hard Rules

- Execute every prompt-driven subtask through `$monitored-subtask-execution`.
- Do not call `corobot_mcp_server___*` or `x2robot_mcp_server___*` directly from this skill.
- Keep each subtask narrow enough to have one prompt and one external success check.
- Prefer deterministic recovery before replanning. Replan only when the current plan no longer matches the world state.
- When the next step should be executed automatically, return a top-level JSON object with `status`, `selected_skill`, `next_skill`, and `skill_args`.

## Workflow

### Step 0: Safety and Preconditions

- Confirm the robot workspace is safe, emergency stop is available, and human intervention is possible.
- Check that the environment is in a valid pre-task state or can be recovered into one.
- Stop before execution if the current state is unsafe or if the task requires manual setup that has not happened yet.

### Step 1: Build or Refine the Subtask Plan

- Decompose `global_goal` into a short ordered list of subtasks.
- Give each subtask one prompt, one success check, one retry budget, and one `reset_after` choice.
- Keep irreversible or high-risk actions separate so failures can be isolated and recovered cleanly.
- Reuse validated prompt text when available. See `references/subtask-plan-template.md` for a minimal schema.

### Step 2: Initialize Run Tracking

- Call `AgentTools___ensure_run_artifacts` to create `run_dir` if needed.
- Start a progress log, for example `run_dir/logs/subtasks.jsonl`, that records every attempt with `subtask_id`, prompt, parameters, timestamps, status text, and verification result.
- Use `AgentTools___append_jsonl_record` for progress-log writes instead of keeping state only in chat history.
- Initialize each planned subtask as `todo`.

### Step 3: Select the Next Subtask

- Choose the first subtask that is not yet complete.
- Before execution, confirm that the subtask preconditions still hold.
- If a completed subtask has been invalidated by drift or recovery, mark the affected downstream subtasks as pending again before proceeding.

### Step 4: Execute the Subtask Through `$monitored-subtask-execution`

- Do not directly inline the subskill instructions. Instead, emit a structured delegation payload so the main Olympus agent can invoke `$monitored-subtask-execution`.
- The delegation payload must be a top-level JSON object in this shape:

```json
{
  "status": "continue",
  "selected_skill": "long-horizon-execution",
  "next_skill": "monitored-subtask-execution",
  "skill_args": {
    "prompt": "<current subtask prompt>",
    "success_check": "<observable verification rule>",
    "reset_after": false,
    "max_retries": 1,
    "timeout_s": 90,
    "poll_interval_s": 1.0
  }
}
```

- Include any pass-through fields needed by `$monitored-subtask-execution`, such as `policy.host`, `policy.port`, or `step_interval`.
- Record the returned status text and attempt metadata in the progress log after the delegated execution returns.

### Step 5: Verify the Result (MANDATORY visual check)

- **You MUST call `AgentTools___fetch_env` to get a fresh image** after each subtask completes, if `$monitored-subtask-execution` has not already done so in this turn.
- Compare the new image against the subtask `success_check`.
- If the visual evidence confirms the success check is met, mark the subtask `done` and move to the next one.
- If the status says "completed" but the image shows the objective was NOT achieved, treat it as a failed attempt.
- Do NOT advance to the next subtask without visual confirmation.

### Step 6: Recover, Retry, or Replan

- If the subtask failed and retries remain, run the same subtask again after any required reset.
- If the environment must return to a reusable state first, use the hard reset path exposed by `$monitored-subtask-execution`.
- Replan only when the remaining steps are no longer valid for the current world state. Preserve the history of completed and failed subtasks in the run log.

### Step 7: Finish the Task

- After every completed subtask, check `completion_criteria`.
- When the overall task is complete, write a final summary with the completed plan, attempt counts, failure reasons, and any manual interventions.
- If the task cannot be finished safely, stop and return the robot to a safe state before exiting.

## Response Contract

Return one top-level JSON object.

- If the next subtask should run automatically, return:
  - `status = "continue"`
  - `selected_skill = "long-horizon-execution"`
  - `next_skill = "monitored-subtask-execution"`
  - `skill_args = {...}` with the exact inputs for the delegated subtask skill
- If human clarification is required, return:
  - `status = "ask_human"`
  - `selected_skill = "long-horizon-execution"`
  - `question`
  - `reason`
- If the overall task is complete, return:
  - `status = "done"`
  - `selected_skill = "long-horizon-execution"`
  - `summary`
  - `completion_check`

## Failure Handling

- If a rollout fails or times out, rely on `$monitored-subtask-execution` to stop and reset deterministically before the next decision.
- If repeated failures happen on the same subtask, stop escalating automatically and require either manual intervention or a revised plan.
- If the robot reaches a state that invalidates future subtasks, update the remaining plan before running any more policies.

## Scope Boundary

- Use this skill for long-horizon task execution.
- Use `$eap-data-collection` instead when the goal is to collect forward/reverse trajectory pairs into dataset `D`.

## References

- `references/subtask-plan-template.md`: minimal subtask schema, logging fields, and a concrete example plan
- `$monitored-subtask-execution`: monitored single-subtask policy execution and hard reset workflow
