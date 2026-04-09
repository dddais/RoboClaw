---
name: eap-data-collection
description: "EAP data collection workflow for prompt-driven robotic rollouts. Use this skill when you need to collect self-resetting forward and reverse trajectory pairs, keep rollout metadata and trajectory records in dataset `D`, and delegate each robot execution step to $monitored-subtask-execution so MCP startup, monitoring, timeout handling, stop, and reset remain centralized."
---

# EAP Data Collection

## Overview

Use one closed-loop workflow to collect self-resetting manipulation data:

- Run a forward rollout that performs the target behavior.
- Run a reverse or recovery rollout that returns the environment to a reusable starting state.
- Persist both trajectories and rollout metadata into dataset `D`.

Important constraint: keep all concrete robot MCP tool-call details (CoRobot or x2robot) inside `$monitored-subtask-execution`. This skill only defines the data collection procedure.

## Inputs (you must provide)

- `task_name`: short identifier for the collection task
- `forward_prompt`: prompt that performs the target behavior
- `reverse_prompt`: prompt that restores the environment to a reusable starting state
- `initial_state_criteria`: observable conditions that define a valid starting state
- `run_dir`: output directory for logs, rollout metadata, and dataset `D`
- `policy host/port`: policy server address, optional, passed through to `$monitored-subtask-execution`
- `step_interval`: optional PolicyTask step interval, passed through to `$monitored-subtask-execution`
- `run_budget`: target number of collection rounds, total runtime limit, and maximum retries per round
- `timeouts/polling`: per-rollout timeout and poll interval, configured at the `$monitored-subtask-execution` layer

## Defaulting Rules

When the user gives only a high-level collection request, do not block on a long parameter questionnaire if one safe demo round can be inferred.

- If `task_name` is missing, derive a short slug from the user request.
- If `forward_prompt` is missing, rewrite the user goal into one direct prompt-driven robot instruction.
- If `reverse_prompt` is missing but the recovery action is obvious, infer the inverse prompt. If the recovery action is not obvious, stop and ask instead of guessing.
- If `run_dir` is missing, create a timestamped default such as `runs/eap/<task_name>-<timestamp>`.
- If `run_budget` is missing, default to one round, zero or one retry, and stop after the first completed pair.
- If timeout or polling values are missing, use conservative defaults compatible with `$monitored-subtask-execution`, for example `timeout_s=90` and `poll_interval_s=1.0`.
- Ask follow-up questions only for safety-critical ambiguity, unclear recovery logic, or missing scene facts that prevent execution.

## Tools Used

Use the same skill for rollout execution and reset:

- `AgentTools___ensure_run_artifacts`: create `run_dir`, `run_dir/logs`, and `run_dir/dataset`
- `AgentTools___append_jsonl_record`: append round logs, status logs, and dataset episode records as JSONL
- `$monitored-subtask-execution`: wraps the active robot MCP service (corobot or x2robot) startup, polling, stop, and reset so all robot-specific tool calls remain centralized in one skill

## Run Artifacts

Persist collection outputs into simple run artifacts instead of keeping state only in conversation context.

Recommended directory layout inside `run_dir`:

- `run_dir/logs/rounds.jsonl`
- `run_dir/logs/status_text.jsonl`
- `run_dir/logs/tool_calls.jsonl` (optional)
- `run_dir/dataset/episodes.jsonl` or an equivalent dataset `D` format

Keep at least these records:

- one round record per forward/reverse pair, including prompt versions, parameters, status, and success outcome
- raw status text from `$monitored-subtask-execution` so failures can be replayed or debugged later
- dataset episodes for forward and reverse trajectories, each linked back to the round identifier

## Main Loop

### Step 0: Safety + Preconditions

- Confirm the robot workspace is safe, emergency stop is available, and human intervention is possible.
- Define the starting state explicitly. Example: robot at home pose, object in a designated table region, drawer closed.
- Stop the run if repeated failures or unsafe behavior appear. Reset the environment before continuing.

### Step 1: Initialize the Run

- Call `AgentTools___ensure_run_artifacts` to create `run_dir` and the logging and dataset subdirectories if they do not already exist.
- Write one run header record that stores `task_name`, prompt text or prompt hashes, `initial_state_criteria`, and rollout parameters.
- Use `AgentTools___append_jsonl_record` to write that run header into `run_dir/logs/rounds.jsonl` or an equivalent JSONL run log.
- Initialize round counters and stop conditions from `run_budget`.

### Step 2: Verify the Starting State

- Check the current scene against `initial_state_criteria`.
- If the starting state is not valid, use the hard reset path from `$monitored-subtask-execution` or require human intervention before collecting more data.

### Step 3: Execute the Forward Rollout

- Call `$monitored-subtask-execution` with:
- `prompt = forward_prompt`
- `reset_after = false`
- the current policy, timeout, poll interval, and retry parameters
- Save the rollout result, timestamps, and status text into the round log with `AgentTools___append_jsonl_record`.

### Step 4: Persist the Forward Trajectory

- Persist the forward trajectory as one dataset episode.
- Attach at least:
- `task_name`
- `direction = forward`
- `prompt`
- `success` or `failure_reason`
- `policy_host/port`, `step_interval`, `timeout_s`, `poll_interval_s`
- trajectory data fields such as `o_t`, `q_t`, and `a_t`
- Use `AgentTools___append_jsonl_record` to persist the episode into `run_dir/dataset/episodes.jsonl`.

### Step 5: Execute the Reverse or Recovery Rollout

- If a reliable `reverse_prompt` is available, call `$monitored-subtask-execution` again with:
- `prompt = reverse_prompt`
- `reset_after = true`
- the same timeout, polling, and retry policy used for the forward run unless there is a clear reason to differ
- If no reliable `reverse_prompt` exists, use the hard reset path from `$monitored-subtask-execution` and mark the reverse trajectory as missing.

### Step 6: Persist the Reverse Trajectory and Round Outcome

- Persist the reverse trajectory as a second dataset episode when it exists.
- Write one round summary record containing:
- `round_id`
- `task_name`
- forward status and reverse status
- whether the ending state satisfies `initial_state_criteria`
- whether the round is complete, failed, or requires human reset
- Write the round summary with `AgentTools___append_jsonl_record`.

### Step 7: Decide Whether to Continue

- Continue collecting while:
- the target round count has not been reached
- the runtime limit has not been exceeded
- failure counts stay below the allowed threshold
- Stop and require manual reset if the environment can no longer be restored to the starting state reliably.

## Failure Handling

- If the forward rollout fails, record the failure, reset the robot, and decide whether to retry the round or stop.
- If the reverse rollout fails, record the failure, force a reset, and do not start the next round until the starting state is valid again.
- If status becomes uncertain at any point, prefer `stop_task` followed by `reset_task` through `$monitored-subtask-execution` rather than continuing blindly.

## References

- `references/eap-prompt-pairs.md`: validated forward/reverse prompt pairs that reduce prompt drift
- `references/data-collection-guardrails.md`: stability rules, minimum logging fields, and stop conditions
- `$monitored-subtask-execution`: single-subtask rollout procedure with safe startup, monitoring, stop, and reset
