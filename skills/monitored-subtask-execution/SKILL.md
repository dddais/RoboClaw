---
name: monitored-subtask-execution
description: "Monitored single-subtask execution workflow. Use the active robot MCP service (corobot_mcp_server or x2robot_mcp_server) to start, monitor, stop, and reset one prompt-driven robot rollout through the sequence set_evaluate_params, poll get_status, then stop/reset. Use this skill whenever a larger workflow needs one repeatable, safe execution unit with timeout handling and deterministic cleanup."
---

# Monitored Subtask Execution

## Overview

Use a standardized procedure to call the active robot MCP service tools and execute one monitored robot subtask. The active robot service is whichever of `corobot_mcp_server` or `x2robot_mcp_server` is currently enabled in the Server Registry. In practice this is one robot task run defined by a prompt, an optional policy host/port, and an optional `step_interval`, followed by deterministic monitoring and cleanup.

## Determining the Active Robot Service

Check the Server Registry to identify the active robot MCP service. The tool prefix will be one of:
- `corobot_mcp_server___` — when using CoRobot (Agibot G01)
- `x2robot_mcp_server___` — when using x2robot (Turtle2)

Throughout this skill, `{robot_svc}` refers to whichever service is active. For example, `{robot_svc}___set_evaluate_params` means `corobot_mcp_server___set_evaluate_params` or `x2robot_mcp_server___set_evaluate_params`.

## Inputs (provide per run)

- `prompt`: task instruction to execute; keep it directly executable and unambiguous
- `policy.host / policy.port`: optional policy server; if omitted, the service default is used (CoRobot: `127.0.0.1:8001`, x2robot: `192.168.0.20:57770`)
- `step_interval`: optional step interval
- `timeout_s`: maximum time to wait for this run
- `poll_interval_s`: interval for polling `get_status`, for example `0.5` to `2.0`
- `reset_after`: whether to call `reset_task` after completion and return to the home pose; collection workflows usually want `true`
- `max_retries`: number of retries after failure; usually `0` to `2`

## Tools (MCP)

Core tools, named as `{service_name}___{tool_name}`:

- `{robot_svc}___set_evaluate_params`: set the prompt, policy, and `step_interval`, then auto-start the task after a short delay
- `{robot_svc}___get_status`: retrieve task status to determine running, success, or failure state
- `{robot_svc}___stop_task`: stop the current task; prefer this before manual intervention
- `{robot_svc}___reset_task`: stop and reset the robot to the initial pose

Optional tools, usually not needed:

- `{robot_svc}___start_task`: explicit start; usually unnecessary because `set_evaluate_params` already auto-starts
- `{robot_svc}___get_prompt` / `{robot_svc}___set_prompt`: inspect or set the current prompt
- `{robot_svc}___emergency_stop` (x2robot only): immediate emergency stop

For exact argument structure, see `references/mcp-tool-map.md`.

## Quick Actions

### Hard reset (no prompt)

When no prompt should be executed and you only need to return the robot to a reusable start state:

1. Call `{robot_svc}___stop_task` if a task might still be running.
2. Call `{robot_svc}___reset_task`.

## Workflow: Execute One Subtask (single prompt)

### Step 0: Safety + Preconditions

- Confirm the robot workspace is safe, emergency stop is available, and human intervention is possible.
- Confirm the robot MCP service is enabled and reachable in `src/agent_demo/config/ormcp_services.json`.
- Decide whether this run should preserve its terminal state or return to a reusable start state:
- If only the action result matters, use `reset_after=false`.
- If the next round needs a reusable start state, use `reset_after=true`.

### Step 1: Start (set params + auto-start)

Call `{robot_svc}___set_evaluate_params`.

Argument template:

```json
{
  "evaluate_params": {
    "policy": {"host": "<inference_host>", "port": <inference_port>},
    "prompt": "<prompt>",
    "step_interval": 1.5
  }
}
```

Important behavior:

- On success, `set_evaluate_params` waits briefly and auto-calls `start_task`. Do not call `start_task` again unless you have a specific reason.

### Step 2: Monitor (poll get_status with a timeout)

- Poll `{robot_svc}___get_status` every `poll_interval_s` until one of these conditions is met:
- the returned status clearly indicates completion, success, or failure
- `timeout_s` is reached
- behavior becomes unsafe or manual intervention is required
- Save the raw status text from each poll as a log record even if it cannot be parsed into a structured schema.

### Step 3: Visual verification after completion (MANDATORY)

**After `get_status` returns completed, you MUST fetch a fresh image before making any success/failure judgment.**

1. Call `AgentTools___fetch_env` to get the latest camera image.
2. Compare the new image against the `success_check` criteria provided in the inputs.
3. Only then determine whether the subtask truly succeeded or failed.

Do NOT skip this step. Do NOT assume success just because `get_status` returned "completed" — the status only means the execution window ended, not that the task objective was achieved.

### Step 4: Handle success / failure deterministically

Success path (visual verification confirms the success_check is met):

- If `reset_after=true`, call `{robot_svc}___reset_task`.
- Record the run metadata: `prompt`, `policy host/port`, `step_interval`, start time, end time, final status text, and visual verification result.

Failure path (visual verification shows the objective was NOT met, or timeout/error):

1. Call `{robot_svc}___stop_task`.
2. Call `{robot_svc}___reset_task`.
3. Record the failure reason: what the image showed vs. what was expected.
4. If `max_retries` allows another attempt, return to Step 1. Otherwise mark the run as failed and exit.

## Troubleshooting

- **CoRobot**: If the result indicates a connection failure, verify that the local CoRobot HTTP service is reachable at `http://localhost:8765`.
- **x2robot**: If the result indicates a connection failure, verify that the x2robot bridge server is reachable at the configured `X2ROBOT_BRIDGE_URL` and that `video_streaming.py` and `socket2ros_async.py` are running on the robot.
- If the task gets stuck, follow the failure path `stop_task -> reset_task`. Pause and restore the environment manually if needed.

## References

- `references/mcp-tool-map.md`: robot MCP service tool lists, naming rules, and argument structure
