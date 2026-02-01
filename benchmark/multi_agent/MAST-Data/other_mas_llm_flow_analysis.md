# Other MAS Systems LLM Flow Analysis (One Trace Each)

Based on **MAD_full_dataset.json** and **MAD_full_dataset_expanded.json**, this document summarizes the LLM call flow for each MAS except ChatDev (see `trace0_llm_flow_analysis.md`). Each MAS is analyzed using one representative trace.

---

## 1. AppWorld

**Sample trace**: first record with `mas_name: AppWorld` (e.g. task: "Keep going to the next song on Spotify until you reach a song by Lily Moon").

**Flow type**: **Hierarchical message-passing**. A **Supervisor Agent** (LLM) orchestrates; it can call **sub-agents** (e.g. spotify Agent) via `send_message`. Each sub-agent runs in a message loop (Entering/Exiting); sub-agent responses are also LLM outputs. Code execution runs in the supervisor’s environment (e.g. `send_message(...)`, `apis.api_docs.show_api_descriptions(...)`).

**LLM flow (conceptual)**:

```
Task (one line)
    |
    v
Supervisor Agent (LLM)  -->  response (reasoning + code, e.g. send_message(app_name='spotify', ...))
    |
    v
Code Execution Output  -->  run supervisor code (may call extension)
    |
    v
Entering X Agent message loop
    |
    v
Message to X Agent  -->  (from supervisor or API)
    v
Response from X Agent (LLM)  -->  sub-agent output (e.g. spotify)
    |
    v
(message / response may repeat; then)
    v
Exiting X Agent message loop  -->  Reply from X to Supervisor
    |
    v
Response from send_message API  -->  supervisor receives reply
    |
    v
Supervisor Agent (LLM) again  -->  next reasoning + code
    |
    ... (repeat until task done or fail)
    v
Evaluation (JSON: success, difficulty, passes, failures)
```

**From expanded (one trace)**: ~29 `response_from_agent`, ~28 `message_to_agent`, 9 `code_execution`, 7 `entering_agent_loop` / `exiting_agent_loop` / `api_response`. Agents: Supervisor Agent, spotify Agent (and send_message API). So **multiple LLM calls** for Supervisor and for each sub-agent in turn; **not linear**—supervisor and sub-agents alternate, with code execution in between.

---

## 2. MetaGPT

**Sample trace**: first record with `mas_name: MetaGPT` (same Checkers task as ChatDev trace0).

**Flow type**: **Linear role pipeline**. Human broadcasts the task to all roles; then a fixed sequence of **roles** each produces one (or more) message. Each role is an LLM call.

**LLM flow (conceptual)**:

```
[FROM: Human TO: {'<all>'}]
ACTION: UserRequirement
CONTENT: task_prompt
    |
    v
Role 1 (e.g. SimpleCoder)  -->  NEW MESSAGES: SimpleCoder: <code/output>
    |
    v
Role 2 (e.g. SimpleTester)  -->  NEW MESSAGES: SimpleTester: <test/output>
    |
    v
Role 3 (e.g. SimpleReviewer)  -->  NEW MESSAGES: SimpleReviewer: <review>
    |
    v
(optional repeat) SimpleTester  -->  NEW MESSAGES
    |
    v
(optional repeat) SimpleReviewer  -->  NEW MESSAGES
    |
    ... (fixed pipeline, possibly 2 review cycles)
```

**From raw trajectory**: Agent sequence observed: **SimpleCoder -> SimpleTester -> SimpleReviewer -> SimpleTester -> SimpleReviewer**. So **linear** phase order, with a **short loop** (Tester -> Reviewer x2). Each "NEW MESSAGES" block with a role name corresponds to one (or one batch of) LLM call(s) for that role.

---

## 3. AG2

**Sample trace**: first record with `mas_name: AG2` (e.g. GSM/Scrabble math problem: "Joey has 214 points... By how many points is Joey now winning?").

**Flow type**: **Single-shot or short chain**. The trajectory has `instance_id`, `problem_statement`, `other_data` (seed_question, seed_solution, seed_answer), and a single `trajectory.content` block: a **query** (e.g. "Let's use Python to solve... Query requirements: ...") and the **model response** (reasoning + code). So one main LLM call (or very few) per trace: **problem_statement + format instructions -> LLM -> solution**.

**LLM flow (conceptual)**:

```
problem_statement ( + other_data: seed_question, seed_solution, seed_answer )
    |
    v
trajectory.content  -->  one prompt (query requirements + problem)
    |
    v
LLM  -->  single (or few) response(s): reasoning + code/output
```

**Linear**: one direction, no loops or sub-agents in the expanded view.

---

## 4. HyperAgent

**Sample trace**: first record with `mas_name: HyperAgent` (e.g. SWE-Bench-style: "DataSet.update causes chunked dask DataArray to evaluate its values eagerly").

**Flow type**: **ReAct-style agent loop**. The raw trajectory shows `Planner's Response: Thought: ...` and tool calls like `run(path_to_file="xarray/core/dataset.py", ...)`. So: **Planner (LLM)** produces thought + action -> **tool run** (e.g. read file, run code) -> **Planner (LLM)** again with observation -> repeat until done.

**LLM flow (conceptual)**:

```
problem_statement (SWE-Bench style: issue, repo, base_commit, etc.)
    |
    v
Planner (LLM)  -->  Thought: ... Action: run(...) / other tools
    |
    v
Tool execution  -->  result (e.g. file content, command output)
    |
    v
Planner (LLM)  -->  next Thought + Action
    |
    ... (loop until fix or stop)
```

**Not linear**: **loop** (Planner -> tool -> Planner). Expanded sections may be 0 if the parser did not split this format; the raw trajectory still shows this ReAct pattern.

---

## 5. Magentic

**Sample trace**: first record with `mas_name: Magentic` (autogen testbed).

**Flow type**: **Unclear from trajectory**. The expanded trajectory is dominated by **build/env log** (RUN.SH STARTING, Docker, Processing/Installing autogen-core, autogen-ext, autogen-agentchat, requirements). So the **visible** part is linear: env setup steps. The **actual agent chat / LLM calls** (e.g. multi-agent conversation) are not clearly delimited in the first part of the trajectory; they may appear later in the log or in a different format. So we only state: **trajectory head = linear env/build log**; **LLM flow for Magentic not inferred** from this single trace sample.

---

## 6. OpenManus

**Sample trace**: first record with `mas_name: OpenManus` (e.g. "Build Cell Phone Towers on Road" from task.txt).

**Flow type**: **Plan-then-execute**. Log shows: "Read prompt from task.txt", "Creating initial plan", **Token usage (LLM)**, "Plan created successfully" with steps (0–7). So **one LLM call** (or a short planning phase) produces a **plan** (steps); execution of steps likely follows (each step may be LLM or tool, not fully visible in the short sample).

**LLM flow (conceptual)**:

```
Task (from task.txt)
    |
    v
LLM (planning)  -->  Plan: steps 0..N (e.g. "Analyze road layout", "Identify tower locations", ...)
    |
    v
Token usage  -->  Input=426, Completion=116 (one call)
    |
    v
Steps execution  -->  (each step may invoke LLM or browser/tools; detail in full log)
```

**Linear** at the top level (task -> plan -> steps); steps themselves may contain further LLM/tool calls.

---

## 7. Summary Table

| MAS       | Flow type              | Linear? | Loop / hierarchy           | Typical LLM role                    |
|-----------|------------------------|--------|----------------------------|-------------------------------------|
| ChatDev   | Phase pipeline         | No     | CodeReview x3 loop         | CEO/CPO/CTO/Programmer/Reviewer     |
| AppWorld  | Hierarchical messaging | No     | Supervisor <-> sub-agents  | Supervisor Agent, spotify Agent     |
| MetaGPT   | Role pipeline          | Mostly | Tester–Reviewer x2         | SimpleCoder, SimpleTester, Reviewer |
| AG2       | Single/short chain     | Yes    | None                       | One model (solution)                |
| HyperAgent| ReAct loop             | No     | Planner <-> tools          | Planner                             |
| Magentic  | Unclear                | Log yes| N/A                        | N/A (build log in sample)           |
| OpenManus | Plan-then-execute      | Yes    | Plan -> steps              | Planner (and possibly per-step)     |

All analyses above are from **one trace per MAS** in MAD_full_dataset.json and MAD_full_dataset_expanded.json.
