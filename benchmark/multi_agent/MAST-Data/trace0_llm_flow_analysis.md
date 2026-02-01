# First Trace LLM Flow Analysis (ChatDev / ProgramDev)

Based on **MAD_full_dataset.json** and **MAD_full_dataset_expanded.json**, this document analyzes the LLM flow of the first trace (trace_id 0).

---

## 1. Trace Overview

| Field | Value |
|-------|--------|
| mas_name | ChatDev |
| llm_name | GPT-4o |
| benchmark_name | ProgramDev |
| trace_id | 0 |
| trace.key | ChatDev_ProgramDev_GPT4o |
| task_prompt | Develop a Checkers (Draughts) game. Use an 8x8 board, alternate turns between two players, and apply standard capture and kinging rules. Prompt for moves in notation (e.g., from-to positions) and update the board state accordingly. |
| project_name | Checkers |

**LLM config** (from trajectory_expanded.metadata.ChatGPTConfig): temperature=0.2, top_p=1.0, n=1, stream=False, no custom stop/max_tokens.

---

## 2. LLM Flow Summary

ChatDev runs a **phase pipeline**. Each phase is a **role-playing chat** between two agents (user role and assistant role). Every chat turn is one **LLM call**: the system builds messages from role prompts + phase_prompt (with placeholders like `{task}`, `{codes}`, `{language}` filled), then the LLM generates the assistant reply. Multiple phases use the same LLM (GPT-4o); only the system/user content and roles change.

High-level order:

1. **Preprocessing** (no LLM): load config, set task_prompt, project_name, ChatDevConfig, ChatGPTConfig.
2. **DemandAnalysis** → **LanguageChoose** → **Coding** → **CodeReviewComment** / **CodeReviewModification** (3 cycles) → **EnvironmentDoc** → **Reflection** → **Manual**.

Each phase may have multiple turns (chat_turn_limit); each turn = one LLM request/response.

---

## 3. Phase-by-Phase LLM Flow

### 3.1 DemandAnalysis

- **User role**: Chief Executive Officer (CEO)  
- **Assistant role**: Chief Product Officer (CPO)  
- **chat_turn_limit**: 10  

**LLM input (conceptually)**  
- System/context: `background_prompt` (ChatDev company intro) + `user_role_prompt` (CEO) + `assistant_role_prompt` (CPO), with `{task}` = task_prompt.  
- Phase-specific: phase_prompt asks to discuss **product modality** only and agree on one line (e.g. "PowerPoint").  
- Placeholders: `{task}` only; no codes yet.  

**LLM behavior**: CEO and CPO take turns; each assistant turn is one LLM call. Up to 10 turns until they agree on modality (e.g. "application").

---

### 3.2 LanguageChoose

- **User role**: CEO  
- **Assistant role**: Chief Technology Officer (CTO)  
- **chat_turn_limit**: 10  

**LLM input**  
- Same style: CEO/CTO role prompts + phase_prompt with `{task}`, `{modality}`, `{ideas}`.  
- Placeholders: task, modality (from DemandAnalysis), ideas.  

**LLM behavior**: Decide programming language (e.g. Python) and possibly GUI choice. Outputs feed next phase (language, gui).

---

### 3.3 Coding

- **User role**: CTO  
- **Assistant role**: Programmer  
- **chat_turn_limit**: 1  

**LLM input**  
- CTO vs Programmer role prompts + phase_prompt with `{task}`, `{language}`, `{gui}`, etc.  
- Placeholders: task, modality, ideas, language, gui (e.g. GUI required).  

**LLM behavior**: Single turn; LLM (Programmer) produces **code** (e.g. piece.py, board.py, main.py). This code becomes the `codes` placeholder for CodeReview and later phases.

---

### 3.4 CodeReviewComment (×3) + CodeReviewModification (×3)

- **CodeReviewComment**: user = Programmer, assistant = Code Reviewer, chat_turn_limit 1.  
- **CodeReviewModification**: user = Code Reviewer, assistant = Programmer, chat_turn_limit 1.  

**LLM input**  
- Role prompts (Programmer / Code Reviewer) + phase_prompt.  
- Placeholders: task, codes (current codebase), cycle_num (e.g. 3), cycle_index, modification_conclusion (from previous Modification phase).  

**LLM behavior**  
- **Comment**: One LLM call as Code Reviewer → review comments/suggestions.  
- **Modification**: One LLM call as Programmer → revised code; stored as new `codes` and optionally `modification_conclusion` for next cycle.  
- Repeated 3 times; after 3 cycles the final `codes` are used in EnvironmentDoc and Manual.

---

### 3.5 EnvironmentDoc

- **User role**: CTO  
- **Assistant role**: Programmer  
- **chat_turn_limit**: 1  

**LLM input**  
- Placeholders: task, modality, ideas, language, codes (final after CodeReview).  
- Phase_prompt: ask for env/setup documentation (e.g. requirements, run instructions).  

**LLM behavior**: One LLM call; output is documentation (e.g. requirements.txt, how to run).

---

### 3.6 Reflection

- **Assistant role**: CEO  
- **User role**: Counselor  

**LLM input**  
- Placeholders: `conversations` = **full prior dialogue** (DemandAnalysis through EnvironmentDoc).  
- Phase_prompt: reflect on the process and outcome.  

**LLM behavior**: LLM (as CEO) reflects on the development process; typically one or few turns.

---

### 3.7 Manual

- **User role**: CEO  
- **Assistant role**: CPO  
- **chat_turn_limit**: 1  

**LLM input**  
- Placeholders: task, modality, ideas, language, codes, requirements (from EnvironmentDoc).  
- Phase_prompt: write a **manual.md** (user manual: main functions, install, how to use/play).  

**LLM behavior**: One LLM call; output is the final user manual (e.g. manual.md content in metadata.main.py in expanded).

---

## 4. Data Flow Between Phases (Linear + One Loop)

The flow is **not fully linear**: phases run in order, but **CodeReview** is a **fixed 3-cycle loop** (Comment -> Modification -> Comment -> Modification -> Comment -> Modification). The rest is sequential.

```
task_prompt (Preprocessing)
    |
    v
DemandAnalysis  -->  modality (e.g. application)
    |
    v
LanguageChoose  -->  language (e.g. Python), gui
    |
    v
Coding  -->  codes (piece.py, board.py, main.py, ...)
    |
    v
+-- CodeReviewComment  -->  review comments
|       |
|       v
|   CodeReviewModification  -->  updated codes  ----+
|       |                                          |
|       v                                          |
+-- (repeat x3, then exit)  -----------------------+
    |
    v
EnvironmentDoc  -->  requirements, run/docs
    |
    v
Reflection  -->  uses full conversations (no new artifact)
    |
    v
Manual  -->  manual.md (final user-facing doc)
```

- **Linear**: Preprocessing -> DemandAnalysis -> LanguageChoose -> Coding -> (CodeReview block) -> EnvironmentDoc -> Reflection -> Manual.
- **Loop**: Inside CodeReview, Comment and Modification alternate; the cycle runs exactly 3 times (cycle_num=3), then execution continues to EnvironmentDoc.

---

## 5. Where This Appears in the Two Files

- **MAD_full_dataset.json**: First record has `trace.trajectory` = one long string (~345K chars) containing the full log: Preprocessing block (`**task_prompt**:`, config, etc.) and then `[timestamp INFO]` / `**phase_name** | X` blocks with role prompts and placeholders. The **actual LLM calls** are implicit in the log (each phase block shows the parameters sent into the chat and the resulting dialogue).  
- **MAD_full_dataset_expanded.json**: First record has `trajectory_expanded.metadata` (task_prompt, config_path, project_name, ChatDevConfig, ChatGPTConfig, and later `[chatting]`, `[RolePlaying]`, `main.py` etc. with parameter tables), and `trajectory_expanded.sections` = 12 phase_or_log entries, one per phase (DemandAnalysis, LanguageChoose, Coding, CodeReviewComment, CodeReviewModification x3, EnvironmentDoc, Reflection, Manual). Each section’s `content` holds the **role prompts and placeholders** that define the LLM input for that phase; the raw dialogue text is in the original trajectory.

---

## 6. Approximate LLM Call Count (Trace 0)

- DemandAnalysis: up to 10 turns → up to 10 calls.  
- LanguageChoose: up to 10 turns → up to 10 calls.  
- Coding: 1 turn → 1 call.  
- CodeReviewComment: 3 phases × 1 turn → 3 calls.  
- CodeReviewModification: 3 phases × 1 turn → 3 calls.  
- EnvironmentDoc: 1 call.  
- Reflection: 1 or a few calls.  
- Manual: 1 call.  

**Total**: on the order of **30+ LLM calls** for this trace (exact count would require parsing the raw trajectory for every turn).
