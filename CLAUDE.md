# Workflow Orchestration

## Plan Mode Default
Enter plan mode for non-trivial tasks. Use `EnterPlanMode` tool for any feature/change that could have multiple approaches.

## Subagent Strategy
Use subagents liberally to keep main context window clean. For research, exploration, and parallel tasks, prefer `Agent` tool with appropriate subagent_type (`Explore`, `Plan`, `general-purpose`).

## Self-Improvement Loop
Update `tasks/lessons.md` after corrections. When I make a mistake and get corrected, record the lesson so I don't repeat it.

## Verification Before Done
Never mark a task complete without proving it works. Run tests, check output, verify behavior before marking TodoWrite items complete.

## Demand Elegance (Balanced)
Ask "is there a more elegant way?" After initial implementation, briefly consider if there's a cleaner approach - but don't over-engineer.

## Autonomous Bug Fixing
Just fix bugs, don't ask for hand-holding. When a test fails or error occurs, attempt diagnosis and fix before asking for help.

---

# Task Management

- **Plan First**: Before coding, outline approach in `tasks/todo.md`
- **Verify Plan**: Use `EnterPlanMode` for complex features
- **Track Progress**: Use `TodoWrite` to show real-time progress
- **Document Blockers**: Note any issues in `tasks/lessons.md`

---

# Core Principles

- **Simplicity First**: Prefer simple, readable solutions
- **No Laziness**: Never leave placeholder code or TODOs without explicit permission
- **Minimal Impact**: Make smallest changes needed to achieve goal
