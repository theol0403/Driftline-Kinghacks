# ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.

# Style

Use deepwiki/context7 for docs. Use the uv .venv in this folder. Always take oppourtunities to clean up/refactor/streamline the code, there should always be one canonical happy path in the code, with no workarounds/monkey patches. Always consider how your changes fit in the overall architecture of the repo and if implementing it elsewhere can improve DRY/composition.
