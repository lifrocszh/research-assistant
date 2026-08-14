# Project instructions

Use these skills for every task in this repository:

- `caveman` — terse, high-density communication. Default: `full`.
- `ponytail` — minimal implementation and YAGNI. Default: `full`.

Load both at session start and keep them active for every response and code
change. Explicit user commands such as `stop caveman`, `stop ponytail`, or
`normal mode` override this requirement for the requested scope.

Write focused unit tests for every code change. Run `uv run pytest` before
completion.
