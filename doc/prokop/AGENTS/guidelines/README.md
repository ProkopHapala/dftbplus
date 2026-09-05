# doc/prokop/AGENTS/guidelines/

Concrete, repo-specific guidelines derived from code reviews and post-mortems.
Each document generalizes specific violations into rules that prevent
recurrence.

- **efficiency.md** — 12 rules + 6 general principles for writing efficient
  numerical code, derived from an 18-point review of the CPU DFTB pipeline
  (see `doc/prokop/chats/CPU_Optimization.chat.md`). Covers data lifetime,
  allocation discipline, hot-path data structures, mathematical structure,
  interpolation, derivatives, unit checking, warm starts, and benchmarking
  methodology.
