# 10 — Git Layout and `.gitignore`

The supplied `.gitignore` is for:
```text
nemo_gym/
├── .gitignore            # supplied
├── Gym/                  # upstream clone, likely has its own .git/
├── other-projects/
└── ...
```

## Crucial Git behavior
A **parent `.gitignore` does not govern files committed within a nested `Gym/.git` repository**. If you want to keep `Gym/` as its own repo, maintain a separate `Gym/.gitignore` there, or put local patterns in `Gym/.git/info/exclude`.

If you initialize a Git repository at `nemo_gym/` and also retain `Gym/.git`, Git may treat `Gym/` as an embedded repository. The parent ignore provided here excludes `Gym/` by default to avoid accidental nested-repo commits. Remove that rule only if you intentionally use a Git submodule or restructure the repositories.

## Keep outputs untracked
For the **Gym repo itself**, consider adding these to `Gym/.gitignore` or `Gym/.git/info/exclude`:
```gitignore
/results/
/reports/
*.log
.env
.env.*
!.env.example
__pycache__/
.pytest_cache/
.venv/
```
Do not ignore benchmark fixture JSONL indiscriminately: small reviewed fixtures may need to be versioned. Store large/raw datasets externally or via an approved data-versioning method.

## Validate ignores
```bash
cd ~/nemo_gym
git check-ignore -v Gym/ || true
cd Gym
git check-ignore -v results/llm02/example.jsonl || true
git status --short
```
