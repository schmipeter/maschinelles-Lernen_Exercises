"""
Utility helpers to enforce a *clean* Git working tree (and, optionally,
a fully up‑to‑date branch) before running expensive or irreversible
operations such as **model training**.

Add the decorator ``@require_clean_repo`` to the entry‑point of your
training script (or to any function you want to protect).  Example
------------------------------------------------------------------
```python
from MECH_M_DUAL_2_MLB_DATA.git_safeguards import require_clean_repo

@require_clean_repo(check_remote=True)  # aborts if repo dirty **or** behind remote
def main():
    ...  # existing training code

if __name__ == "__main__":
    main()
```
If the repository contains uncommitted changes (including *untracked*
files) an informative ``DirtyRepositoryError`` is raised.  When
``check_remote=True`` the decorator additionally aborts if the current
branch is behind its upstream tracking branch, raising an
``OutOfDateRepositoryError``.

The implementation prefers **GitPython** (``pip install GitPython``),
but falls back to plain shell commands if the library is unavailable.
"""
from MECH_M_DUAL_2_MLB_DATA.git_safeguards import require_clean_repo

@require_clean_repo(check_remote=True)
def main():


from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Callable, Optional
import subprocess

try:
    from git import Repo, GitCommandError  # type: ignore

    _GITPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover – fallback path
    _GITPYTHON_AVAILABLE = False


class DirtyRepositoryError(RuntimeError):
    """Raised when the repository has uncommitted changes."""


class OutOfDateRepositoryError(RuntimeError):
    """Raised when the local branch is behind its remote counterpart."""


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _repo_root(start: Optional[Path] = None) -> Path:
    """Return the root directory of the Git repository containing *start*."""
    cwd = Path.cwd() if start is None else Path(start).resolve()
    if _GITPYTHON_AVAILABLE:
        repo = Repo(cwd, search_parent_directories=True)
        return Path(repo.git.rev_parse("--show-toplevel"))

    root = (
        subprocess.check_output(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"], text=True)
        .strip()
    )
    return Path(root)


def _is_dirty(root: Path) -> bool:
    """Return *True* when the working tree under *root* has uncommitted changes."""
    if _GITPYTHON_AVAILABLE:
        repo = Repo(root)
        return repo.is_dirty(untracked_files=True)

    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    return bool(status.strip())


def _is_behind_remote(root: Path, branch: str) -> bool:
    """Return *True* if *branch* is behind its upstream tracking branch."""
    if _GITPYTHON_AVAILABLE:
        repo = Repo(root)
        try:
            return next(repo.iter_commits(f"{branch}..@{{u}}"), None) is not None
        except GitCommandError:
            # No upstream configured – treat as up to date
            return False

    # Fallback to porcelain rev‑list analysis
    try:
        subprocess.check_call(
            ["git", "-C", str(root), "remote", "update", "--prune"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False  # network error or no remotes configured

    behind, _ahead = map(
        int,
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "rev-list",
                "--left-right",
                "--count",
                f"{branch}...@{{u}}",
            ],
            text=True,
        )
        .strip()
        .split()
    )
    return behind > 0


# -----------------------------------------------------------------------------
# Public decorator
# -----------------------------------------------------------------------------

def require_clean_repo(*, check_remote: bool = False) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator that **aborts** execution if the Git workspace is *dirty*.

    Parameters
    ----------
    check_remote : bool, default=False
        Additionally verify that the current branch is *not* behind its
        upstream tracking branch.  If it is, an
        :class:`OutOfDateRepositoryError` is raised.
    """

    def decorator(func: Callable[..., _T]) -> Callable[..., _T]:  # type: ignore[name-defined]
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            root = _repo_root()

            if _is_dirty(root):
                raise DirtyRepositoryError(
                    f"Repository at {root} has uncommitted changes. "
                    "Commit, stash, or discard them before continuing."
                )

            if check_remote:
                branch = (
                    subprocess.check_output(
                        ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], text=True
                    )
                    .strip()
                )
                if _is_behind_remote(root, branch):
                    raise OutOfDateRepositoryError(
                        f"Branch '{branch}' is behind its upstream. "
                        "Pull the latest changes before continuing."
                    )

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# -----------------------------------------------------------------------------
# Typing helpers (avoids importing typing_extensions)
# -----------------------------------------------------------------------------
from typing import TypeVar
_T = TypeVar("_T")

if __name__ == "__main__":
    main()