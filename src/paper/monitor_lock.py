
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def exclusive_monitor_lock(database_path):
    db = Path(database_path).resolve()

    if not db.is_file():
        raise RuntimeError("Paper database missing.")

    lock_path = Path(str(db) + ".monitor.lock")

    fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )

    with os.fdopen(fd, "r+") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another paper monitor is already running."
            ) from exc

        try:
            yield
        finally:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_UN,
            )


def main():
    print("MONITOR PROCESS LOCK TEST")
    print("-" * 40)

    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"
        db.touch()

        with exclusive_monitor_lock(db):
            print("First monitor: LOCK ACQUIRED")

            try:
                with exclusive_monitor_lock(db):
                    raise AssertionError(
                        "Second monitor acquired the lock."
                    )
            except RuntimeError:
                print("Second monitor: BLOCKED")

        with exclusive_monitor_lock(db):
            print("After first exits: LOCK ACQUIRED")

    print("PROCESS LOCK TEST: PASSED")
    print("Real exchange orders: NONE")


if __name__ == "__main__":
    main()
