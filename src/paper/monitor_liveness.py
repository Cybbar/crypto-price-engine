
import fcntl
import os
from pathlib import Path
from multiprocessing import get_context
from tempfile import TemporaryDirectory

from .monitor_lock import exclusive_monitor_lock


def monitor_is_running(database_path):
    db = Path(database_path).resolve()

    if not db.is_file():
        return False

    lock_path = Path(str(db) + ".monitor.lock")

    try:
        fd = os.open(
            lock_path,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(
            "Cannot inspect monitor lock. Trading blocked."
        ) from exc

    try:
        try:
            fcntl.flock(
                fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            # Another process currently holds the lock.
            return True

        # Lock was available: no active monitor owns it.
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False

    finally:
        os.close(fd)


def hold_test_monitor(db, ready, release):
    with exclusive_monitor_lock(db):
        ready.set()
        release.wait(15)


def main():
    print("MONITOR LIVENESS SAFETY TEST")
    print("-" * 40)

    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"
        db.touch()

        assert not monitor_is_running(db)
        print("No monitor running: BLOCKED")

        ctx = get_context("fork")
        ready = ctx.Event()
        release = ctx.Event()

        process = ctx.Process(
            target=hold_test_monitor,
            args=(db, ready, release),
        )

        process.start()

        try:
            if not ready.wait(10):
                raise RuntimeError("Monitor failed to start.")

            assert monitor_is_running(db)
            print("Active monitor: VERIFIED")

            # Simulate the monitoring process crashing.
            process.terminate()
            process.join(timeout=10)

            if process.is_alive():
                raise RuntimeError(
                    "Test monitor did not terminate."
                )

            assert not monitor_is_running(db)
            print("After monitor crash: BLOCKED")


        finally:

            release.set()

            if process.is_alive():
                process.terminate()

            process.join(timeout=5)

            if process.is_alive():
                process.kill()

                process.join(timeout=3)

            if process.is_alive():
                raise RuntimeError(

                    "Test process could not be stopped."

                )

    print("MONITOR LIVENESS TEST: PASSED")
    print("Real exchange orders: NONE")


if __name__ == "__main__":
    main()
