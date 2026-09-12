import os
import signal
import subprocess
import sys
import time


def worker_count():
    count = int(os.getenv('WORKER_PROCESSES', '1'))
    if count < 1:
        raise ValueError('WORKER_PROCESSES must be at least 1')
    return count


def stop_workers(workers):
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()

    deadline = time.monotonic() + 10
    for worker in workers:
        timeout = max(0, deadline - time.monotonic())
        try:
            worker.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


def main():
    workers = []
    stopping = False

    def handle_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        for _ in range(worker_count()):
            workers.append(subprocess.Popen([
                sys.executable,
                'object_classification_yolov8.py',
            ]))

        while not stopping:
            for worker in workers:
                return_code = worker.poll()
                if return_code is not None:
                    return return_code or 1
            time.sleep(1)
        return 0
    finally:
        stop_workers(workers)


if __name__ == '__main__':
    sys.exit(main())