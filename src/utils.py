import re

import os
import signal
import logging
import sys
from time import sleep, time
from random import random, randint
from multiprocessing import JoinableQueue, Event, Process
from queue import Empty
from typing import Optional

logger = logging.getLogger(__name__)


def re_findall(pattern, string):
    return [m.groupdict() for m in re.finditer(pattern, string)]


class Task:
    def __init__(self, function, *args, **kwargs) -> None:
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        return self.function(*self.args, **self.kwargs)


class CallbackGenerator:
    def __init__(self, generator, callback):
        self.generator = generator
        self.callback = callback

    def __iter__(self):
        if self.callback is not None and callable(self.callback):
            for t in self.generator:
                self.callback(t)
                yield t
        else:
            yield from self.generator


def start_worker(q: JoinableQueue, stop_event: Event):  # TODO make class?
    logger.info('Starting worker...')
    while True:
        if stop_event.is_set():
            logger.info('Worker exiting because of stop_event')
            break
        # We set a timeout so we loop past 'stop_event' even if the queue is empty
        try:
            task = q.get(timeout=.01)
        except Empty:
            # Run next iteration of loop
            continue

        # Exit if end of queue
        if task is None:
            logger.info('Worker exiting because of None on queue')
            q.task_done()
            break

        try:
            task.run()  # Do the task
        except:  # Will also catch KeyboardInterrupt
            logger.exception(f'Failed to process task {task}', )
            # Can implement some kind of retry handling here
        finally:
            q.task_done()


class InterruptibleTaskPool:

    # https://the-fonz.gitlab.io/posts/python-multiprocessing/
    def __init__(self,
                 tasks=None,
                 num_workers=None,

                 callback=None,  # Fired on start
                 max_queue_size=1,
                 grace_period=2,
                 kill_period=30,
                 ):

        self.tasks = CallbackGenerator(
            [] if tasks is None else tasks, callback)
        self.num_workers = os.cpu_count() if num_workers is None else num_workers

        self.max_queue_size = max_queue_size
        self.grace_period = grace_period
        self.kill_period = kill_period

        # The JoinableQueue has an internal counter that increments when an item is put on the queue and
        # decrements when q.task_done() is called. This allows us to wait until it's empty using .join()
        self.queue = JoinableQueue(maxsize=self.max_queue_size)
        # This is a process-safe version of the 'panic' variable shown above
        self.stop_event = Event()

        # n_workers: Start this many processes
        # max_queue_size: If queue exceeds this size, block when putting items on the queue
        # grace_period: Send SIGINT to processes if they don't exit within this time after SIGINT/SIGTERM
        # kill_period: Send SIGKILL to processes if they don't exit after this many seconds

        # self.on_task_complete = on_task_complete
        # self.raise_after_interrupt = raise_after_interrupt

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def start(self) -> None:
        def handler(signalname):
            """
            Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
            Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
            """
            def f(signal_received, frame):
                raise KeyboardInterrupt(f'{signalname} received')
            return f

        # This will be inherited by the child process if it is forked (not spawned)
        signal.signal(signal.SIGINT, handler('SIGINT'))
        signal.signal(signal.SIGTERM, handler('SIGTERM'))

        procs = []

        for i in range(self.num_workers):
            # Make it a daemon process so it is definitely terminated when this process exits,
            # might be overkill but is a nice feature. See
            # https://docs.python.org/3.8/library/multiprocessing.html#multiprocessing.Process.daemon
            p = Process(name=f'Worker-{i:02d}', daemon=True,
                        target=start_worker, args=(self.queue, self.stop_event))
            procs.append(p)
            p.start()

        try:
            # Put tasks on queue
            for task in self.tasks:
                logger.info(f'Put task {task} on queue')
                self.queue.put(task)

            # Put exit tasks on queue
            for i in range(self.num_workers):
                self.queue.put(None)

            # Wait until all tasks are processed
            self.queue.join()

        except KeyboardInterrupt:
            logger.warning('Caught KeyboardInterrupt! Setting stop event...')
            # raise # TODO add option
        finally:
            self.stop_event.set()
            t = time()
            # Send SIGINT if process doesn't exit quickly enough, and kill it as last resort
            # .is_alive() also implicitly joins the process (good practice in linux)
            while alive_procs := [p for p in procs if p.is_alive()]:
                if time() > t + self.grace_period:
                    for p in alive_procs:
                        os.kill(p.pid, signal.SIGINT)
                        logger.warning(f'Sending SIGINT to {p}')
                elif time() > t + self.kill_period:
                    for p in alive_procs:
                        logger.warning(f'Sending SIGKILL to {p}')
                        # Queues and other inter-process communication primitives can break when
                        # process is killed, but we don't care here
                        p.kill()
                sleep(.01)

            sleep(.1)
            for p in procs:
                logger.info(f'Process status: {p}')


def jaccard(x1, x2, y1, y2):
    # Calculate jaccard index
    intersection = max(0, min(x2, y2)-max(x1, y1))
    filled_union = max(x2, y2) - min(x1, y1)
    return intersection/filled_union if filled_union > 0 else 0
