import re
import asyncio
import os


class Job:
    def __init__(self, function, *args, **kwargs) -> None:
        self.function = function
        self.args = args
        self.kwargs = kwargs

        self.result = None


class InterruptibleThreadPool:
    def __init__(self,
                 num_workers=None,
                 loop=None,
                 shutdown_message='\nAttempting graceful shutdown, press Ctrl+C again to exit...',
                 on_job_complete=None,  # Useful for monitoring progress
                 raise_after_interrupt=False,
                 ) -> None:
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        self.loop = asyncio.get_event_loop() if loop is None else loop
        self.shutdown_message = shutdown_message

        self.sem = asyncio.Semaphore(num_workers)

        self.jobs = []

        self.on_job_complete = on_job_complete
        self.raise_after_interrupt = raise_after_interrupt

    async def _sync_to_async(self, job):
        async with self.sem:  # Limit number of parallel tasks
            job.result = await self.loop.run_in_executor(None, job.function, *job.args, **job.kwargs)

            if callable(self.on_job_complete):
                self.on_job_complete(job)

            return job

    def add_job(self, job):
        self.jobs.append(job)

    def run(self):
        try:
            tasks = [
                # creating task starts coroutine
                asyncio.ensure_future(self._sync_to_async(job))
                for job in self.jobs
            ]

            # https://stackoverflow.com/a/42097478
            self.loop.run_until_complete(
                asyncio.gather(*tasks, return_exceptions=True)
            )

        except KeyboardInterrupt:
            # Optionally show a message if the shutdown may take a while
            print(self.shutdown_message, flush=True)

            # Do not show `asyncio.CancelledError` exceptions during shutdown
            # (a lot of these may be generated, skip this if you prefer to see them)
            def shutdown_exception_handler(loop, context):
                if "exception" not in context \
                        or not isinstance(context["exception"], asyncio.CancelledError):
                    loop.default_exception_handler(context)
            self.loop.set_exception_handler(shutdown_exception_handler)

            # Handle shutdown gracefully by waiting for all tasks to be cancelled
            cancelled_tasks = asyncio.gather(
                *asyncio.all_tasks(loop=self.loop), loop=self.loop, return_exceptions=True)
            cancelled_tasks.add_done_callback(lambda t: self.loop.stop())
            cancelled_tasks.cancel()

            # Keep the event loop running until it is either destroyed or all
            # tasks have really terminated
            while not cancelled_tasks.done() and not self.loop.is_closed():
                self.loop.run_forever()

            if self.raise_after_interrupt:
                raise
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

        return self.jobs


def re_findall(pattern, string):
    return [m.groupdict() for m in re.finditer(pattern, string)]
