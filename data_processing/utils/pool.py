"""Process-pool driver: per-job error capture and progress printing."""
from concurrent.futures import ProcessPoolExecutor, as_completed


def _call(fn, job):
    try:
        return fn(*job), None
    except Exception as e:  # noqa: BLE001 - report and keep going
        return job[0] if isinstance(job, tuple) and job else job, repr(e)


def run_pool(fn, jobs, workers, every=25, initializer=None, initargs=(), label=""):
    """Run fn(*job) for every job; returns (results, errors) in completion order.

    errors holds (job[0], repr(exception)); a failing job never stops the pool.
    """
    jobs = [j if isinstance(j, tuple) else (j,) for j in jobs]
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=initializer,
                             initargs=initargs) as ex:
        futs = [ex.submit(_call, fn, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            res, err = fut.result()
            if err is not None:
                errors.append((res, err))
                print(f"  ERROR {res}: {err}", flush=True)
            else:
                results.append(res)
            if i % every == 0 or i == len(jobs):
                print(f"  {label}{i}/{len(jobs)} done ({len(errors)} errors)", flush=True)
    return results, errors
