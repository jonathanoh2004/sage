import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import numpy as np


def prep_shared_mem(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("input must be a dictionary mapping from names to shared memory objects")
    shr_mems, arrs_shr_mem = {}, {}

    for (key, arr) in mapping.items():
        if arr is not None:
            shm = SharedMemory(create=True, size=arr.nbytes)
            arr_shr_mem = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            arr_shr_mem[:] = arr[:]
            shr_mems[key] = shm
            arrs_shr_mem[key] = arr_shr_mem
        else:
            shr_mems[key] = None

    return shr_mems, arrs_shr_mem


def start_procs(procs):
    for proc in procs:
        proc.start()


def join_procs(procs):
    for proc in procs:
        proc.join()


def get_procs(shape, dim_iter, func, args, kwargs):
    n_cpus = multiprocessing.cpu_count()

    procs = []
    n_iters = shape[dim_iter]
    n_tasks_per_cpu = max(n_iters // n_cpus, 1)

    i_cpu, t_start, t_end = 0, 0, 0
    while i_cpu < n_cpus and t_end < n_iters:
        if i_cpu + 1 >= n_cpus or t_start + n_tasks_per_cpu > n_iters:
            t_end = n_iters
        else:
            t_end = t_start + n_tasks_per_cpu
        args.extend(t_start, t_end)
        proc = multiprocessing.Process(
            target=func,
            args=args,
            kwargs=kwargs,
        )
        procs.append(proc)
        t_start = t_end
        i_cpu += 1
