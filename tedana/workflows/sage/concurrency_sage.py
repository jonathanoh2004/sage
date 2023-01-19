import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import numpy as np


def prep_shared_mem_with_arr(mapping):
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


def prep_shared_mem_with_name(mapping, shape, dtype):
    shr_mems, arrs_shr_mem = {}, {}

    for (key, name) in mapping.items():
        if name is not None:
            shm = SharedMemory(name=name)
            shr_mems[key] = shm
        else:
            shr_mems[key] = None

    for key in shr_mems:
        if shr_mems[key] is None:
            arrs_shr_mem[key] = None
        else:
            arrs_shr_mem[key] = np.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=shr_mems[key].buf,
            )

    return shr_mems, arrs_shr_mem


def get_procs(shape, dim_iter, func, n_procs, args, kwargs):
    n_cpus = multiprocessing.cpu_count()
    n_cpus = n_procs if n_procs > 0 and n_procs < n_cpus else n_cpus

    procs = []
    n_iters = shape[dim_iter]
    n_tasks_per_cpu = max(n_iters // n_cpus, 1)
    args_orig = args
    i_cpu, t_start, t_end = 0, 0, 0
    while i_cpu < n_cpus and t_end < n_iters:
        if i_cpu + 1 >= n_cpus or t_start + n_tasks_per_cpu > n_iters:
            t_end = n_iters
        else:
            t_end = t_start + n_tasks_per_cpu
        args = args_orig + (t_start, t_end)
        proc = multiprocessing.Process(
            target=func,
            args=args,
            kwargs=kwargs,
        )
        procs.append(proc)
        t_start = t_end
        i_cpu += 1
    return procs


def start_procs(procs):
    for proc in procs:
        proc.start()


def join_procs(procs):
    for proc in procs:
        proc.join()


def close_shr_mem(mapping):
    for shr_mem in mapping.values():
        if shr_mem is not None:
            shr_mem.close()


def close_and_unlink_shr_mem(mapping):
    for shr_mem in mapping.values():
        if shr_mem is not None:
            shr_mem.close()
            shr_mem.unlink()
