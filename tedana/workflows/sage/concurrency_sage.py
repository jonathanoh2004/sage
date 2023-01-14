def _prep_shared_mem(mapping):
    """
    Takes in numpy arrays and corresponding keys
    Outputs one dictionary mapping from keys to shared memory names
    as well as a second dictionary mapping from keys to numpy arrays
    referring to same underlying memory
    """
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


def _start_and_join_procs(n_samps, n_echos, n_vols, dtype, fittype, kwargs):
    n_cpus = multiprocessing.cpu_count()

    procs = []
    n_tasks_per_cpu = max(n_vols // n_cpus, 1)

    i_cpu, t_start, t_end = 0, 0, 0
    while i_cpu < n_cpus and t_end < n_vols:
        if i_cpu + 1 >= n_cpus or t_start + n_tasks_per_cpu > n_vols:
            t_end = n_vols
        else:
            t_end = t_start + n_tasks_per_cpu
        proc = multiprocessing.Process(
            target=fit_nonlinear_sage,
            args=(n_samps, n_echos, n_vols, dtype, t_start, t_end, fittype),
            kwargs=kwargs,
        )
        procs.append(proc)
        proc.start()
        t_start = t_end
        i_cpu += 1

    for proc in procs:
        proc.join()
