import os.path

import numpy as np
import pytest
import nilearn
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from tedana import io, utils
from tedana.tests.utils_sage import get_test_data_path_sage
from tedana.workflows.sage import concurrency_sage


def test_prep_shared_mem_with_arr():
    mapping = {
        "A": np.array([1, 2, 3], dtype=np.float64),
        "B": None,
        "C": np.array([4, 5, 6], dtype=np.float64),
        "D": np.array([7, 8, 9], dtype=np.float64),
    }
    res_shr_mems, res_arrs_shr_mem = concurrency_sage.prep_shared_mem_with_arr(mapping)
    assert isinstance(res_shr_mems, dict)
    assert isinstance(res_arrs_shr_mem, dict)
    for (key, val) in mapping:
        if mapping[key] is None:
            assert res_shr_mems[key] is None
        else:
            assert isinstance(res_shr_mems[key], SharedMemory)
            assert np.array_equal(res_arrs_shr_mem[key], mapping[key])
            assert res_arrs_shr_mem[key].shape == mapping[key].shape
            assert res_arrs_shr_mem[key].dtype == mapping[key].dtype

    concurrency_sage.close_and_unlink_shr_mem(res_shr_mems)


def test_prep_shared_mem_with_name():
    shms = {
        "a": SharedMemory(create=True, size=1),
        "b": SharedMemory(create=True, size=1),
        "c": SharedMemory(create=True, size=1),
    }
    mapping = {"A": shms["a"].name, "B": shms["b"].name, "C": shms["c"].name}
    shape = (3,)
    dtype = np.float64
    res_shr_mems, res_arrs_shr_mem = concurrency_sage.prep_shared_mem_with_name(
        mapping, shape, dtype
    )

    assert isinstance(res_shr_mems, dict)
    assert isinstance(res_arrs_shr_mem, dict)
    for key in mapping:
        if mapping[key] is None:
            assert res_shr_mems[key] is None
            assert res_arrs_shr_mem[key] is None
        else:
            assert isinstance(res_shr_mems[key], SharedMemory)
            assert isinstance(res_arrs_shr_mem[key], np.ndarray)
            assert res_arrs_shr_mem[key].shape == shape
            assert res_arrs_shr_mem[key].dtype == dtype

    concurrency_sage.close_shr_mem(res_shr_mems)
    concurrency_sage.close_and_unlink_shr_mem(shms)


def test_get_procs():
    n_iters = 10
    func = print
    n_procs = 2 if multiprocessing.cpu_count() > 1 else 1
    args = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    kwargs = {"A": "name1", "B": "name2", "C": "name3"}
    procs = concurrency_sage.get_procs(n_iters, func, n_procs, args, kwargs)
    assert isinstance(procs, list)
    if n_procs == 1:
        assert len(procs) == 1
        assert procs[0]._args == args + (0, 10)
    elif n_procs == 2:
        assert len(procs) == 2
        assert procs[0]._args == args + (0, 5)
        assert procs[1]._args == args + (5, 10)
    else:
        assert False
    for proc in procs:
        assert proc._target == print
        assert proc._kwargs == kwargs
