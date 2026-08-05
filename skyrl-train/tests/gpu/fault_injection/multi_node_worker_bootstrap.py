"""Enter the production Ray worker bootstrap before loading the NCCL worker."""

from importlib import import_module

from skyrl_train.worker_setup import configure_worker_process


def main() -> None:
    configure_worker_process()
    worker = import_module("tests.gpu.fault_injection.multi_node_collective_stall_worker")
    worker.main()


if __name__ == "__main__":
    main()
