"""Enter the production worker bootstrap before loading the MoE discriminator."""

from importlib import import_module

from skyrl_train.worker_setup import configure_worker_process


def main() -> None:
    configure_worker_process()
    worker = import_module("tests.gpu.fault_injection.multi_node_moe_dispatch_worker")
    worker.main()


if __name__ == "__main__":
    main()
