import argparse
import os
import subprocess
import sys

import determined as det


def main(train_entrypoint: str) -> int:
    info = det.get_cluster_info()
    os.environ["USE_TORCH_DIST"] = "True"

    chief_ip = info.container_addrs[0]
    os.environ["DET_CHIEF_IP"] = chief_ip

    torch_distributed_cmd = [
        "python3",
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(len(info.container_addrs)),
        "--nproc_per_node",
        str(len(info.slot_ids)),
        "--node_rank",
        str(info.container_rank),
        "--master_addr",
        chief_ip,
        "--module"
    ]

    harness_cmd = [
        "determined.exec.harness",
        "--train-entrypoint",
        train_entrypoint,
    ]

    return subprocess.Popen(
        torch_distributed_cmd + harness_cmd
    ).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_entrypoint", type=str)
    args = parser.parse_args()
    sys.exit(main(args.train_entrypoint))
