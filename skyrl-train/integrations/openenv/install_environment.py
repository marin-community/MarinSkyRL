"""Download the selected OpenEnv HTTP server image without starting a container."""

import argparse
import subprocess

ENV_IMAGES = {
    "atari-env": "ghcr.io/meta-pytorch/openenv-atari-env:sha-64d4b10",
    "coding-env": "ghcr.io/meta-pytorch/openenv-coding-env:sha-64d4b10",
    "echo-env": "ghcr.io/meta-pytorch/openenv-echo-env:sha-64d4b10",
    "openspiel-env": "ghcr.io/meta-pytorch/openenv-openspiel-base:sha-e622c7e",
    "sumo-rl-env": "ghcr.io/meta-pytorch/openenv-sumo-rl-env:sha-c25298c",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_name", nargs="?", choices=[*ENV_IMAGES, "finrl-env"])
    parser.add_argument("--image", help="An explicit image reference; necessary for FinRL")
    args = parser.parse_args()
    if args.image is not None and args.env_name is None:
        parser.error("--image requires an environment name")
    if args.env_name == "finrl-env" and args.image is None:
        parser.error("FinRL requires --image with a built HTTP server image")
    images = [args.image or ENV_IMAGES[args.env_name]] if args.env_name is not None else list(ENV_IMAGES.values())
    for image in images:
        subprocess.run(["docker", "pull", image], check=True)


if __name__ == "__main__":
    main()
