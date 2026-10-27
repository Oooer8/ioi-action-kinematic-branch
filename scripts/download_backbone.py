"""Download only when explicitly run by the repository user."""

import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/Wan2.1-T2V-1.3B")
    args = parser.parse_args()
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=args.output)


if __name__ == "__main__":
    main()
