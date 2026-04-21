"""Reference helper for building OpenFace from source inside WSL2 Ubuntu."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def setup_openface(project_root: str | None = None) -> str:
    """
    Build OpenFace from source in Ubuntu/WSL2.

    The default project root is the current repository root, so this script can
    be run from `/mnt/e/deception_pipeline/pipeline/setup_openface.py` inside WSL.
    """
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent)

    if sys.platform.startswith("win"):
        raise RuntimeError("Run setup_openface.py from WSL2 Ubuntu, not native Windows.")

    openface_dir = os.path.join(project_root, "OpenFace")
    build_dir = os.path.join(openface_dir, "build")
    binary = os.path.join(build_dir, "bin", "FeatureExtraction")

    if os.path.isfile(binary):
        print(f"OpenFace already built at {binary}")
        return binary

    apt_packages = [
        "build-essential",
        "cmake",
        "git",
        "wget",
        "libopencv-dev",
        "libopenblas-dev",
        "liblapack-dev",
        "libatlas-base-dev",
        "libboost-all-dev",
        "libgtk2.0-dev",
        "pkg-config",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "ffmpeg",
    ]

    print("[1/5] Installing system dependencies...")
    subprocess.run(["sudo", "apt-get", "update"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", *apt_packages], check=True)

    print("[2/5] Cloning OpenFace...")
    if not os.path.isdir(openface_dir):
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/TadasBaltrusaitis/OpenFace.git",
                openface_dir,
            ],
            check=True,
        )

    print("[3/5] Locating OpenCVConfig.cmake...")
    result = subprocess.run(["dpkg", "-L", "libopencv-dev"], capture_output=True, text=True)
    opencv_dir = "/usr/lib/x86_64-linux-gnu/cmake/opencv4"
    for line in result.stdout.splitlines():
        if line.endswith("OpenCVConfig.cmake"):
            opencv_dir = os.path.dirname(line.strip())
            break
    print(f"Using OpenCV_DIR={opencv_dir}")

    print("[4/5] Building OpenFace...")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-DCMAKE_BUILD_TYPE=RELEASE",
            "-DCMAKE_INSTALL_PREFIX=/usr/local",
            f"-DOpenCV_DIR={opencv_dir}",
            "..",
        ],
        cwd=build_dir,
        check=True,
    )
    subprocess.run(["make", f"-j{os.cpu_count() or 4}"], cwd=build_dir, check=True)

    print("[5/5] Downloading face models...")
    model_script = os.path.join(openface_dir, "download_models.sh")
    if os.path.isfile(model_script):
        subprocess.run(["bash", model_script], cwd=openface_dir, check=True)

    if not os.path.isfile(binary):
        raise RuntimeError(f"Build completed but binary was not found at {binary}")

    print(f"OpenFace built successfully: {binary}")
    return binary


if __name__ == "__main__":
    setup_openface()
