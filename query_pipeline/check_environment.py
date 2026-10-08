"""Run before starting the API. Reports import failures without a traceback wall."""
import importlib.metadata
import sys


def main():
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    for package in ("torch", "numpy", "langgraph", "fastapi", "pydantic"):
        try:
            print(package + ":", importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            print(package + ": not installed")
    try:
        import torch
    except (ImportError, OSError) as exc:
        print("\nPyTorch native-library import failed:", exc)
        if "iJIT_NotifyEvent" in str(exc):
            print("This is the known older-PyTorch/MKL incompatibility. In your active Conda environment run:")
            print('conda install "mkl=2024.0.0" "intel-openmp=2024.0.0" "numpy=1.26.4"')
            print("Restart Python after installation. Project code cannot repair a failing import torch.")
        return 1
    print("PyTorch import: OK")
    print("CUDA available:", torch.cuda.is_available())
    print("This project's model loaders use CPU, including with a CUDA-enabled PyTorch build.")
    if torch.__version__.split("+")[0] != "2.2.1":
        print("Note: this release targets PyTorch 2.2.1; another version is installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
