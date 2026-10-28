"""Exercise the CMS libraries and real CUDA kernels on the allocated EAF GPU."""
import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch


def main():
    versions = {}
    for name in (
        "numpy", "scipy", "pandas", "sklearn", "numba", "h5py", "pyarrow",
        "matplotlib", "seaborn", "streamlit", "awkward", "uproot", "vector", "hist",
        "mplhep", "iminuit", "lmfit", "ROOT", "pythia8", "lhapdf", "XRootD",
        "fsspec_xrootd", "torchvision", "lightning", "accelerate", "datasets",
        "transformers", "wandb", "hepdata_lib", "hepdata_validator", "heptokens",
    ):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "import OK"))
    print(json.dumps(versions, indent=2), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: start an EAF GPU session and check nvidia-smi.")
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    # Matrix multiplication, backward pass, and NumPy interoperability.
    x = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(32, 32)
    x.requires_grad_()
    result = x @ x.T
    result.mean().backward()
    torch.cuda.synchronize()
    expected = x.detach().cpu().numpy() @ x.detach().cpu().numpy().T
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=1e-5)
    assert torch.isfinite(x.grad).all().item()
    # CuPy compilation and BLAS must work, not just import successfully.
    import cupy as cp
    a = cp.arange(1024, dtype=cp.float32).reshape(32, 32)
    np.testing.assert_allclose(cp.asnumpy(a @ a.T), result.detach().cpu().numpy(), rtol=1e-5)
    np.testing.assert_allclose(
        cp.asnumpy(cp.sqrt(cp.arange(16, dtype=cp.float32))),
        np.sqrt(np.arange(16, dtype=np.float32)),
    )
    cp.cuda.Stream.null.synchronize()
    print(f"CuPy {cp.__version__}; CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}", flush=True)
    # ROOT JIT plus a ROOT-file round trip through uproot.
    import ROOT
    import uproot
    assert ROOT.gInterpreter.Declare("int cms_gpu_smoke() { return 42; }")
    assert ROOT.cms_gpu_smoke() == 42
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "check.root"
        with uproot.recreate(path) as output:
            output["events"] = {"x": np.arange(10, dtype=np.float64)}
        with uproot.open(path) as source:
            np.testing.assert_array_equal(source["events"]["x"].array(library="np"), np.arange(10))
    print(f"ROOT {ROOT.gROOT.GetVersion()}: JIT and ROOT-file round trip passed")
    print("PASS: package imports, PyTorch GPU forward/backward, CuPy GPU kernels, and ROOT I/O")


if __name__ == "__main__":
    main()
