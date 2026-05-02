# SymTorch environment setup

## Purpose

Create an isolated environment for `torch-symbolic` / `symtorch` without disturbing the transfer-learning environment used for the MTSLSTM experiments.

## New environment

- Environment name: `mtslstm_symtorch`
- Config file: `symtorch_env.yml`
- Activate with:

```bash
conda activate mtslstm_symtorch
```

## Why a new environment was needed

The transfer-learning environment was based on Python `3.10`, while `torch-symbolic` currently requires Python `>= 3.11`.

To avoid dependency conflicts, the new environment keeps the same scientific stack where possible and only changes the minimum needed pieces:

- Python upgraded from `3.10` to `3.11`
- `torch-symbolic` added
- PyTorch reinstalled inside the isolated environment

## Installed core versions

- Python: `3.11.15`
- Torch: `2.5.1+cu124`
- TorchVision: `0.20.1`
- TorchAudio: `2.5.1`
- SymTorch package: `torch-symbolic 1.0.1.post1`
- Top-level import module: `symtorch`

## Important caveat

`symtorch` depends on `PySR`, which triggers `juliacall` on first import. That first import may need to download and initialize a Julia runtime if one is not already available locally.

In other words:

- `pip install torch-symbolic` is complete
- first `from symtorch import SymbolicModel` may still trigger Julia-side setup

## First import test

The package install succeeded, but the first import attempted to initialize Julia and required network access for Julia discovery/setup. This is a runtime bootstrap issue, not a Python package conflict.

Example test command:

```bash
conda run -n mtslstm_symtorch python -c "from symtorch import SymbolicModel; print(SymbolicModel)"
```

## Successful initialization

Julia was initialized successfully inside the new environment.

- Julia executable:
  `/home/kongw0a/miniconda3/envs/mtslstm_symtorch/julia_env/pyjuliapkg/install/bin/julia`
- Julia version:
  `1.12.6`

To make the first import deterministic, this command worked:

```bash
PYTHON_JULIACALL_EXE=/home/kongw0a/miniconda3/envs/mtslstm_symtorch/julia_env/pyjuliapkg/install/bin/julia \
conda run -n mtslstm_symtorch python -c "from symtorch import SymbolicModel; print(SymbolicModel)"
```

Expected output:

```text
<class 'symtorch.SymbolicModel.SymbolicModel'>
```

## Files

- Environment YAML: `symtorch_env.yml`
- This note: `symtorch_env_notes.md`
