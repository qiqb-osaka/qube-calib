qubecalib is a Python library for quantum control experiments using QuBE/QuEL.


## Requirements

- Python 3.9+


## Installation

You can install qubecalib with the following steps.

### 1. Install Python (optional)

Confirm that Python 3.9 or later is installed.

```bash
python --version
```

If not, install an appropriate Python version using [pyenv](https://github.com/pyenv/pyenv) or other tools.


### 2. Create and activate a virtual environment

Create a dedicated virtual environment using venv.

```bash
# Move to your workspace
cd YOUR_WORKSPACE

# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install qubecalib

Install qubecalib from the GitHub repository using pip.

```bash
pip install git+https://github.com/qiqb-osaka/qube-calib.git
```

To install a specific version (x.y.z), run the following command.

```bash
pip install git+https://github.com/qiqb-osaka/qube-calib.git@x.y.z
```

Check available versions on the [release page](https://github.com/qiqb-osaka/qube-calib/releases).


## Notes

- `QubeServer.py` has been moved to the [qube-server](https://github.com/qiqb-osaka/qube-server) repository.