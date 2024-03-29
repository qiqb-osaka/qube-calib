## Pre-requirements (Recommended)

- libffi-dev libbz2-dev libreadline-dev libsqlite3-dev (for Ubuntu 20.04)
- Pyenv
- Python 3.9.10
- Pipenv
- e7awg_sw
- adi_api_mod

```
sudo apt install libffi-dev libbz2-dev libreadline-dev libsqlite3-dev # ex. for Ubuntu 20.04
pyenv install 3.9.10
git clone https://github.com/qiqb-osaka/qube-calib.git --recursive
cd qube-calib
cd adi_api_mod/src
make
cd ../v1.0.6/src
make
cd ../../..
pipenv shell
pipenv install
```

## Quick start

To setup QuBe,

```
pipenv shell
source /tools/Xilinx/Vivado/2020.1/settings64.sh # depends on your installed path
python examples/qube_ctrl/init.py 10.5.0.14 --bitfile=/home/miyo/bin/06805e.bit
python examples/qube_ctrl/ad9082_read_info.py 10.5.0.14 
python e7awg_sw/examples/send_recv/send_recv.py --ipaddr=10.1.0.14 
```

In this example, `10.5.0.14` and `10.1.0.14` are IP-addresses of ctrl(1GbE) and data(10GbE), respectively.

## With Jupyter-Lab

```
pipenv shell
jupyter lab --ip=* --no-browser --NotebookApp.token='' 
```


## Setup Pyenv

```
git clone https://github.com/pyenv/pyenv.git .pyenv
```

Add the following in `.bashrc`,

```
export PIPENV_VENV_IN_PROJECT=true

if [ -e $HOME/.pyenv ]; then
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init --path)"
  #eval "$(pyenv init -)"
fi
```

After that, execute `source ~/.bashrc`.
