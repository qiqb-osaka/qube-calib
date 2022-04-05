"""
AWG から 50MHz の余弦波を出力して, 信号処理モジュールを全て無効にしてキャプチャします.
"""
import sys
import pathlib
import argparse

lib_path = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(lib_path)
from e7awgsw import *
from e7awgsw.labrad import *

IP_ADDR = '10.0.0.16'

def create_awg_ctrl(use_labrad, server_ip_addr):
    if use_labrad:
        return RemoteAwgCtrl(server_ip_addr, IP_ADDR)
    else:
        return AwgCtrl(IP_ADDR)


def main(use_labrad, server_ip_addr, awgs):
    with create_awg_ctrl(use_labrad, server_ip_addr) as awg_ctrl:
        awg_ctrl.terminate_awgs(*awgs)
        print('end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipaddr')
    parser.add_argument('--awgs')
    parser.add_argument('--server-ipaddr')
    parser.add_argument('--labrad', action='store_true')
    args = parser.parse_args()

    if args.ipaddr is not None:
        IP_ADDR = args.ipaddr

    server_ip_addr = 'localhost'
    if args.server_ipaddr is not None:
        server_ip_addr = args.server_ipaddr

    awgs = AWG.all()
    if args.awgs is not None:
        awgs = [AWG.of(int(x)) for x in args.awgs.split(',')]

    main(args.labrad, server_ip_addr, awgs)
