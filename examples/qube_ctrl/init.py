import argparse
import qubelsi.qube

def main(addr, path, bitfile=None):
    q = qubelsi.qube.Qube("10.5.0.14", "/home/miyo/adi_api_mod/")
    if bitfile is None:
        q.do_init(message_out=True) # without FPGA configuration
    else:
        q.do_init(bitfile=bitfile, message_out=True) # FPGA configuration with hoge.bit
                                   # required Vivado settings
    print(q.ad9082[0].get_jesd_status())
    print(q.ad9082[1].get_jesd_status())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('addr', help='IP address of ctrl-port of QuBE')
    parser.add_argument('--path', help='PATH of adi_api_mod', default='./adi_api_mod')
    parser.add_argument('--bitfile', help='PATH of adi_api_mod', default=None)
    args = parser.parse_args()

    main(args.addr, args.path, args.bitfile)

