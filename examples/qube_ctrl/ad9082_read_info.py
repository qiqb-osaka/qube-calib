import argparse
import qubelsi.qube

def main(addr, path):
    print("addr={}, path={}".format(addr, path))
    q = qubelsi.qube.Qube(addr, path)
    print(q.ad9082[0].read_info())
    print(q.ad9082[1].read_info())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('addr', help='IP address of ctrl-port of QuBE')
    parser.add_argument('--path', help='PATH of adi_api_mod', default='./adi_api_mod')
    args = parser.parse_args()

    main(args.addr, args.path)

