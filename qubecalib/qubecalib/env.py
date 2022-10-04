import os

get_env = lambda x, y=os.getcwd(): os.environ[x] if x in os.environ else y

PATH_TO_BITFILE: str = get_env('PATH_TO_BITFILE')
PATH_TO_QUBECALIB_PACKAGE: str = get_env('PATH_TO_QUBECALIB_PACKAGE')
