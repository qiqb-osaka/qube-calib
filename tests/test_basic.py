from dotenv import load_dotenv

load_dotenv()


def test_import_modules() -> None:
    import qubecalib

    qubecalib
    qubecalib.qube
    qubecalib.backendqube
    qubecalib.neopulse
    qubecalib.ui
