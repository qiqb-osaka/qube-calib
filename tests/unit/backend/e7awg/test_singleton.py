from qubecalib.backend.e7awg.singleton import Singleton


def test_singleton() -> None:
    class ClassA(Singleton):
        def __new__(cls, info: str):
            super().__new__(cls)
        def _initialize()

    class ClassB(Singleton):
        pass

    obja1 = ClassA("obja1")
    obja2 = ClassA("ojba2")
    objb1 = ClassB()
    objb2 = ClassB()

    assert obja1 is obja2
    assert objb1 is objb2
    assert obja1 is not objb1
    assert obja1 is not objb2
    assert obja2 is not objb1
    assert obja2 is not objb2
    assert obja1.info == "obja2"
    assert obja2.info == "obja2"
