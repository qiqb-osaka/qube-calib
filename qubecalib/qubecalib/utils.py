from pathlib import Path

from .rc import __running_config__ as rc


def get_absolute_path_to_config(config_path: Path) -> Path:
    """basename で指定されたファイルのフルパスを返す. ipynbを実行したディレクトリに
    basename のファイルが存在すればそのフルパスを，そうでなければ dir 内に
    存在するかを確認してそのフルパスを，存在しなければ FileNotFoundError を raise する.

    Args:
        basename (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        Path: _description_
    """
    absolute = (
        Path(config_path.absolute())
        if config_path.exists()
        else rc.path_to_config / Path(config_path.name)
    )
    if not absolute.exists():
        raise FileNotFoundError(f"File {absolute} not found")
    return absolute
