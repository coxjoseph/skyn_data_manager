import sys
import os
from sdm_user_interface import SkynDataManagerApp


def setup_base_path() -> str:
    if getattr(sys, 'frozen', False):
        # If bundled by PyInstaller, use sys._MEIPASS as the base path
        return getattr(sys,'._MEIPASS', os.getcwd())
    else:
        # If running normally, use the current directory
        return os.path.abspath(os.path.dirname(__file__))


if __name__ == '__main__':
    base_path = setup_base_path()

    # TODO: configure and create logging
    print(f"Running from base path {base_path}")
    sys.path.append(base_path)

    SkynDataManagerApp()
