from sys import platform


def get_font_size(text_type):
    if platform == "linux" or platform == "linux2":
        return 16 if text_type == 'header' else 12
    elif platform == "darwin":
        return 16 if text_type == 'header' else 12
    elif platform == "win32":
        return 13 if text_type == 'header' else 10
