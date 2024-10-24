from pathlib import Path
from typing import Optional
import pandas as pd
import string
import random


def rename(filepath: Path, new_path: Path):
    try:
        filepath.rename(new_path)
        print(f'Renamed: {filepath} -> {new_path}')
    except FileNotFoundError:
        print(f'Error: {filepath} not found.')
    except PermissionError:
        print(f'Error: Permission denied when renaming {filepath}.')
    except OSError as e:
        print(f'Error renaming {filepath}: {e}')


# TODO: pathy signatures, and this seems like a useless function, and what does # have to do with anything?
def modify_filenames(directory_path: str, insert_index: int, insert_character: str) -> None:
    directory = Path(directory_path)

    if not directory.is_dir():
        raise NotADirectoryError(f"Error: {directory_path} is not a directory")

    for filepath in directory.iterdir():
        if filepath.is_file():
            filename, extension = filepath.name, filepath.suffix

            if (extension == '.csv' or extension == '.xlsx') and filename[insert_index] != '#':
                new_name = filename[:insert_index] + insert_character + filename[insert_index:]
                new_path = filepath.with_name(new_name)
                rename(filepath, new_path)


def replace_substring_in_filenames(directory_path: str, substring_find: str, substring_replace: str) -> None:
    directory = Path(directory_path)
    if not directory.is_dir():
        raise NotADirectoryError(f"Error: {directory_path} is not a directory")

    for filepath in directory.iterdir():
        if filepath.is_file() and filepath.suffix in ['.csv', '.xlsx']:
            filename = filepath.name
            if substring_find in filename:
                new_filename = filename.replace(substring_find, substring_replace)
                new_filepath = filepath.with_name(new_filename)
                rename(filepath, new_filepath)


def modify_filenames_with_randomization(directory_path: str, randomization_filepath: str, starting_indices_subid: list,
                                        subid_length: int, strings_to_replace=None) -> None:
    if strings_to_replace is None:
        strings_to_replace = [' A.', ' B.']

    # TODO: are we sure filepath is an excel file?
    randomization = pd.read_excel(randomization_filepath)
    directory = Path(directory_path)

    if not directory.is_dir():
        raise NotADirectoryError(f"Error: {directory_path} is not a directory")

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix in [".xlsx", ".csv"]:
            filename = file_path.name
            print('filename:', filename)

            subid = extract_subid_from_filename(filename, starting_indices_subid, subid_length)
            if subid is None:
                print(f"Subid not found in {filename}. Skipping file.")
                continue

            rando_code = fetch_randomization_code(subid, randomization)
            if rando_code is None:
                print(f"Randomization code for subid {subid} not found. Skipping file.")
                continue

            session_order = [' non.' if rando_code == 1 else ' alc.',
                             ' non.' if rando_code == 2 else ' alc.']

            new_filename = replace_strings_in_filename(filename, strings_to_replace, session_order)
            if new_filename:
                new_file_path = file_path.with_name(new_filename)
                print(f'Renamed: {file_path} -> {new_file_path}')
                file_path.rename(new_file_path)


def generate_random_id(length: int = 6) -> str:
    if length < 1:
        raise ValueError('Length must be greater than or equal to 1.')

    first_digit = random.choice(string.digits[1:])
    remaining_digits = ''.join(random.choices(string.digits, k=length-1))
    return first_digit + remaining_digits


# TODO: this is coercing string to int for subid which we want to avoid
def extract_subid_from_filename(filename: str, starting_indices: list, subid_length: int) -> Optional[int]:
    for index in starting_indices:
        potential_subid = filename[index: index + subid_length]
        if potential_subid.isnumeric():
            return int(potential_subid)
    return None


def fetch_randomization_code(subid: int, randomization: pd.DataFrame) -> Optional[int]:
    rando_row = randomization.loc[randomization['subid'] == subid, 'rando_code']
    if not rando_row.empty:
        return rando_row.iloc[0]
    return None


def replace_strings_in_filename(filename: str, strings_to_replace: list, session_order: list) -> Optional[str]:
    new_filename = filename
    for i, s in enumerate(strings_to_replace):
        if s in new_filename:
            new_filename = new_filename.replace(s, session_order[i])
    return new_filename if new_filename != filename else None
