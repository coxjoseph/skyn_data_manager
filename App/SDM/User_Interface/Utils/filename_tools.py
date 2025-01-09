import os
import re
from typing import Optional

import numpy as np


def is_subid_valid(subid):
    return (2 < len(str(subid))) and (7 > len(str(subid))) and (subid.isnumeric())


def is_dataset_id_valid(episode_id: str, used_ids: list[str | int], assess_new: bool = False) -> bool:
    try:
        if len(episode_id) != 3 or not episode_id.isdigit() or int(episode_id) == 0:
            return False

        if used_ids is not None:
            episode_id_int = int(episode_id)

            if assess_new:
                return episode_id not in used_ids and episode_id_int not in used_ids
            else:
                return used_ids.count(episode_id) + used_ids.count(episode_id_int) <= 1

        return True

    except (ValueError, TypeError) as e:
        print(e)
        return False


def matches_filename_convention(filename, used_ids, assess_new=False):
    subid = extract_subid(filename)
    dataset_id = extract_dataset_identifier(filename)
    return (is_subid_valid(subid)) and (is_dataset_id_valid(dataset_id, used_ids, assess_new))


def stringify_dataset_id(dataset_identifier):
    return "".join(['0' for _ in range(3 - len(str(dataset_identifier)))]) + str(dataset_identifier)


def extract_file_extension(filename):
    filename, file_extension = os.path.splitext(filename)
    return file_extension[1:]


def extract_subid(input_string, validate=True):
    pattern = re.compile(r'^(\d{3,6})')
    match = pattern.findall(input_string)

    if match:
        if validate:
            return match[0] if is_subid_valid(match[0]) else ''
        else:
            return match[0]
    else:
        return ''


def extract_dataset_identifier(filename: str, used_ids: Optional[list[int | str]] = None,
                               validate: bool = True, assess_new: bool = False) -> str:
    try:
        dataset_identifier = filename.split(".")[0].split("_")[1]

        if validate:
            return dataset_identifier if is_dataset_id_valid(dataset_identifier, used_ids, assess_new=assess_new) \
              else ''
        else:
            return str(dataset_identifier)[:3]
    except (IndexError, ValueError, AttributeError) as e:
        print(f"Error extracting dataset identifier: {e}")
        return ''


def extract_additional_filename_text(filename: str) -> str:
    try:
        return filename.split(".")[0].split("_")[2][:3]
    except (IndexError, AttributeError) as e:
        print(f"Error extracting additional filename text: {e}")
        return ''


def assign_dataset_identifier(filename, used_epi_ids=None):
    if used_epi_ids is None:
        used_epi_ids = []
    dataset_identifier = extract_dataset_identifier(filename, used_epi_ids)
    if dataset_identifier:
        return dataset_identifier
    else:
        dataset_identifier_number = 1
        while dataset_identifier_number in [int(i) for i in used_epi_ids]:
            dataset_identifier_number += 1
        dataset_identifier = stringify_dataset_id(dataset_identifier_number)
        return dataset_identifier


def get_used_dataset_identifiers(filenames):
    unique_subids = set([extract_subid(filename) for filename in filenames])
    subid_dataset_ids = {subid: [] for subid in unique_subids}
    for filename in filenames:
        subid = extract_subid(filename)
        used_ids = subid_dataset_ids[subid]
        dataset_identifier = extract_dataset_identifier(filename, used_ids, validate=False)
        subid_dataset_ids[subid].append(dataset_identifier)
    return subid_dataset_ids


def directory_analysis_ready(directory):
    filenames = [file for file in os.listdir(directory) if file.endswith((".xlsx", ".csv"))]
    used_dataset_ids_per_subid = get_used_dataset_identifiers(filenames)
    filenames_valid = all(
        [matches_filename_convention(filename, used_dataset_ids_per_subid[extract_subid(filename)]) for filename in
         filenames])
    subids_consistent = all(
        [len(extract_subid(filename)) == len(extract_subid(filenames[0])) for filename in filenames])
    dataset_identifiers_consistent = all([len(extract_dataset_identifier(filename, used_dataset_ids_per_subid[
        extract_subid(filename)])) == len(
        extract_dataset_identifier(filenames[0], used_dataset_ids_per_subid[extract_subid(filenames[0])])) for filename
                                          in filenames])

    return filenames_valid and subids_consistent and dataset_identifiers_consistent and (len(filenames) > 0)


def get_parsing_indices(path):
    """takes either filename or directory filepath"""
    if os.path.isdir(path):
        files = os.listdir(path)
        filename = os.path.basename(files[0])
    else:
        filename = path

    subid = extract_subid(filename)
    subid_length = len(subid)
    indices = [0, subid_length - 1, subid_length + 1, subid_length + 3]
    return indices


def update_filename_parsing_indices(sdm_interface, filename):
    """takes sdm_interface class and filename, saves indices that indicate where the filenames should be sliced to
    ascertain subid, dataset_id, etc
    """
    indices = get_parsing_indices(filename)
    sdm_interface.parsing_indices = indices
    sdm_interface.subid_i_start = indices[0]
    sdm_interface.subid_i_end = indices[1]
    sdm_interface.dataset_identifier_i_start = indices[2]
    sdm_interface.dataset_identifier_i_end = indices[3]


def create_metadata_from_cohort_folder(cohort_folder):
    files = [file for file in os.listdir(cohort_folder) if (file[-3:] == 'csv') or (file[-4:] == 'xlsx')]
    default_parsing_idx = get_parsing_indices(files[0])

    return {
        'SubID': [int(file[int(default_parsing_idx[0]):int(default_parsing_idx[1]) + 1]) for file in files],
        'Dataset_Identifier':
            [int(file[int(default_parsing_idx[2]):int(default_parsing_idx[3]) + 1]) for file in files],
        'Episode_Identifier': [1 for _ in files],
        'Use_Data': ["Y" for _ in range(len(files))],
        'Condition': [attempt_condition_extraction(file) for file in files],
        'TotalDrks': [np.nan for _ in files],
        'Crop Begin Date': [np.nan for _ in files],
        'Crop Begin Time': [np.nan for _ in files],
        'Crop End Date': [np.nan for _ in files],
        'Crop End Time': [np.nan for _ in files],
        'Time Zone': [np.nan for _ in files],
        'Notes': ["" for _ in files],
    }


def attempt_condition_extraction(filename):
    if 'Alc' in filename:
        return 'Alc'
    elif 'Non' in filename:
        return 'Non'
    else:
        return 'Unk'


def processor_data_ready(processor, min_datasets_required):
    return (len(processor.occasions) > min_datasets_required) and (
            len(processor.features) > min_datasets_required) if hasattr(processor, 'occasions') else False
