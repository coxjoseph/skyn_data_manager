import pickle
import xlsxwriter
from pathlib import Path


# TODO: Go back and make this pathy function signature
def save_to_computer(o, filename, folder, extension='sdm'):
    if extension.startswith('.'):
        extension = extension[1:]
    outfile = Path(folder) / f'{filename}.{extension}'

    with open(outfile, mode='wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(o, f)

    print(f'SAVE SUCCESSFUL: {folder}/{filename}.{extension}')


def load(name, folder):
    try:
        extension = 'sdm'
        pickle_in = open(f'{folder}/{name}.{extension}', "rb")
    except:
        extension = 'pickle'
        pickle_in = open(f'{folder}/{name}.{extension}', "rb")
    object = pickle.load(pickle_in)
    pickle_in.close()
    return object


def load_default_model(name='Alc_vs_Non', type='RF'):
    for extension in ['sdmtm', 'pickle']:
        try:
            if name == 'Alc_vs_Non':
                pickle_in = open(f'App/SDM/Trained_Models/MARS2C4{type}_Alc_vs_Non.{extension}', "rb")
            if name == 'AUD':
                pickle_in = open(f'App/SDM/Trained_Models/MARS2C4{type}_AUD.{extension}', "rb")
            if name == 'Binge':
                pickle_in = open(f'App/SDM/Trained_Models/MARS2C4{type}_Binge.{extension}', "rb")
            if name == 'worn_vs_removed':
                pickle_in = open(f'App/SDM/Trained_Models/worn_vs_removed_{type}.{extension}', "rb")
                type = 'LinReg'
            if name == 'fall_duration':
                pickle_in = open(f'App/SDM/Trained_Models/fall_duration_CLN_LinearReg.{extension}', "rb")
                type = 'LinReg'
            if name == 'fall_rate':
                pickle_in = open(f'App/SDM/Trained_Models/fall_rate_CLN_LinearReg.{extension}', "rb")
                type = 'LinReg'
            if name == 'rise_duration':
                pickle_in = open(f'App/SDM/Trained_Models/rise_duration_CLN_LinearReg.{extension}', "rb")
                type = 'LinReg'
            if name == 'rise_rate':
                pickle_in = open(f'App/SDM/Trained_Models/rise_rate_CLN_LinearReg.{extension}', "rb")
                type = 'LinReg'
        except:
            pass

    object = pickle.load(pickle_in)
    pickle_in.close()

    return object


def get_model_summary_sheet_name(model_name, data_version):
    model_name_new = model_name.split('_')[0][0].upper() + model_name.split('_')[0][1:] + ' ' + \
                     model_name.split('_')[1][0].upper() + model_name.split('_')[1][1:]
    return f'{model_name_new} - {data_version}'


def reorder_tabs(analyses_out_folder, cohort_name):
    workbook = xlsxwriter.Workbook(f'{analyses_out_folder}/skyn_report_{cohort_name}.xlsx')

    sheetlist = workbook.worksheets._name
    sheetlist.insert(1, sheetlist.pop(len(sheetlist) - 1))
    #does this bring a tab from back to front?
    workbook.worksheets_objs.sort(key=lambda x: sheetlist.index(x.name))
    workbook.close()


def merge_using_subid(sdm_results, merge_variables):
    for file, info in merge_variables.items():
        df = info['df']
        data_to_add = df[[info['subid_column']] + info['variables']]
        sdm_results = sdm_results.merge(data_to_add, on=info['subid_column'], how='left')
    return sdm_results
