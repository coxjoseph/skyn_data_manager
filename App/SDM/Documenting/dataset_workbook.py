import pandas as pd


# TODO: This seems like it is just creating a hardcoded excel file - find usages?
def export_skyn_workbook(dataset, unprocessed_dataset, stats, subid, condition, dataset_identifier, predictions,
                         save_path, simple_plot_paths, complex_plot_paths):
    variable_key = pd.DataFrame({'Variable/Descriptor': [
        'datetime / device timestamp', 'Time', 'time_elapsed_hours', 'potential_artifact', 'TAC', 'Cleaned', 'Greedy',
        'Smooth_XYZ', 'TAC_change', 'Motion', 'Temperature C', 'device_id', 'FOR MORE INFORMATION:'],
        'Explanation': [
            'Timestamp originally produced by biosensor.',
            'Timestamp converted to AM/PM time format.',
            'The amount of time passed since initial biosensor reading.',
            'A Binary variable that indicates whether the data point was detected as an '
            'outlier by the greedy algorithm. A 1 indicates that the data point may be an '
            'artifact of device error.',
            'Data produced by Skyn biosensor. Units are: ug/L(air)',
            'Data cleaned by the standard cleaning method. Abbreviation: C.',
            'Data cleaned by the greedy cleaning method. Abbreviation: CG.',
            'Data smoothed with Savitzky-Golay filter at a window of size XYZ.',
            'The amount of TAC change from the current time point to the previous time point.',
            'A measurement of device motion. Produced by the Skyn.',
            'A measurement of temperature in Celsius. Produced by the Skyn',
            'Device ID',
            # TODO: Find the actual word document
            'See Word document saved here (path removed)']})

    # TODO: paths
    writer = pd.ExcelWriter(f'{save_path}/skyn_{subid}_{condition}{dataset_identifier}.xlsx', engine='xlsxwriter')

    dataset.to_excel(writer, sheet_name='Data', index=False)
    unprocessed_dataset.to_excel(writer, sheet_name='Unprocessed', index=False)

    features = pd.DataFrame([stats], index=[0])
    features.to_excel(writer, sheet_name='Features')
    variable_key.to_excel(writer, sheet_name='Variable Key', index=False)

    if len(predictions) > 0:
        predictions.to_excel(writer, sheet_name='Predictions', index=False)

    workbook = writer.book
    worksheet = workbook.add_worksheet('Signal Processing Visuals')
    worksheet.set_default_row(20)
    row_index = 1
    for plot_path in complex_plot_paths:
        image_start_cell = 'B' + str(row_index)
        worksheet.insert_image(image_start_cell, plot_path)
        row_index += 30

    workbook = writer.book
    worksheet = workbook.add_worksheet('Motion & Temp Graphs')
    row_index = 1
    for plot_path in simple_plot_paths:
        image_start_cell = 'B' + str(row_index)
        worksheet.insert_image(image_start_cell, plot_path)
        row_index += 20

    writer.close()
