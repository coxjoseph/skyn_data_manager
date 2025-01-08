from SDM.Skyn_Processors.skyn_cohort import SkynCohort
from SDM.Skyn_Processors.skyn_dataset import skynDataset
from SDM.User_Interface.Utils.filename_tools import *
from SDM.User_Interface.Utils.filename_tools import extract_subid
from SDM.User_Interface.Sub_Windows.rename_files_window import RenameFilesWindow
from SDM.User_Interface.Sub_Windows.filenames_confirmation_window import FilenamesConfirmationWindow
from tkinter import filedialog, ttk, simpledialog
from tkinter import messagebox
from datetime import date, datetime
import pickle
import traceback

def verify_metadata(filenames, metadata):

  required_columns = ['SubID', 'Dataset_Identifier', 'Episode_Identifier', 'Use_Data', 'Condition', 'TotalDrks']
  if not all(col in metadata.columns for col in required_columns):
      messagebox.showerror('SDM Error', f'Metadata file must have columns: {", ".join(required_columns)}.')
      return False
  
  for file in filenames:
    subid = extract_subid(file)
    dataset_id = extract_dataset_identifier(file)
    matching_metadata = metadata[(metadata['SubID']==int(subid)) & (metadata['Dataset_Identifier']==int(dataset_id))]
    episode_ids = matching_metadata['Episode_Identifier'].tolist()

    if len(matching_metadata) == 0:
      messagebox.showerror('SDM Error', f'No matching metadata for SubID: {subid}, Dataset ID: {dataset_id} ')
      return False
    elif len(episode_ids) != len(set(episode_ids)):
      messagebox.showerror('SDM Error', f'Duplicate episode IDs ({", ".join(episode_ids)}) for SubID: {subid}, Dataset ID: {dataset_id} ')
      return False
    
  return True

def verify_filename(sdm_interface, filename, directory):
    
    sdm_interface.filenames = [file for file in os.listdir(directory)]
    used_dataset_ids_per_subid = get_used_dataset_identifiers(sdm_interface.filenames)
    subid = extract_subid(filename)

    if subid in list(used_dataset_ids_per_subid.keys()):
      dataset_identifier = extract_dataset_identifier(filename, used_dataset_ids_per_subid[subid], assess_new=False)
      filename_valid = matches_filename_convention(filename, used_dataset_ids_per_subid[subid], assess_new=False)
    else:
      filename_valid = False

    confirmed_by_user = False
    if filename_valid:
      confirmed_by_user = messagebox.askyesno('SDM', f'Is data set info correct?\nSubID = {subid}\nDataset ID = {dataset_identifier}')
    
    if filename_valid and confirmed_by_user:
      return True
    else:
      renameFiles = RenameFilesWindow(sdm_interface, directory, sdm_interface.filenames)
      renameFiles.grab_set()
      sdm_interface.wait_window(renameFiles)
      return False
  
def load_skyn_dataset(sdm_interface, label):
  filename_valid = False
  while not filename_valid:
    skyn_dataset_file = filedialog.askopenfile(mode='r', filetypes=[('Skyn dataset','*.xlsx *.csv *.CSV')])

    #if the select file dialog box is canceled, the loop will break and command will end 
    if not skyn_dataset_file:
      return ''

    #otherwise, continue to try to get a valid filename
    else:
      full_filename = os.path.abspath(skyn_dataset_file.name)
      filename=os.path.split(full_filename)[-1]
      directory = os.path.dirname(full_filename)
      skyn_dataset_file.close()
      filename_valid = verify_filename(sdm_interface, filename, directory)
      if filename_valid:
        label['text'] = f'Dataset selected: {filename}'
        label.config(fg='green')
        update_filename_parsing_indices(sdm_interface, filename)
        return full_filename
      else:
        label['text'] = f'Dataset selected:'
        label.config(fg='red')

def verify_directory(sdm_interface, directory):
  sdm_interface.filenames = [file for file in os.listdir(directory)]

  directory_valid = directory_analysis_ready(directory)
  confirmed_by_user = False

  if directory_valid:
    update_filename_parsing_indices(sdm_interface, sdm_interface.filenames[0])
    confirmation_window = FilenamesConfirmationWindow(sdm_interface, sdm_interface.parsing_indices, directory)
    confirmation_window.grab_set()
    sdm_interface.wait_window(confirmation_window)
    confirmed_by_user = confirmation_window.confirmed

  if directory_valid and confirmed_by_user:
    update_filename_parsing_indices(sdm_interface, sdm_interface.filenames[0])
    return directory
  else:
    renameFiles = RenameFilesWindow(sdm_interface, directory, sdm_interface.filenames)
    renameFiles.grab_set()
    sdm_interface.wait_window(renameFiles)
    return ''
  
def load_skyn_directory(sdm_interface, label):
  directory_verified = False
  while not directory_verified:
    cohort_data_folder = filedialog.askdirectory()

    #if the select file dialog box is canceled, the loop will break and command will end 
    if not cohort_data_folder:
      return ''

    #otherwise, continue to try to get a valid directory
    else:
      cohort_data_folder = cohort_data_folder + '/'
      directory_verified = verify_directory(sdm_interface, cohort_data_folder)
      if directory_verified:
        label['text'] = f'Cohort Data:  {cohort_data_folder.split("/")[-2]}'
        label.config(fg='green')
        directory_verified = True
        return cohort_data_folder
      else:
        label['text'] = f'Cohort Data:'
        label.config(fg='red')
  
def load_processor(sdm_interface, label):
  processor_verified = False
  while not processor_verified:
    processor_file = filedialog.askopenfile(mode='r', filetypes=[('SDM File', '*.sdm'),('SDM File', '*.pickle')])

    if not processor_file:
      return ''

    else:
      min_required_datasets = 1 if sdm_interface.selected_programs['Predict'] else 10
      skyn_processor = verify_processor(processor_file, min_required_datasets)
      if skyn_processor:
        sdm_interface.previous_processor = skyn_processor
        label['text'] = f'Skyn processor: {str(processor_file.name.split("/")[-1])}'
        label.config(fg = 'green')
        # if sdm_interface.program_selection_frame.grid_info():
        #   sdm_interface.program_selection_frame.update_processor_programs(model_avail)
        return processor_file
      else:
        sdm_interface.previous_processor=skyn_processor
        label['text'] = f'Failed to load: {str(processor_file.name.split("/")[-1])}'
        label.config(fg = 'red')

def verify_processor(processor_file, min_datasets_required):
  try:
    pickle_in = open(processor_file.name, "rb") 
    skyn_processor = pickle.load(pickle_in)
    pickle_in.close()
    
    data_avail = processor_data_ready(skyn_processor, min_datasets_required)
    if not data_avail:
      messagebox.showerror('SDM Error', 'This file does not have processed Skyn data.')
      return None
    
    conditions_valid = all([occasion.condition in ['Alc', 'Non', 'Unk'] for occasion in skyn_processor.occasions])
    if not conditions_valid:
      messagebox.showerror('SDM Error', 'Datasets are not assigned valid "Condition" labels in Metadata. Labels must be: Alc, Non, or Unk.')
      return None
    
    return skyn_processor
    
  except Exception:
    print(traceback.format_exc())
    messagebox.showerror('SDM Error', traceback.format_exc())
    return None
  
def create_processor(settings_window, sdm_interface):
  
  data_format = sdm_interface.data_loading_method
  cohort_name = sdm_interface.cohortNameEntry.get() if data_format not in ['Processor','Test'] else sdm_interface.previous_processor.cohort_name if data_format == 'Processor' else 'Test'

  data_out = f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}/Processed_Datasets'
  graphs_out = f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}/Plots'
  analyses_out = f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}/Model_Performance'

  if data_format != 'Processor':

    create_results_directories(cohort_name)

    if data_format == 'Test':
      sdm_processor = SkynCohort(
        os.path.abspath('Inputs/Skyn_Data/TestData/') + '/',
        metadata_path = 'Inputs/Metadata/Cohort Metadata TEST.xlsx',
        cohort_name = 'Test',
        merge_variables = {},
        disable_crop_start = False,
        disable_crop_end = False,
        data_out_folder = data_out,
        graphs_out_folder = graphs_out,
        analyses_out_folder = analyses_out,
        max_dataset_duration = 24,
        skyn_upload_timezone = 'CST',
      )

    elif data_format == 'Folder':
      sdm_processor = SkynCohort(
        sdm_interface.selected_data,
        metadata_path = sdm_interface.metadata,
        cohort_name = cohort_name,
        merge_variables = {},
        disable_crop_start = False,
        disable_crop_end = False,
        data_out_folder = data_out,
        graphs_out_folder=graphs_out,
        analyses_out_folder=analyses_out,
        max_dataset_duration=24,
        skyn_upload_timezone='CST'
      )
    
    elif data_format == 'Single':
      subid = extract_subid(os.path.basename(sdm_interface.selected_data))
      dataset_identifier = extract_dataset_identifier(os.path.basename(sdm_interface.selected_data))
      
      if sdm_interface.metadata != '':
        matching_metadata = sdm_interface.metadata_df.loc[((sdm_interface.metadata_df['SubID']==str(subid)) | (sdm_interface.metadata_df['SubID']==int(subid))) & ((sdm_interface.metadata_df['Dataset_Identifier']==str(dataset_identifier)) | (sdm_interface.metadata_df['Dataset_Identifier'] == int(dataset_identifier)))].reset_index(drop=True)
        matching_episode_ids = [int(val) for val in matching_metadata['Episode_Identifier'].tolist() if val]
      else:
        matching_metadata = []
        matching_episode_ids = []

      #Find episode identifier in metadata; if multiple, user must choose
      episode_identifier = None
      if len(matching_metadata) > 1:
        episode_identifier_registered = False
        while not episode_identifier_registered:
          episode_identifier = simpledialog.askinteger("Metadata indicates that multiple episodes were identified within the chosen dataset", f'Please enter one of the available Episode IDs: {" ,".join([str(id) for id in matching_episode_ids])}')
          episode_identifier_registered = episode_identifier in matching_episode_ids
          if episode_identifier_registered:
            break
      elif len(matching_metadata) == 1:
        episode_identifier = str(matching_episode_ids[0])
      else:
        episode_identifier = 1

      if not episode_identifier:
        messagebox.showerror('SDM Error', f'Episode Identifier could not be loaded from metadata. Please check metadata: {sdm_interface.metadata}')
      else:
        print(sdm_interface.selected_data)
        sdm_processor = skynDataset(
          sdm_interface.selected_data,
          data_out,
          graphs_out,
          int(subid),
          int(dataset_identifier),
          'e' + str(episode_identifier),
          False,
          False,
          'CST',
          24,
          metadata_path = sdm_interface.metadata,
        )

  else:
    #USING PRIOR PROCESSOR
    sdm_processor = sdm_interface.previous_processor
    
    if sdm_interface.selected_programs['Train+Test']:
      existing_model_names = [model.model_name for model in sdm_interface.previous_processor.models]
      pending_model_names = settings_window.TrainSettingsFrame.selected_models
      for name in [name for name in existing_model_names if name in pending_model_names]:
        remove_model_from_queue = messagebox.askyesno('SDM', f'Model [{name}] has already been trained; would you like to retrain and overwrite it?\nTo overwrite, click YES. Otherwise, click NO.')
        if remove_model_from_queue:
          settings_window.TrainSettingsFrame.unclick_model(name)
        else:
          model_to_overwite = [model for model in sdm_processor.models if model.model_name == name][0]
          sdm_processor.models.remove(model_to_overwite)

  #modify according to user customizations
  settings = load_settings(settings_window)
  for key, val in settings.items():
    if hasattr(sdm_processor, key) and val:
      setattr(sdm_processor, key, val)
    
  return sdm_processor

def create_results_directories(cohort_name):
  if not os.path.exists('Results'):
    os.mkdir('Results')
  if not os.path.exists(f'Results/{cohort_name}'):
    os.mkdir(f'Results/{cohort_name}')
    print('created', f'Results/{cohort_name}')
  if not os.path.exists(f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}'):
    os.mkdir(f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}')
    print('created', f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}')

  if not os.path.exists(f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}/Model_Performance'):
    os.mkdir(f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}/Model_Performance')
    print('created', f'Results/{cohort_name}/{date.today().strftime("%m.%d.%Y")}/Model_Performance')

def load_settings(settings_window):
  if settings_window.sdm_interface.data_loading_method != 'Processor':
    return {
      'major_cleaning_threshold': 
      settings_window.ProcessSignalSettingsFrame.majorThreshold.get(),
      'minor_cleaning_threshold': 
        settings_window.ProcessSignalSettingsFrame.minorThreshold.get(),
      'smoothing_window': 
        settings_window.ProcessSignalSettingsFrame.smoothingWindow.get(),
      'device_removal_detection_method': 
        settings_window.ProcessSignalSettingsFrame.deviceRemovalDetection.get(),
      'max_duration': 
        settings_window.CropSettingsFrame.maxDatasetDuration.get(),
      'skyn_upload_timezone': 
        settings_window.CropSettingsFrame.timezoneUpload.get(),
      'disable_crop_start': 
        settings_window.CropSettingsFrame.disableCropStart.get() if settings_window.CropSettingsFrame.crop_start_timestamps_available else 1,
      'disable_crop_end': 
        settings_window.CropSettingsFrame.disableCropEnd.get() if settings_window.CropSettingsFrame.crop_end_timestamps_available else 1,
      'min_duration_active': 
        settings_window.QualityControlFrame.min_duration_active.get() / 60,
      'max_percentage_inactive': 
        settings_window.QualityControlFrame.max_percentage_inactive.get(),
      'max_percentage_imputed': 
        settings_window.QualityControlFrame.max_percentage_imputed.get(),
      'training_features': 
        [key for key, val in settings_window.TrainSettingsFrame.predictor_checkboxes.items() if val.get()] if settings_window.sdm_interface.selected_programs['Train+Test'] else settings_window.TrainSettingsFrame.predictor_options,
      }
  else:
    return {
      'models': 
        settings_window.sdm_interface.previous_processor.models,
      'training_features': 
        [key for key, val in settings_window.TrainSettingsFrame.predictor_checkboxes.items() if val.get()] if settings_window.sdm_interface.selected_programs['Train+Test'] else settings_window.sdm_interface.previous_processor.training_features
    }
          
