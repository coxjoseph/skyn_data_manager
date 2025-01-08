from SDM.User_Interface.Frames.model_selection import ModelSelection
from SDM.User_Interface.Frames.crop_settings import CropSettings
from SDM.User_Interface.Frames.process_signal_settings import ProcessSignalSettings
from SDM.User_Interface.Frames.train_settings import TrainSettings
from SDM.User_Interface.Frames.quality_control_settings import QualityControlSettings
from SDM.User_Interface.Utils.processor_config import create_processor, create_results_directories
from SDM.User_Interface.Frames.header_menu import HeaderMenu
from tkinter import *
from tkinter import filedialog, StringVar, IntVar
from tkinter import ttk
from tkinter.filedialog import askopenfile
from tkinter import messagebox
import traceback

class SettingsWindow(Toplevel):
  def __init__(self, sdm_interface):
    super().__init__(sdm_interface)

    self.geometry("650x750")
    self.title('Options')
    self.grab_set()
    self.lift()
    self.header_menu = HeaderMenu(self)

    self.sdm_interface = sdm_interface

    tab_frame = Frame(self)
    tab_frame.grid(row=1, column=0, padx=20, pady=20)

    tabControl = ttk.Notebook(tab_frame)

    self.ProcessSignalSettingsFrame = ProcessSignalSettings(tabControl, self)
    self.ModelSelectionFrame = ModelSelection(tabControl, self)
    self.TrainSettingsFrame = TrainSettings(tabControl, self)
    self.QualityControlFrame = QualityControlSettings(tabControl, self)
    if self.sdm_interface.data_loading_method != 'Processor':
      self.CropSettingsFrame = CropSettings(tabControl, self)

    if self.sdm_interface.selected_programs['ProcessSignal']:
      tabControl.add(self.CropSettingsFrame, text='Cropping & Timestamps')
      tabControl.add(self.ProcessSignalSettingsFrame, text='Signal Processing')
      tabControl.add(self.QualityControlFrame, text='Quality Control')
    if self.sdm_interface.selected_programs['Predict']:
      tabControl.add(self.ModelSelectionFrame, text='Model Selection')
    if self.sdm_interface.selected_programs['Train+Test']:
      tabControl.add(self.TrainSettingsFrame, text='Model Training')
    tabControl.pack(expand=1, fill="both")

    submitButton = Button(self, text='Launch SDM', command=self.run)
    submitButton.grid(row=2, column=0, padx=10, pady=10)
  
  def run(self):
    try:
      processor = create_processor(self, self.sdm_interface)

      if self.sdm_interface.data_loading_method == 'Single':
        if self.sdm_interface.selected_programs['ProcessSignal']:
          processor.process_with_default_settings(make_plots=True, export=~self.sdm_interface.selected_programs['Predict'])
        if self.sdm_interface.selected_programs['Predict']:
          if processor.valid_occasion:
              processor.make_prediction(self.ModelSelectionFrame.selected_models)
        if processor.valid_occasion:
          processor.export_workbook()
          messagebox.showinfo('SDM', f'SDM complete.')
        else:
          processor.export_workbook()
          messagebox.showerror('SDM', f'Unable to make prediction; dataset is invalid due to {processor.invalid_reason}')
    
      if self.sdm_interface.data_loading_method in ['Folder', 'Test', 'Processor']:
        if self.sdm_interface.selected_programs['ProcessSignal']:
          processor.process_cohort()
        if self.sdm_interface.selected_programs['Predict']:
            processor.make_predictions_using_default_models(self.ModelSelectionFrame.selected_models)
        if self.sdm_interface.selected_programs['Train+Test']:
          processor.train_models_using_episode_features(self.TrainSettingsFrame.selected_models)
        messagebox.showinfo('SDM', f'SDM complete. See Results/{self.sdm_interface.cohortNameEntry.get()}')

    #IF PREVIOUS PROCESSOR
    except Exception:
      messagebox.showerror('SDM Error', f'Failed to load. See error: {traceback.format_exc()}')
      print('SDM Error', f'Failed to load. See error: {traceback.format_exc()}')

    """
    When clean signal = NONE

    """











    
