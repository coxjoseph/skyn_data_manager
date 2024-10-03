import tkinter

from SDM.User_Interface.Utils import processor_config
from SDM.User_Interface.Sub_Windows.settings_window import SettingsWindow
from SDM.User_Interface.Frames.data_loading_method_frame import DataLoadingMethodSelection
from SDM.User_Interface.Frames.program_selection_frame import ProgramSelectionFrame
from SDM.User_Interface.Frames.header_menu import HeaderMenu
from SDM.User_Interface.Utils.get_font_size import get_font_size
from tkinter import filedialog, ttk
from tkinter import messagebox
import os
import pandas as pd
import sys


class SkynDataManagerApp(tkinter.Tk):
    def __init__(self):
        super().__init__()
        self.selected_programs = None
        self.filenames = []
        self.exportReportButton = None
        self.metadata_df = None
        self.geometry("750x500")
        self.title("SDM")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.models = []
        self.files_to_merge = {}
        self.skyn_upload_timezone = 'CST'
        self.max_dataset_duration = 24
        self.previous_processor = None

        self.header_menu = HeaderMenu(self)

        style = ttk.Style()
        style.configure("BW-Label", foreground="black", background="white", fontsize=14)
        style.configure("Header", foreground="black", background="white", fontsize=14)

        self.required_inputs_frame = tkinter.Frame(self, highlightbackground="black", highlightthickness=3)
        self.required_inputs_frame.grid(row=0, column=0, padx=20, pady=20, sticky='ew')

        self.header_style = (None, get_font_size('header'), 'bold')
        self.label_style = (None, get_font_size('label'))

        self.mainHeader = tkinter.Label(self.required_inputs_frame, text='Skyn Data Manager', font=self.header_style)
        self.mainHeader.grid(row=0, column=1, padx=5, pady=8)

        self.data_loading_method = None
        self.data_loading_method_frame = DataLoadingMethodSelection(self.required_inputs_frame, self)
        self.data_loading_method_frame.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.program = None
        self.program_selection_frame = ProgramSelectionFrame(self.required_inputs_frame, self)
        self.program_specs = {
            # signal processing only
            'SP': {
                'ProcessSignal': True,
                'Predict': False,
                'Train+Test': False,
            },
            # signal processing and predictions
            'SP_P': {
                'ProcessSignal': True,
                'Predict': True,
                'Train+Test': False
            },
            # signal processing and model training
            'SP_ML': {
                'ProcessSignal': True,
                'Predict': False,
                'Train+Test': True
            },
            # predictions only
            'P': {
                'ProcessSignal': False,
                'Predict': True,
                'Train+Test': False
            },
            # model training only
            'ML': {
                'ProcessSignal': False,
                'Predict': False,
                'Train+Test': True
            }
        }

        self.data_loading_frame = tkinter.Frame(self.required_inputs_frame)

        self.x_separator_mid = ttk.Separator(self.required_inputs_frame, orient='horizontal')

        # cohort name
        self.cohortNameLabelText = '1. Cohort Name:'
        self.cohortNameLabel = tkinter.Label(self.data_loading_frame, text=self.cohortNameLabelText)
        self.validate_cohort_name_length = self.register(lambda p: len(p) <= 13)
        self.cohortNameEntry = tkinter.Entry(self.data_loading_frame, width=20, validate="key",
                                             validatecommand=(self.validate_cohort_name_length, "%P"))

        # Select data folder
        self.selected_data = ''
        self.selectDataLabelText = '2. Select Data:'
        self.selectDataLabel = tkinter.Label(self.data_loading_frame, text=self.selectDataLabelText)
        self.selectDataButton = tkinter.Button(self.data_loading_frame, width=27, text='Select Data',
                                               command=self.select_skyn_data)
        self.selectDataButton.config(font=(None, 8), height=1)

        # select metadata
        self.metadata = ''
        self.metadataLabelText = '3. Select Metadata:'
        self.metadataLabel = tkinter.Label(self.data_loading_frame, text=self.metadataLabelText)
        self.metadataFolderButton = tkinter.Button(self.data_loading_frame, width=20, text='Select Metadata (.xlsx)',
                                                   command=self.open_metadata, state='disabled')
        self.metadataFolderButton.config(font=(None, 8), height=1)

        self.runProgramButton = tkinter.Button(self, text='Submit', fg='blue', command=self.open_settings)
        self.cohortNameEntry.bind("<KeyRelease>", self.toggle_run_button)
        self.cohortNameEntry.bind("<KeyRelease>", self.toggle_metadata_button)

        # LOAD FILE DIRECTLY

        self.update_idletasks()
        tkinter.mainloop()

    def update_data_loading_method(self, selection):
        # possible selections are Processor, Folder, Single, or Test
        self.program = None
        self.data_loading_method = selection
        self.program_selection_frame.grid_forget()

        self.program_selection_frame = ProgramSelectionFrame(self.required_inputs_frame, self)
        self.program_selection_frame.grid(row=2, column=1, padx=5, pady=(0, 15), sticky='w')
        self.x_separator_mid.grid(row=3, column=1, columnspan=2, sticky='ew')

        self.unload_data()

        if self.data_loading_method == 'Test':
            self.selected_data = os.path.abspath('Inputs/Skyn_Data/TestData/') + '/'
            self.filenames = [file for file in os.listdir(self.selected_data)]
            self.metadata = 'Inputs/Metadata/Cohort Metadata TEST.xlsx'
            self.metadata_df = pd.read_excel(self.metadata)

        self.refresh_user_interface()

    def update_program(self, selection):
        self.program = selection
        self.selected_programs = self.program_specs[self.program]

        self.refresh_user_interface()

    def refresh_user_interface(self):
        if self.data_loading_method in ['Single', 'Folder', 'Processor']:
            self.data_loading_frame.grid_forget()
            self.selectDataLabel.grid_forget()
            self.selectDataButton.grid_forget()
            self.data_loading_frame.grid(row=4, column=1, padx=5, pady=0)
            self.selectDataLabel.grid(row=0, column=1, padx=(0, 20), pady=(2, 0))
            self.selectDataButton.grid(row=1, column=1, padx=(0, 20), pady=(0, 5))
        else:
            self.data_loading_frame.grid_forget()
            self.selectDataLabel.grid_forget()
            self.selectDataButton.grid_forget()

        if self.data_loading_method == 'Single' or self.data_loading_method == 'Folder':
            self.cohortNameLabel.grid(row=0, column=0, padx=(0, 20), pady=(2, 0))
            self.cohortNameEntry.grid(row=1, column=0, padx=(0, 20), pady=(0, 5))
            self.metadataLabel.grid(row=0, column=2, padx=(20, 0), pady=(2, 0))
            self.metadataFolderButton.grid(row=1, column=2, padx=(20, 0), pady=(0, 5))
        else:
            self.metadataLabel.grid_forget()
            self.metadataFolderButton.grid_forget()
            self.cohortNameLabel.grid_forget()
            self.cohortNameEntry.grid_forget()

        if self.data_loading_method == 'Single':
            self.selectDataButton['text'] = 'Select Dataset (.xlsx or .csv)'
            self.selectDataButton.config(width=27)
        elif self.data_loading_method == 'Processor':
            self.selectDataLabel['text'] = 'Select SDM Processor'
            self.selectDataButton['text'] = 'Select File (.sdm)'
            self.selectDataButton.grid(row=1, column=1, padx=(0, 20), pady=(0, 5))
            self.selectDataButton.config(width=24)
        else:
            self.selectDataButton['text'] = 'Select Folder (.xlsx and/or .csv)'
            self.selectDataButton.config(width=32)

        self.toggle_export_report_button()
        self.toggle_run_button()

    def toggle_export_report_button(self):
        if self.previous_processor:
            self.exportReportButton = tkinter.Button(self, text='Export SDM Report', command=self.export_report)
            self.exportReportButton.grid(row=34, column=0, pady=(30, 15), padx=(100, 0))
        else:
            try:
                self.exportReportButton.grid_forget()
            except (AttributeError, tkinter.TclError):
                pass

    def export_report(self):
        if self.previous_processor:
            self.previous_processor.export_SDM_report()
        else:
            messagebox.showerror('SDM Error', 'No processor loaded.')

    def toggle_run_button(self, _event=None):
        if self.program:
            if self.data_loading_method == 'Test':
                self.runProgramButton.grid(row=35, column=0, pady=(30, 15), padx=(100, 0), sticky='w')
            elif (self.data_loading_method == 'Folder' or self.data_loading_method == 'Single') and (
                    self.selected_data and self.cohortNameEntry.get()):
                self.runProgramButton.grid(row=35, column=0, pady=(30, 15), padx=(100, 0), sticky='w')
            elif (self.data_loading_method == 'Processor') and self.previous_processor:
                self.runProgramButton.grid(row=35, column=0, pady=(30, 15), padx=(100, 0), sticky='w')
            else:
                self.runProgramButton.grid_forget()
        else:
            self.runProgramButton.grid_forget()

    def unload_data(self):
        self.selectDataLabel['text'] = '2. Select Data'
        self.selectDataLabel.config(fg='black')
        self.metadataLabel['text'] = '3. Select Metadata'
        self.metadataLabel.config(fg='black')
        self.selectDataButton['text'] = self.selectDataLabelText
        self.selected_data = ''
        self.filenames = []
        self.previous_processor = None
        self.metadata = ''
        self.metadata_df = pd.DataFrame()

    def select_skyn_data(self):
        if self.data_loading_method == 'Single':
            self.selected_data = processor_config.load_skyn_dataset(self, self.selectDataLabel)
            self.toggle_metadata_button()
        elif self.data_loading_method == 'Processor':
            self.selected_data = processor_config.load_processor(self, self.selectDataLabel)
        else:
            self.selected_data = processor_config.load_skyn_directory(self, self.selectDataLabel)
            self.toggle_metadata_button()

        if not self.selected_data:
            self.selectDataLabel[
                'text'] = '2. Select Data' if not self.data_loading_method == 'Processor' else 'Select SDM Processor'
            self.selectDataLabel.config(fg='black')

        self.toggle_run_button()

    def update_models(self, model):
        self.models.append(model)

    def reset_models(self):
        self.models = []

    def update_files_to_merge(self, files_to_merge):
        self.files_to_merge = files_to_merge

    def toggle_metadata_button(self, _event=None):
        if self.cohortNameEntry.get() and self.selected_data and (
                self.data_loading_method == 'Single' or self.data_loading_method == 'Folder'):
            self.metadataFolderButton.config(state="normal")
        else:
            self.metadataFolderButton.config(state="disabled")

    def open_metadata(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('Metadata Excel file', '*.xlsx')])
        if file:
            metadata_file = os.path.abspath(file.name)
            metadata_df = pd.read_excel(metadata_file)

            if processor_config.verify_metadata(self.filenames, metadata_df):
                self.metadataLabel['text'] = f'Metadata: {str(file.name.split("/")[-1])}'
                self.metadataLabel.config(fg='green')
                self.metadata = metadata_file
                self.metadata_df = pd.read_excel(metadata_file)
            else:
                self.metadataLabel['text'] = '3. Select Metadata (none loaded)'
                self.metadataLabel.config(fg='black')
                self.metadata = ''
                self.metadata_df = pd.DataFrame()

    def open_settings(self):
        settings = SettingsWindow(self)
        settings.grab_set()
        self.wait_window(settings)

    def on_closing(self):
        self.destroy()
        sys.exit(0)
