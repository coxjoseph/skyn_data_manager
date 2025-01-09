from SDM.User_Interface.Utils.filename_tools import (create_metadata_from_cohort_folder, extract_subid,
                                                     extract_dataset_identifier, directory_analysis_ready)
from SDM.User_Interface.Sub_Windows.rename_files_window import RenameFilesWindow
from tkinter import filedialog, Toplevel, Frame, Button, Label, Entry, Listbox, Variable, MULTIPLE
from tkinter import messagebox
import pandas as pd
import os


class CreateMetadataWindow(Toplevel):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.filenames = None
        self.episode_labels = None
        self.selected_text_list = None
        self.folder_metadata = None
        self.main_window = main_window
        self.geometry("400x300")

        # FRAME
        self.metadata_frame = Frame(self, width=600, height=800, highlightbackground="black", highlightthickness=2)
        self.metadata_frame.grid(row=0, column=1, padx=7, pady=7)

        # Contained within frame...
        # Select Cohort Folder
        self.select_cohort_folder = Button(self.metadata_frame, text="Select Folder with Skyn Data", width=24,
                                           command=self.load_cohort_folder)
        self.select_cohort_folder.grid(row=0, column=1, pady=5, padx=5)

        # Select Cohort Name
        self.cohortNameLabelText = 'Cohort Name'
        self.cohortNameLabel = Label(self.metadata_frame, text=self.cohortNameLabelText)
        self.validate_cohort_name_length = self.metadata_frame.register(lambda p: len(p) <= 10)
        self.cohortNameEntry = Entry(self.metadata_frame, width=20, validate="key",
                                     validatecommand=(self.validate_cohort_name_length, "%P"))
        self.cohortNameLabel.grid(row=1, column=1, padx=5, pady=(10, 2))
        self.cohortNameEntry.grid(row=2, column=1, padx=5, pady=(2, 10))

        # To load after cohort folder and cohort name are selected...
        # InstructionalText
        self.instructions = (
            'Confirm that SubIDs, Conditions, and Dataset IDs have been properly read for each episode '
            'listed below.\nNext, select each episode that you want to exclude from analyses. \nTo '
            'create metadata file, click the button below.')
        self.instructionsLabel = Label(self.metadata_frame, text=self.instructions, anchor='w', justify='left')

        # A row for each file
        self.exclude_subids_listbox = Listbox(self.metadata_frame, selectmode=MULTIPLE, height=18, width=60)

        self.filenames_verified = False

        # When files are verified, show button to create metadata template
        self.createMetadataButton = Button(self.metadata_frame, text='Create metadata template (.xlsx)', width=25,
                                           command=self.create_metadata, fg='blue')

    def verify_directory(self, directory):
        self.filenames_verified = directory_analysis_ready(directory)

        if self.filenames_verified:
            self.geometry("580x490")
            self.folder_metadata = create_metadata_from_cohort_folder(directory)
        else:
            rename_files = RenameFilesWindow(self, directory, self.filenames)
            rename_files.grab_set()

    def load_cohort_folder(self):
        cohort_data_folder = filedialog.askdirectory()
        self.lift()
        if cohort_data_folder:
            self.filenames = os.listdir(cohort_data_folder)
            self.verify_directory(cohort_data_folder + '/')

        print('verified?', self.filenames_verified)
        if self.filenames_verified:
            self.filenames = os.listdir(cohort_data_folder)
            subids = [extract_subid(filename) for filename in self.filenames]
            dataset_ids = [extract_dataset_identifier(filename) for filename in self.filenames]
            print(self.filenames)
            print(len(subids), 'subid list length')
            print(len(dataset_ids), 'id list length')
            print(dataset_ids)

            self.episode_labels = Variable(
                value=[f'Subject: {subids[i]} | ID: {dataset_ids[i]}' for i in range(0, len(subids))])
            self.instructionsLabel.grid(row=3, column=1, padx=5, pady=5)
            self.exclude_subids_listbox.configure(listvariable=self.episode_labels)
            self.exclude_subids_listbox.grid(row=4, column=1, padx=5, pady=5)
            self.createMetadataButton.grid(row=5, column=1, padx=5, pady=7)

    def create_metadata(self):
        try:
            self.selected_text_list = [i for i in self.exclude_subids_listbox.curselection()]
            self.folder_metadata['Use_Data'] = [
                self.folder_metadata['Use_Data'][i] if i not in self.selected_text_list else 'N' for i in
                range(0, len(self.folder_metadata['Use_Data']))]
            meta_df = pd.DataFrame(self.folder_metadata)
            meta_df.to_excel(f'Inputs/Metadata/{self.cohortNameEntry.get()}_Metadata.xlsx', index=False)
            messagebox.showinfo('Success', f'File created: '
                                           f'Inputs/Metadata/{self.cohortNameEntry.get()}_Metadata.xlsx')
            self.destroy()
        except (AttributeError, KeyError, IndexError, TypeError,
                PermissionError, RuntimeError, FileNotFoundError,
                OSError, ValueError, RuntimeError) as e:
            print(f"Error: {e}")
            messagebox.showerror(f'Error creating metadata file: {e}')
