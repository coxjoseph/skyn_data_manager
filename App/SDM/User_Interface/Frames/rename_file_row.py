from tkinter import StringVar, IntVar, Label, Frame, Checkbutton, Entry, Button, messagebox
from SDM.User_Interface.Utils.filename_tools import (matches_filename_convention, extract_additional_filename_text,
                                                     extract_file_extension, is_subid_valid, extract_subid,
                                                     extract_dataset_identifier, stringify_dataset_id)
import os


class RenameFileRow(Frame):
    def __init__(self, frame, parent, filename, i, directory, used_dataset_ids):
        super().__init__(frame)
        self.only_additional_text_is_different = None
        self.parent = parent
        self.filename = filename
        self.pending_filename = filename
        self.used_dataset_ids = used_dataset_ids.copy()
        self.filename_valid = matches_filename_convention(self.filename, self.used_dataset_ids, assess_new=False)
        self.new_filename = self.filename
        self.extension = extract_file_extension(self.filename)
        self.directory = directory
        self.original_dataset_identifier = extract_dataset_identifier(self.filename, self.used_dataset_ids,
                                                                      assess_new=False)
        if self.original_dataset_identifier in self.used_dataset_ids:
            self.used_dataset_ids.remove(self.original_dataset_identifier)

        self.configure(highlightthickness=0)
        self.row_idx = Label(self, text=i + 1)
        self.row_idx.config(font=(None, 12, 'bold'))
        self.row_idx.grid(row=1, column=0, padx=1, pady=1)

        valid_label_text = "Filename Matches Convention:" if self.filename_valid else "Invalid Filename:"
        valid_label_text = "Partially Valid Filename:" if not self.filename_valid and (
            is_subid_valid(extract_subid(self.filename))) else valid_label_text

        self.filenameLabel = Label(self, text=f'{valid_label_text} {filename}', width=50)
        self.filenameLabel.config(font=(None, 12, 'bold'), fg='red' if not self.filename_valid else 'green')
        self.filenameLabel.grid(row=1, column=2, padx=(5, 10), pady=(1, 2), sticky='w')

        self.renameFile = IntVar()
        self.renameFile.set(1 if not self.filename_valid else 0)
        self.renameFileCheckbutton = Checkbutton(self, text="Rename File?", variable=self.renameFile,
                                                 command=self.show_renaming_entries)
        self.filenameLabel.config(font=(None, 11, 'italic'))
        self.renameFileCheckbutton.grid(row=1, column=3, padx=5, pady=(1, 2), sticky='e')

        self.subidLabel = Label(self, text='')
        self.subidLabel.config(font=(None, 8, 'normal'))
        self.subidLabel.grid(row=0, column=4, pady=0, padx=5)

        validate_numeric = self.register(lambda p: len(p) == 0 or (len(p) <= 6 and p.lstrip('0').isdigit()))
        self.subid_text = StringVar()
        self.subid_text.set(extract_subid(self.filename))
        self.subidEntry = Entry(self, width=7, validate='key', validatecommand=(validate_numeric, "%P"),
                                textvariable=self.subid_text)
        self.subidEntry.bind('<KeyRelease>', self.update_filename_text)

        self.datasetIdentifierLabel = Label(self, text='')
        self.datasetIdentifierLabel.config(font=(None, 8, 'normal'))
        self.datasetIdentifierLabel.grid(row=0, column=5, pady=0, padx=5)

        self.datasetIdentifier = StringVar(self)
        self.datasetIdentifier.set(self.original_dataset_identifier)
        self.datasetIdentifierEntry = Entry(self, textvariable=self.datasetIdentifier, width=7)
        self.datasetIdentifierEntry.bind('<KeyRelease>', self.update_filename_text)

        self.additionalTextLabel = Label(self, text='')
        self.additionalTextLabel.config(font=(None, 8, 'normal'))
        self.additionalTextLabel.grid(row=0, column=6, pady=0, padx=5)

        validate_additional_text = self.register(lambda text: len(text) <= 8)
        self.additionalText = StringVar(self)
        self.additionalText.set(extract_additional_filename_text(self.filename))
        self.additionalTextEntry = Entry(self, textvariable=self.additionalText, width=10, validate='key',
                                         validatecommand=(validate_additional_text, "%P"))
        self.additionalTextEntry.bind('<KeyRelease>', self.update_filename_text)

        self.renameFileButton = Button(self, text='Rename File', width=10, command=self.rename_file)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=3)
        self.columnconfigure(3, weight=3)
        self.columnconfigure(4, weight=3)
        self.columnconfigure(5, weight=2)
        self.columnconfigure(6, weight=2)
        self.columnconfigure(7, weight=2)
        self.columnconfigure(8, weight=2)

        self.show_renaming_entries()

    def show_renaming_entries(self):
        if self.renameFile.get() == 1:
            self.subidEntry.grid(row=1, column=4, padx=5, pady=(1, 2))
            self.subidLabel['text'] = 'SubID (3-6 digits)'
            self.datasetIdentifierEntry.grid(row=1, column=5, padx=5, pady=(1, 2))
            self.datasetIdentifierLabel['text'] = 'Dataset ID (1-3 digits)'
            self.additionalTextEntry.grid(row=1, column=6, padx=5, pady=(1, 2))
            self.additionalTextLabel['text'] = 'Additional Text'
            self.renameFileButton.grid(row=1, column=7, padx=5, pady=(1, 2))
        else:
            self.subidEntry.grid_forget()
            self.subidLabel['text'] = ''
            self.datasetIdentifierEntry.grid_forget()
            self.datasetIdentifierLabel['text'] = ''
            self.additionalTextEntry.grid_forget()
            self.additionalTextLabel['text'] = ''
            self.renameFileButton.grid_forget()

    def update_filename_text(self, _event=None):
        subid = self.subidEntry.get()
        dataset_identifier = stringify_dataset_id(self.datasetIdentifierEntry.get())
        additional_text = self.additionalTextEntry.get()
        self.new_filename = f'{subid}_{dataset_identifier}.{self.extension}' if not (
          additional_text) else f'{subid}_{dataset_identifier}_{additional_text}.{self.extension}'

        if self.new_filename != self.pending_filename:
            subid_no_change = (extract_subid(self.pending_filename) == extract_subid(self.new_filename))
            dataset_id_no_change = (extract_dataset_identifier(self.pending_filename) == extract_dataset_identifier(
                self.new_filename))
            self.only_additional_text_is_different = subid_no_change and dataset_id_no_change
            # if self.only_additional_text_is_different and (dataset_identifier in self.used_dataset_ids):
            #   self.used_dataset_ids.remove(dataset_identifier)

            self.pending_filename = self.new_filename
            self.filename_valid = matches_filename_convention(self.new_filename, self.used_dataset_ids, assess_new=True)
            self.filenameLabel['text'] = f'Revised Filename: {self.new_filename}'
            self.filenameLabel['fg'] = 'red' if not self.filename_valid else 'green'
            self.parent.update_used_dataset_ids()
            # remove the present dataset ids from used IDs, else changes to this filename will have conflict
            self.used_dataset_ids.remove(extract_dataset_identifier(self.new_filename))

    def rename_file(self):
        if self.filename_valid:
            current_filename = os.path.join(self.directory, self.filename)
            new_filename = os.path.join(self.directory, self.new_filename)
            try:
                os.rename(current_filename, new_filename)
                self.filenameLabel['text'] = f'Filename Matches Convention: {self.new_filename}'
                self.filename = self.new_filename
                self.pending_filename = self.new_filename
                self.renameFile.set(0)
                self.show_renaming_entries()
            except (FileNotFoundError, PermissionError, OSError, IsADirectoryError, FileExistsError) as e:
                print(f"Error renaming file: {e}")
                messagebox.showerror('Error', e)
        else:
            messagebox.showerror('SDM Guidance',
                                 'Renaming did not occur. \nSubIDs must consist of 3-6 numeric characters.\n'
                                 'Dataset Identifier must consist of 1-3 numeric characters.')
