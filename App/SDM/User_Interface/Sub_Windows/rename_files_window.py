from SDM.User_Interface.Frames.rename_file_row import RenameFileRow
from SDM.User_Interface.Utils.filename_tools import get_used_dataset_identifiers, extract_subid
from tkinter import Toplevel, Label, Frame, Button, Canvas, Scrollbar, messagebox
from datetime import datetime, date
import pandas as pd
import os


class RenameFilesWindow(Toplevel):
    def __init__(self, parent, directory, filenames):
        super().__init__(parent)
        self.title("Rename Files")
        self.geometry("1400x900")
        self.parent = parent
        self.directory = directory
        self.filenames = filenames
        print(filenames)
        self.used_dataset_ids_per_subid = get_used_dataset_identifiers(self.filenames)

        self.columnconfigure(0, weight=5)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(10, weight=1)

        self.header = Label(self, text='Rename Files according to SDM Convention')
        self.header.grid(row=0, column=0, padx=5, pady=(5, 10))
        self.header.config(font=(None, 14, 'bold'))

        self.buttons_frame = Frame(self)
        self.buttons_frame.grid(row=4, column=0)

        self.submit = Button(self.buttons_frame, text='Rename All Files (Irreversible)', width=25,
                             command=self.rename_files)
        self.submit.grid(row=0, column=0, padx=10, pady=2, sticky='w')

        self.close = Button(self.buttons_frame, text='Close', width=7, command=self.destroy)
        self.close.grid(row=0, column=1, padx=10, pady=2)

        self.rows_frame = Frame(self, height=800, width=800, highlightbackground="black", highlightthickness=2)
        self.rows_frame.grid(row=10, column=0, padx=5, pady=5, sticky="nsew")

        self.canvas = Canvas(self.rows_frame)
        self.canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.rows_frame.columnconfigure(0, weight=1)  # Make the column expand with the window
        self.rows_frame.rowconfigure(0, weight=1)
        self.rows_frame.grid_rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1)

        self.scrollbar = Scrollbar(self.rows_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set, highlightthickness=0)

        self.inner_frame = Frame(self.canvas, highlightthickness=0)
        self.inner_frame.grid_rowconfigure(0, weight=1)
        self.inner_frame.grid_columnconfigure(0, weight=1)
        self.canvas.create_window((5, 5), window=self.inner_frame, anchor="nw")

        self.renaming_widgets = {}
        for i, filename in enumerate(self.filenames):
            used_ids = self.used_dataset_ids_per_subid[extract_subid(filename)]
            self.renaming_widgets[filename] = RenameFileRow(self.inner_frame, self, filename, i, self.directory,
                                                            used_ids)
            self.renaming_widgets[filename].grid(row=i, column=0, pady=2, padx=2, sticky='w')

        self.inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def rename_files(self):
        if not all([widget.filename_valid for widget in list(self.renaming_widgets.values())]):
            messagebox.showerror('SDM Guidance',
                                 'Renaming did not occur. \n'
                                 'SubIDs must consist of 3-6 numeric numbers.\n'
                                 'Episode Identifier must consist of 1-3 numeric numbers. Do NOT use leading zeros.')
        else:
            original_filenames = [widget.filename for widget in list(self.renaming_widgets.values())]
            new_filenames = [widget.new_filename for widget in list(self.renaming_widgets.values())]

            filename_renaming = {}
            for i, filename in enumerate(original_filenames):
                if filename != new_filenames[i]:
                    current_filename = os.path.join(self.directory, filename)
                    new_filename = os.path.join(self.directory, new_filenames[i])
                    filename_renaming[current_filename] = new_filename
                    if os.path.exists(current_filename):
                        try:
                            os.rename(current_filename, new_filename)
                        except OSError as e:
                            print(e)
                            messagebox.showerror('Error', str(e))

            self.filenames = [file for file in os.listdir(self.directory)]
            self.used_dataset_ids_per_subid = get_used_dataset_identifiers(self.filenames)
            for widget in self.renaming_widgets.values():
                widget.grid_forget()
            self.renaming_widgets = {}
            for i, filename in enumerate(self.filenames):
                used_ids = self.used_dataset_ids_per_subid[extract_subid(filename)]
                self.renaming_widgets[filename] = RenameFileRow(self.inner_frame, self, filename, i, self.directory,
                                                                used_ids)
                self.renaming_widgets[filename].grid(row=2 + i, column=0, pady=2, padx=2, sticky='w')
            filenames_df = pd.DataFrame({
                'Previous Filename': list(filename_renaming.keys()),
                'New Filename': list(filename_renaming.values()),
            })
            if not os.path.exists('Results/File_Renaming/'):
                os.mkdir('Results/File_Renaming/')
            filepath = 'Results/File_Renaming/' + (f'{self.directory.split("/")[-2]}_renaming_'
                                                   f'{date.today().strftime("%m.%d.%Y")}'
                                                   f'{datetime.now().strftime("%H-%M-%S")}.xlsx')
            print(filenames_df)
            filenames_df.to_excel(filepath, index=False)
            messagebox.showinfo('SDM Update',
                                f'Previous and new filenames are recorded in excel file located here'
                                f'{filepath}')

    def update_used_dataset_ids(self):
        filenames = [widget.new_filename for widget in list(self.renaming_widgets.values())]
        self.used_dataset_ids_per_subid = get_used_dataset_identifiers(filenames)
        print(self.used_dataset_ids_per_subid)
        for widget in list(self.renaming_widgets.values()):
            widget.used_dataset_ids = self.used_dataset_ids_per_subid[extract_subid(widget.new_filename)]
