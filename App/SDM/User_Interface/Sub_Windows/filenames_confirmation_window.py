from SDM.User_Interface.Utils.filename_tools import create_metadata_from_cohort_folder
from tkinter import Toplevel, Listbox, Label, Frame, Button, Variable


class FilenamesConfirmationWindow(Toplevel):
    def __init__(self, parent, parsing_indices, cohort_folder):
        super().__init__(parent)

        self.geometry("500x600")
        self.title('Confirm Filenames')
        self.grab_set()
        self.lift()

        self.parent = parent
        self.parsing_indices = parsing_indices

        data = create_metadata_from_cohort_folder(cohort_folder)
        self.episode_labels = Variable(
            value=[
                f'Subject: {data["SubID"][i]} | '
                f'ID: {data["Dataset_Identifier"][i] if data["Dataset_Identifier"][i] else "NA"}'
                for i in range(0, len(data['SubID']))])

        self.fileDataListbox = Listbox(self, selectmode="none", height=18, width=60, listvariable=self.episode_labels)
        self.fileDataListbox.grid(row=22, column=1, padx=5, pady=5)

        self.verifyLabel = Label(self, text='Are filenames parsed correctly?')
        self.verifyLabel.grid(row=23, column=1, padx=5, pady=(7, 2))

        self.yes_no_frame = Frame(self)
        self.yes_no_frame.grid(row=24, column=1)

        self.confirmed = False
        self.yesButton = Button(self.yes_no_frame, text='Yes', width=10, command=self.submit_yes)
        self.yesButton.grid(column=0, row=0, padx=(0, 10), pady=5)
        self.noButton = Button(self.yes_no_frame, text='No', width=10, command=self.submit_no)
        self.noButton.grid(column=1, row=0, padx=(10, 0), pady=5)

    def submit_yes(self):
        self.confirmed = True
        self.destroy()

    def submit_no(self):
        self.confirmed = False
        self.destroy()
