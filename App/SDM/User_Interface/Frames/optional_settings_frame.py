from tkinter import filedialog, IntVar, Frame, Label, Button, Checkbutton
from tkinter import messagebox
from App.SDM.User_Interface.Frames.crop_settings import CropSettings
from App.SDM.User_Interface.Frames.merge_files import MergeFiles
import pickle


class OptionalSettingsFrame(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.configure(highlightbackground="black", highlightthickness=2)

        self.header = Label(self, text='Optional Settings')
        self.header.grid(row=0, column=0, padx=5, pady=(3, 7))
        self.header.config(font=(None, 12, 'bold'))

        if self.parent.program != 'P' and self.parent.data_loading_method != 'Single':
            # checkbox to enable merging other Excel datasets
            self.mergeVariablesButton = Button(self, text='Merge Files with Results',
                                               command=self.open_merge_files_window)
            self.mergeVariablesButton.grid(row=1, column=0, padx=5, pady=5)

        # checkbox to enable cropping of TAC datasets
        if self.parent.data_loading_method != 'Processor':
            self.cropDatasetsButton = Button(self, text='Crop Datasets', command=self.open_crop_settings_window)
            self.cropDatasetsButton.grid(row=2, column=0, padx=5, pady=5)

        # checkbox to load model
        self.models = self.parent.models
        self.selected_model_files = []
        if self.parent.program:
            if self.parent.program == 'PP':
                self.loadModel = IntVar()
                self.loadModelCheckbox = Checkbutton(self,
                                                     text='Check to load a different model than the built-in / default model.',
                                                     variable=self.loadModel, command=self.show_model_button)
                self.loadModelCheckbox.grid(row=6, column=0, padx=5, pady=(5, 3))

                self.openModelButton = Button(self, text='Select SDM Trained Model (.sdmtm)', width=40,
                                              command=self.open_model)
                self.clearModelsButton = Button(self, text='Clear all models', width=20, command=self.reset_models)

                self.submittedModelsLabel = Label(self, text='Submitted Models:')

    # Todo: This isnt right... something's wrong here...
    def open_merge_files_window(self):
        sub_window = MergeFiles(self)
        sub_window.grab_set()
        self.wait_window(sub_window)

    def open_crop_settings_window(self, window, tab):
        sub_window = CropSettings(tab, window)
        sub_window.grab_set()
        self.wait_window(sub_window)

    def show_model_button(self):
        if self.loadModel.get() == 1:
            self.openModelButton.grid(row=7, column=0, padx=5, pady=2)
            self.clearModelsButton.grid(row=8, column=0, padx=5, pady=2)
            self.submittedModelsLabel.grid(row=9, column=0, padx=10, pady=5, sticky='w')
        else:
            self.openModelButton.grid_forget()
            self.clearModelsButton.grid_forget()
            self.submittedModelsLabel.grid_forget()
            self.reset_models()

    def reset_models(self):
        self.parent.reset_models()
        self.selected_model_files = []
        self.models = []
        self.submittedModelsLabel['text'] = "Submitted Models: "

    def open_model(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('Trained Model', '*.sdmtm'), ('Trained Model', '*.pickle')])
        if file:
            try:
                pickle_in = open(file.name, "rb")
                if file.name in self.selected_model_files:
                    messagebox.showerror('SDM Error', 'Model already selected.')
                else:
                    self.selected_model_files.append(file.name)
                    model = pickle.load(pickle_in)
                    pickle_in.close()
                    self.models.append(model)
                    self.parent.update_models(model)
                    self.submittedModelsLabel['text'] = "Submitted Models: " + ", ".join(
                        [model.model_name for model in self.models])
            except (FileNotFoundError, IOError):
                messagebox.showerror('SDM Error', f'Failed to load: {str(file.name.split("/")[-1])}')
