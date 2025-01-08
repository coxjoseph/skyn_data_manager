from SDM.Configuration.file_management import load_default_model
from tkinter import IntVar, Label, Checkbutton, Frame


class ModelSelection(Frame):
    def __init__(self, tab, settings_window):
        super().__init__(settings_window)

        self.tab = tab
        self.settings_window = settings_window
        self.sdm_interface = settings_window.sdm_interface

        self.header = Label(self, text='Make predictions using the following built-in models:')
        self.selected_models = []

        self.AlcNonRF = IntVar(self)
        self.AlcNonRFCheckbutton = Checkbutton(self, text='Alcohol vs. Non-Alcohol (Random Forest)',
                                               variable=self.AlcNonRF, command=self.update_model_selections)
        self.AlcNonRFCheckbutton.grid(row=3, column=0, padx=10, pady=5, sticky='w')

        self.AlcNonLR = IntVar(self)
        self.AlcNonLRCheckbutton = Checkbutton(self, text='Alcohol vs. Non-Alcohol (Logistic Regression)',
                                               variable=self.AlcNonLR, command=self.update_model_selections)
        self.AlcNonLRCheckbutton.grid(row=4, column=0, padx=10, pady=5, sticky='w')

        self.BingeRF = IntVar(self)
        self.BingeRFCheckbutton = Checkbutton(self, text='Heavy vs. Light vs. No Drinking (Random Forest)',
                                              variable=self.BingeRF, command=self.update_model_selections)
        self.BingeRFCheckbutton.grid(row=5, column=0, padx=10, pady=5, sticky='w')

        self.BingeLR = IntVar(self)
        self.BingeLRCheckbutton = Checkbutton(self, text='Heavy vs. Light vs. No Drinking (Logistic Regression)',
                                              variable=self.BingeLR, command=self.update_model_selections)
        self.BingeLRCheckbutton.grid(row=6, column=0, padx=10, pady=5, sticky='w')

    def update_model_selections(self):
        """to do: move this function to when the app runs
        """

        model_outcomes = ['Alc_vs_Non', 'Alc_vs_Non', 'Binge', 'Binge']
        model_types = ['RF', 'LR', 'RF', 'LR']
        options = [self.AlcNonRF.get(), self.AlcNonLR.get(), self.BingeRF.get(), self.BingeLR.get()]

        for i, option in enumerate(options):
            model = load_default_model(model_outcomes[i], model_types[i])
            if option == 1:
                already_saved = any(
                    selected_model.model_name == model.model_name for selected_model in self.selected_models)
                if not already_saved:
                    self.selected_models.append(model)
            else:
                index_to_remove = next((index for index, selected_model in enumerate(self.selected_models) if
                                        selected_model.model_name == model.model_name), None)
                if index_to_remove is not None:
                    self.selected_models.pop(index_to_remove)

        self.sdm_interface.models = self.selected_models
