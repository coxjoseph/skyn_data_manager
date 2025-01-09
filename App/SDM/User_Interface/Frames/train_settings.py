from tkinter import StringVar, IntVar, Frame, Label, Checkbutton, OptionMenu


class TrainSettings(Frame):
    def __init__(self, tab, settings_window):
        super().__init__(tab)

        self.tab = tab
        self.settings_window = settings_window
        self.sdm_interface = self.settings_window.sdm_interface

        self.selected_models = []

        self.ModelTypeHeader = Label(self, text='Choose Model Types:')
        self.ModelTypeHeader.config(font=(None, 12, 'bold'))
        self.ModelTypeHeader.grid(column=1, row=0, pady=(5, 0), padx=10, sticky='w')

        self.ModelTypeFrame = Frame(self)
        self.ModelTypeFrame.grid(column=1, row=1, padx=5, pady=5, sticky='w')

        self.AlcNonRF = IntVar(self)
        self.AlcNonRFCheckbutton = Checkbutton(self.ModelTypeFrame, text='Alcohol vs. Non-Alcohol (Random Forest)',
                                               variable=self.AlcNonRF, command=self.update_model_selections)
        self.AlcNonRFCheckbutton.grid(row=1, column=0, padx=10, pady=5, sticky='w')

        self.AlcNonLR = IntVar(self)
        self.AlcNonLRCheckbutton = Checkbutton(self.ModelTypeFrame,
                                               text='Alcohol vs. Non-Alcohol (Logistic Regression)',
                                               variable=self.AlcNonLR, command=self.update_model_selections)
        self.AlcNonLRCheckbutton.grid(row=2, column=0, padx=10, pady=5, sticky='w')

        self.BingeRF = IntVar(self)
        self.BingeRFCheckbutton = Checkbutton(self.ModelTypeFrame,
                                              text='Heavy vs. Light vs. No Drinking (Random Forest)',
                                              variable=self.BingeRF, command=self.update_model_selections)
        self.BingeRFCheckbutton.grid(row=3, column=0, padx=10, pady=5, sticky='w')

        self.BingeLR = IntVar(self)
        self.BingeLRCheckbutton = Checkbutton(self.ModelTypeFrame,
                                              text='Heavy vs. Light vs. No Drinking (Logistic Regression)',
                                              variable=self.BingeLR, command=self.update_model_selections)
        self.BingeLRCheckbutton.grid(row=4, column=0, padx=10, pady=5, sticky='w')

        # choose column from metadata
        self.binary_columns = [''] + [col for col in self.sdm_interface.metadata_df.columns if
                                      self.sdm_interface.metadata_df[col].nunique() == 2]

        self.CustomRF = IntVar(self)
        self.CustomRFCheckbutton = Checkbutton(self.ModelTypeFrame, text='Custom (Random Forest)',
                                               variable=self.CustomRF, command=self.update_model_selections)
        self.CustomRFCheckbutton.grid(row=5, column=0, padx=10, pady=5, sticky='w')

        self.CustomRFOutcome = StringVar(self)
        self.CustomRFOutcome.set('')
        self.CustomRFDropdown = OptionMenu(self.ModelTypeFrame, self.CustomRFOutcome, *self.binary_columns,
                                           command=self.set_custom_rf)

        self.CustomLR = IntVar(self)
        self.CustomLRCheckbutton = Checkbutton(self.ModelTypeFrame, text='Custom (Logistic Regression)',
                                               variable=self.CustomLR, command=self.update_model_selections)
        self.CustomLRCheckbutton.grid(row=7, column=0, padx=10, pady=5, sticky='w')

        self.CustomLROutcome = StringVar(self)
        self.CustomLROutcome.set('')
        self.CustomLRDropdown = OptionMenu(self.ModelTypeFrame, self.CustomLROutcome, *self.binary_columns,
                                           command=self.set_custom_lr)

        self.features_header = Label(self, text='Choose Features:')
        self.features_header.config(font=(None, 12, 'bold'))
        self.features_header.grid(column=1, row=10, pady=(5, 0), padx=10, sticky='w')

        self.note1 = Label(self, text='CLN = feature calculated after cleaning/smoothing')
        self.note1.config(font=(None, 8))
        self.note1.grid(column=1, row=11, pady=(0, 0), padx=20, sticky='w')

        self.note2 = Label(self, text='RAW = feature calculated from raw/unprocessed data')
        self.note2.config(font=(None, 8))
        self.note2.grid(column=1, row=12, pady=(0, 0), padx=20, sticky='w')

        self.note3 = Label(self, text='For more details on features, see Inputs')
        self.note3.config(font=(None, 8))
        self.note3.grid(column=1, row=13, pady=(0, 5), padx=20, sticky='w')

        self.predictor_options = ['rise_rate_CLN', 'rise_duration_CLN', 'fall_rate_CLN', 'fall_duration_CLN',
                                  'peak_CLN', 'relative_peak_CLN', 'completed_curve_count_CLN', 'curve_alterations_CLN',
                                  'auc_total_CLN', 'major_outlier_N', 'minor_outlier_N', 'rise_rate_RAW',
                                  'rise_duration_RAW', 'fall_rate_RAW', 'fall_duration_RAW',
                                  'completed_curve_count_RAW', 'curve_alterations_RAW', 'auc_total_RAW',
                                  'avg_tac_diff_RAW', 'tac_alt_perc_RAW', 'fall_completion', 'curve_start_time']

        self.predictor_frame = Frame(self)
        self.predictor_frame.grid(column=1, row=14, padx=10, pady=10)

        row_idx = 0
        col_idx = 0
        self.predictor_checkboxes = {}
        for i, predictor in enumerate(self.predictor_options):
            checked = IntVar(self.predictor_frame, value=1)
            checkbox = Checkbutton(self.predictor_frame, text=predictor, variable=checked, height=1)
            checkbox.config(font=(None, 9))
            checkbox.grid(column=col_idx, row=row_idx, pady=0, padx=3, sticky='w')
            self.predictor_checkboxes[predictor] = checked

            if row_idx == 7:
                row_idx = 0
            else:
                row_idx += 1

            if (i + 1) % 8 == 0:
                col_idx += 1

    def update_model_selections(self):
        model_outcomes = ['Alc_vs_Non', 'Alc_vs_Non', 'Binge', 'Binge', 'Custom', 'Custom']
        model_types = ['RF', 'LR', 'RF', 'LR', 'RF', 'LR']
        options = [self.AlcNonRF.get(), self.AlcNonLR.get(), self.BingeRF.get(), self.BingeLR.get(),
                   self.CustomRFOutcome.get(), self.CustomLROutcome.get()]

        for i, option in enumerate(options):
            model_name = model_types[i] + '_' + model_outcomes[i] if model_outcomes[i] != 'Custom' else (model_types[
                                                                                                            i] +
                                                                                                         '_' + option)
            if (isinstance(option, int) and option == 1) or (isinstance(option, str) and option != ''):
                if model_name not in self.selected_models:
                    self.selected_models.append(model_name)
            else:
                if model_name in self.selected_models:
                    self.selected_models.remove(model_name)

        if self.CustomRF.get():
            self.CustomRFDropdown.grid(row=6, column=0, pady=0, padx=20, sticky='w')
        else:
            self.CustomRFDropdown.grid_forget()

        if self.CustomLR.get():
            self.CustomLRDropdown.grid(row=8, column=0, pady=0, padx=20, sticky='w')
        else:
            self.CustomLRDropdown.grid_forget()

    def set_custom_rf(self, val):
        self.CustomRFOutcome.set(val)

    def set_custom_lr(self, val):
        self.CustomLROutcome.set(val)
