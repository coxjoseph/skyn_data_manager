from tkinter import StringVar, Frame, Label, OptionMenu


class QualityControlSettings(Frame):
    def __init__(self, tab, settings_window):
        super().__init__(tab)

        self.tab = tab
        self.settings_window = settings_window
        self.sdm_interface = self.settings_window.sdm_interface

        # Header
        self.QualityControlHeader = Label(self, text='Set Exclusion Criteria')
        self.QualityControlHeader.config(font=(None, 12, 'bold'))
        self.QualityControlHeader.grid(row=0, column=0, pady=(10, 0), padx=10, sticky='w')

        # Note
        self.QualityControlNote = Label(self,
                                        text='Datasets that exceed below thresholds will be considered invalid and '
                                             'excluded from downstream analyses.')
        self.QualityControlNote.config(font=(None, 8, 'italic'))
        self.QualityControlNote.grid(row=1, column=0, pady=(2, 5), padx=10, sticky='w')

        # defaults
        self.min_duration_active_default = 1
        self.max_percentage_inactive_default = 30 if self.sdm_interface.selected_programs['Train+Test'] else 40
        self.max_percentage_imputed_default = 65 if self.sdm_interface.selected_programs['Train+Test'] else 70

        # Min Duration Active
        self.min_duration_active_text = ("Minimum Duration (minutes) with device worn and turned on. "
                                         "Lower values are more lenient.")
        self.min_duration_active_label = Label(self, text=self.min_duration_active_text)
        self.min_duration_active_label.grid(row=3, column=0, padx=5, pady=(8, 3), sticky='w')

        self.min_duration_active = StringVar(self)
        self.min_duration_active.set(str(round(self.min_duration_active_default * 60)))
        self.min_duration_options = [str(value) for value in range(45, 121)]
        self.min_duration_dropdown = OptionMenu(self, self.min_duration_active, *self.min_duration_options)
        self.min_duration_dropdown.grid(row=4, column=0, padx=20, pady=(3, 7), sticky='w')

        # Max Percentage Inactive
        self.max_percentage_inactive_text = ("Maximum Percentage of device inactivity (non-wear, not turned on). "
                                             "Higher values are more lenient.")
        self.max_percentage_inactive_label = Label(self, text=self.max_percentage_inactive_text)
        self.max_percentage_inactive_label.grid(row=5, column=0, padx=5, pady=3, sticky='w')

        self.max_percentage_inactive = StringVar(self)
        self.max_percentage_inactive.set(str(round(self.max_percentage_inactive_default)))
        self.max_percentage_inactive_options = [str(value) for value in range(10, 75)]
        self.max_percentage_inactive_dropdown = OptionMenu(self, self.max_percentage_inactive,
                                                           *self.max_percentage_inactive_options)
        self.max_percentage_inactive_dropdown.grid(row=6, column=0, padx=20, pady=(3, 7), sticky='w')

        # Max Percentage Imputed
        self.max_percentage_imputed_text = "Maximum Percentage of imputed values. Higher values are more lenient."
        self.max_percentage_imputed_label = Label(self, text=self.max_percentage_imputed_text)
        self.max_percentage_imputed_label.grid(row=7, column=0, padx=5, pady=3, sticky='w')

        self.max_percentage_imputed = StringVar(self)
        self.max_percentage_imputed.set(str(round(self.max_percentage_imputed_default)))
        self.max_percentage_imputed_options = [str(value) for value in range(40, 80)]
        self.max_percentage_imputed_dropdown = OptionMenu(self, self.max_percentage_imputed,
                                                          *self.max_percentage_imputed_options)
        self.max_percentage_imputed_dropdown.grid(row=8, column=0, padx=20, pady=(3, 7), sticky='w')
