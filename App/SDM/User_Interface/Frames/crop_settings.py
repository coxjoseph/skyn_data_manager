from tkinter import StringVar, IntVar, Frame, Label, Entry, Checkbutton, OptionMenu


class CropSettings(Frame):
    def __init__(self, tab, settings_window):
        super().__init__(tab)

        self.tab = tab
        self.settings_window = settings_window
        self.sdm_interface = self.settings_window.sdm_interface

        self.crop_window = None

        # set max episode duration
        self.maxDatasetDurationLabelText = 'Max Dataset Duration (hours)'
        self.maxDatasetDurationLabel = Label(self, text=self.maxDatasetDurationLabelText)
        self.maxDatasetDurationLabel.config(font=(None, 11, 'bold'))
        self.maxDatasetDurationLabel.grid(row=1, column=0, padx=10, pady=(5, 2), sticky='w')

        self.max_dataset_duration = self.sdm_interface.max_dataset_duration
        self.maxDatasetDuration = IntVar()
        self.maxDatasetDuration.set(self.max_dataset_duration)
        self.validate_max_dataset_duration = self.register(lambda p: len(p) <= 2)
        self.maxDatasetDurationEntry = Entry(self, textvariable=self.maxDatasetDuration, width=6, validate="key",
                                             validatecommand=(self.validate_max_dataset_duration, "%P"))
        self.maxDatasetDurationEntry.grid(row=2, column=0, padx=20, pady=(2, 5), sticky='w')

        if len(self.sdm_interface.metadata_df):
            self.crop_start_timestamps_available = (self.sdm_interface.metadata_df['Crop Begin Date'].notnull().any()
                                                    and self.sdm_interface.metadata_df['Crop Begin Time'].
                                                    notnull().any())
        else:
            self.crop_start_timestamps_available = False

        self.disableCropStartText = "Crop datasets at START, using timestamps detected in metadata."
        self.disableCropStart = IntVar(self)
        self.disableCropStart.set(1 if self.crop_start_timestamps_available else 0)
        self.disableCropStartCheckbox = Checkbutton(self, text=self.disableCropStartText,
                                                    variable=self.disableCropStart)
        if self.crop_start_timestamps_available:
            self.disableCropStartCheckbox.grid(row=3, column=0, padx=10, pady=5, sticky='w')

        if len(self.sdm_interface.metadata_df):
            self.crop_end_timestamps_available = self.sdm_interface.metadata_df['Crop End Date'].notnull().any() and \
                                                 self.sdm_interface.metadata_df['Crop End Time'].notnull().any()
        else:
            self.crop_end_timestamps_available = False
        self.disableCropEndText = "Crop datasets at END, using timestamps detected in metadata."
        self.disableCropEnd = IntVar(self)
        self.disableCropEnd.set(1 if self.crop_end_timestamps_available else 0)
        self.disableCropEndCheckbox = Checkbutton(self, text=self.disableCropEndText, variable=self.disableCropEnd)
        if self.crop_end_timestamps_available:
            self.disableCropEndCheckbox.grid(row=4, column=0, padx=10, pady=5, sticky='w')

        self.timezoneLabelText = "Timezone of Skyn Upload"
        self.timezoneLabel = Label(self, text=self.timezoneLabelText)
        self.timezoneLabel.config(font=(None, 11, 'bold'))
        self.timezoneLabel.grid(row=5, column=0, padx=10, pady=(5, 0), sticky='w')

        self.timezoneNote = Label(self, text='If participants uploaded their own data, enter 999.')
        self.timezoneNote.config(font=(None, 8, 'italic'))
        self.timezoneNote.grid(row=6, column=0, padx=10, pady=(0, 5), sticky='w')

        # self.skyn_upload_timezone = self.sdm_interface.skyn_upload_timezone
        self.timezoneUpload = StringVar(self)
        self.timezoneUpload.set('999')
        self.timezoneDropMenu = OptionMenu(self, self.timezoneUpload, *['CST', 'EST', 'MST', 'PST', '999'])
        self.timezoneDropMenu.grid(row=7, column=0, padx=20, pady=5, sticky='w')
