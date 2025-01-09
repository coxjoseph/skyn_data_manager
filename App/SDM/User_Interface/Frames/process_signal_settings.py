from tkinter import ttk
from tkinter import Frame, Label, OptionMenu, StringVar
from tkinter import messagebox


class ProcessSignalSettings(Frame):
    def __init__(self, tab, settings_window):
        super().__init__(tab)

        self.tab = tab
        self.settings_window = settings_window
        self.sdm_interface = self.settings_window.sdm_interface

        # Remove Artifacts Checkbox
        self.imputeArtifactsLabel = ttk.Label(self, text='Remove & Impute TAC Artifacts')
        self.imputeArtifactsLabel.config(font=(None, 11, 'bold'))
        self.imputeArtifactsLabel.grid(column=0, row=1, pady=(5, 0), padx=10, sticky='w')

        # Major Cleaning Threshold
        self.majorThresholdLabel = Label(self, text='Major Cleaning Threshold')
        self.majorThreshold = StringVar(self)
        self.majorThreshold.set("0.8")
        self.majorThresholdDropdown = OptionMenu(self, self.majorThreshold,
                                                 *[str(round(0.5 + 0.05 * i, 3)) for i in range(11)])
        self.majorThreshold.trace_add("write", self.validate_threshold_selections)

        # Minor Cleaning Threshold
        self.minorThresholdLabel = Label(self, text='Minor Cleaning Threshold')
        self.minorThreshold = StringVar(self)
        self.minorThreshold.set("0.5")
        self.minorThresholdDropdown = OptionMenu(self, self.minorThreshold,
                                                 *[str(round(0.3 + 0.05 * i, 3)) for i in range(12)])
        self.minorThreshold.trace_add("write", self.validate_threshold_selections)

        # Smooth Signal
        self.smoothSignalLabel = Label(self, text='Smooth TAC Signal')
        self.smoothSignalLabel.config(font=(None, 11, 'bold'))
        self.smoothSignalLabel.grid(column=0, row=7, padx=10, pady=(3, 0), sticky='w')

        # Smoothing Window Selection
        self.smoothingWindowLabel = Label(self, text='Select window length (N data points) for smoothing.')
        self.smoothingWindowNote = Label(self,
                                         text='As window size increases, curves become more smoothed / less jagged.')
        self.smoothingWindowNote.config(font=(None, 8, 'italic'))
        self.smoothingWindow = StringVar(self)
        self.smoothingWindow.set("51")
        self.smoothingWindowDropdown = OptionMenu(self, self.smoothingWindow, *[11 + (10 * i) for i in range(0, 16)])

        # Device Removal Detection
        self.imputeDeviceRemovalLabel = Label(self, text='Detect and Impute Device Removal')
        self.imputeDeviceRemovalLabel.config(font=(None, 11, 'bold'))
        self.imputeDeviceRemovalLabel.grid(column=0, row=15, padx=10, pady=(3, 0), sticky='w')

        # Device Removal Detection Method
        self.deviceRemovalDetectionLabel = Label(self, text='Select method for detection device removal.')
        self.deviceRemovalDetection = StringVar(self)
        self.deviceRemovalDetection.set('Built-in Algorithm')
        self.deviceRemovalDetectionDropdown = OptionMenu(self, self.deviceRemovalDetection,
                                                         *['Built-in Algorithm', 'Temp Cutoff (30 Celsius)',
                                                           'Temp Cutoff (29 Celsius)', 'Temp Cutoff (28 Celsius)',
                                                           'Temp Cutoff (27 Celsius)', 'Temp Cutoff (26 Celsius)',
                                                           'Temp Cutoff (25 Celsius)'])

        self.majorThresholdLabel.grid(column=0, row=2, pady=(2, 0), padx=(35, 10), sticky='w')
        self.majorThresholdDropdown.grid(column=0, row=3, pady=(2, 5), padx=(35, 10), sticky='w')
        self.minorThresholdLabel.grid(column=0, row=4, pady=(2, 0), padx=(35, 10), sticky='w')
        self.minorThresholdDropdown.grid(column=0, row=5, pady=(2, 5), padx=(35, 10), sticky='w')
        self.smoothingWindowLabel.grid(column=0, row=9, pady=(2, 0), padx=(35, 10), sticky='w')
        self.smoothingWindowNote.grid(column=0, row=10, pady=0, padx=(35, 10), sticky='w')
        self.smoothingWindowDropdown.grid(column=0, row=11, pady=(0, 10), padx=(35, 10), sticky='w')

        self.deviceRemovalDetectionDropdown.grid(column=0, row=16, pady=(0, 10), padx=(35, 10), sticky='w')

    def validate_threshold_selections(self, *_):
        # Get the selected values from the OptionMenus
        major = round(float(self.majorThreshold.get()), 3)
        minor = round(float(self.minorThreshold.get()), 3)

        # major threshold must be greater than minor
        if major <= minor:
            threshold_diff = minor - major
            revised_major = round(major + (0.05 + threshold_diff) if minor != 1 else major, 3)
            revised_minor = round(minor - (0.05 + threshold_diff) if minor == 1 else minor, 3)
            messagebox.showwarning("Invalid Selection",
                                   f'Major threshold must be greater than the minor threshold. '
                                   f'{"Major" if revised_major != major else "Minor"} threshold has been set to '
                                   f'{revised_major if revised_major != major else revised_minor}')

            if revised_major != major:
                self.majorThreshold.set(str(revised_major))
            else:
                self.minorThreshold.set(str(revised_minor))
