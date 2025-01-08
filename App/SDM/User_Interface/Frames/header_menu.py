from tkinter import Menu, Toplevel, Text, WORD, BOTH, END, NORMAL
import pandas as pd
from SDM.User_Interface.Sub_Windows.file_splitting_tool_window import FileSplittingToolWindow
from SDM.User_Interface.Sub_Windows.create_metadata_window import CreateMetadataWindow
from SDM.User_Interface.Utils.dataframe_dialog import DataFrameDialog
import webbrowser


class HeaderMenu(Menu):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.config(menu=self)

        # Elements of FILE dropdown
        file_menu = Menu(self)
        file_menu.add_command(label='Exit',
                              command=self.parent.destroy)
        file_menu.add_command(label='Split File',
                              command=self.open_file_splitter)
        file_menu.add_command(label='Create Metadata Template',
                              command=self.open_metadata_creator)

        self.add_cascade(
            label='File',
            menu=file_menu,
            underline=0
        )

        resources = Menu(self)
        resources.add_command(label='TAC Feature Key (sorted by category)',
                              command=self.open_variable_key)
        resources.add_command(label='TAC Feature Key (alphabetized)',
                              command=self.open_sorted_variable_key)
        resources.add_command(label='Feature Category Key',
                              command=self.open_feature_category_key)
        resources.add_command(label='Github Page',
                              command=self.open_github)
        resources.add_command(label='How To Cite',
                              command=self.open_citation)

        self.add_cascade(
            label='Resources',
            menu=resources,
            underline=0
        )

    def open_file_splitter(self):
        window = FileSplittingToolWindow(self, self.parent)
        self.wait_window(window)

    def open_metadata_creator(self):
        window = CreateMetadataWindow(self)
        self.wait_window(window)

    def open_variable_key(self, alphabetize=False):
        variable_key = pd.read_excel('App/SDM/Documenting/FeatureKey.xlsx')
        if alphabetize:
            variable_key = variable_key.sort_values(by='Name', key=lambda col: col.str.lower())

        DataFrameDialog(self, dataframe=variable_key, title='TAC Feature Key',
                        width_specs={'Name': 200, 'Category': 180, 'Definition': 900})

    def open_sorted_variable_key(self):
        self.open_variable_key(alphabetize=True)

    def open_feature_category_key(self):
        category_key = pd.read_excel('App/SDM/Documenting/FeatureKey.xlsx', sheet_name='Categories')

        DataFrameDialog(self, dataframe=category_key, title='TAC Feature Key',
                        width_specs={'Dictionary Categories': 130, 'Description': 1060})

    @staticmethod
    def open_github():
        webbrowser.open('https://github.com/ndidier3/skyn_data_manager')

    def open_citation(self):
        """
    Opens a new Tkinter window with a text widget that allows copy-paste.
    """
        citation = ("Didier, N. A., King, A. C., Polley, E. C., & Fridberg, D. J. (2024). Signal processing and "
                    "machine learning with transdermal alcohol concentration to predict natural environment alcohol "
                    "consumption. Experimental and Clinical Psychopharmacology, 32(2), 245–254. "
                    "https://doi.org/10.1037/pha0000683")

        # Create a new window
        citation_window = Toplevel(self)
        citation_window.title("Citation")

        text_widget = Text(citation_window, wrap=WORD, height=15, width=50)
        text_widget.pack(expand=True, fill=BOTH)
        text_widget.insert(END, citation)
        text_widget.config(state=NORMAL)

        citation_window.grab_set()
