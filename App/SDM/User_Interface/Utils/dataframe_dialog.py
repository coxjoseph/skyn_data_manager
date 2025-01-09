from tkinter import ttk, Toplevel


class DataFrameDialog(Toplevel):
    def __init__(self, master=None, dataframe=None, title='Variable Key', width_specs=None):
        super().__init__(master)
        if width_specs is None:
            width_specs = {}
        self.title(title)
        self.geometry("1200x400")

        table_height = len(dataframe) if len(dataframe) < 20 else 20
        treeview = ttk.Treeview(self, height=table_height)
        treeview.pack(expand=True)

        treeview["columns"] = dataframe.columns.tolist()
        treeview["show"] = "headings"

        # Define columns in the Treeview
        for col in dataframe.columns:
            treeview.heading(col, text=col)  # Set the column heading

            if col in list(width_specs.keys()):
                treeview.column(col, width=width_specs[col], anchor=W)
            else:
                treeview.column(col, width=600, anchor=W)  # Set column alignment and width

        # Insert data from the DataFrame into the Treeview
        for index, row in dataframe.iterrows():
            treeview.insert("", "end", values=row.tolist())
