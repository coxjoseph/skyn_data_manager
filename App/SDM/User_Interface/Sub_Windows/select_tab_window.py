from tkinter import Toplevel, Listbox, SINGLE, END, Button, messagebox, Label


class SelectTabWindow(Toplevel):
    def __init__(self, parent, sheet_names):
        super().__init__(parent)
        self.title("Select Tab")
        self.geometry("200x300")

        self.parent = parent
        self.selected_sheet = self.parent.selected_sheet

        self.selectTabLabel = Label(self, text='Select Sheet Name for Merge')
        self.selectTabLabel.grid(row=0, column=0, padx=1, pady=5)

        self.sheetNamesListbox = Listbox(self, height=12, selectmode=SINGLE)
        for sheet_name in sheet_names:
            self.sheetNamesListbox.insert(END, sheet_name)
        self.sheetNamesListbox.grid(row=1, column=0, padx=5, pady=5)

        self.submitButton = Button(self, text='Submit', command=self.submit_sheet)
        self.submitButton.grid(row=2, column=0, pady=(10, 5), padx=5)

    def submit_sheet(self):
        selected_index = self.sheetNamesListbox.curselection()
        print(selected_index, 'index')
        if selected_index:
            self.selected_sheet = self.sheetNamesListbox.get(selected_index[0])
            print(self.selected_sheet)
            self.destroy()
        else:
            messagebox.showerror('SDM Error', 'You must select a tab.')
