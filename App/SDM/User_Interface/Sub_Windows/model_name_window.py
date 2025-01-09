from tkinter import Toplevel, messagebox, Button, Label, Entry


class ModelNameWindow(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.geometry("200x100")
        self.title('Model Name')
        self.parent = parent

        self.model_name = ''

        self.header = Label(self, text='Give a name for this model.')
        self.header.grid(row=0, column=0, padx=10, pady=5)

        self.model_name_entry = Entry(self, width=20)
        self.model_name_entry.grid(row=1, column=0, padx=10, pady=(0, 5))

        self.submit_button = Button(self, width=10, text='Submit', command=self.submit)
        self.submit_button.grid(row=2, column=0, padx=10, pady=10)

    def submit(self):
        if str(self.model_name_entry.get()) == '':
            messagebox.showerror('SDM Error', 'Must provide model name')
        elif str(self.model_name_entry.get()) in list(self.parent.models.keys()):
            messagebox.showerror('SDM Error', 'Model name already used.')
        else:
            self.model_name = str(self.model_name_entry.get())
            self.destroy()
