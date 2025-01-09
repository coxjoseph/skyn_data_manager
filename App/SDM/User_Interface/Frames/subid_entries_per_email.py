from tkinter import StringVar, LabelFrame, Canvas, Scrollbar, Frame, Label, Entry


class SubidEntriesPerEmail(LabelFrame):
    def __init__(self, parent, emails, height=400):
        super().__init__(parent, height=height)

        self.subid_entries_per_email = {}

        # Create a canvas to hold the grid
        self.canvas = Canvas(self, height=height)
        self.scrollbar = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas to hold the grid
        self.grid_frame = Frame(self.canvas)
        self.grid_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Add grid frame to canvas
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Create header labels
        Label(self.grid_frame, text="Email").grid(row=0, column=0, padx=10, pady=5)
        Label(self.grid_frame, text="SubID").grid(row=0, column=1, padx=10, pady=5)

        self.validate_numeric = self.register(lambda p: (p.isdigit() and len(p) <= 6) or len(p) == 0)

        # Populate grid with email labels and SubID entries
        for i, email in enumerate(emails, start=1):
            Label(self.grid_frame, text=email).grid(row=i, column=0, padx=10, pady=5, sticky="w")
            subid_var = StringVar()
            entry = Entry(self.grid_frame, textvariable=subid_var,
                          validate="key", validatecommand=(self.validate_numeric, '%P'))
            entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
            self.subid_entries_per_email[email] = subid_var

        # Configure grid resizing
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Bind mouse wheel scrolling
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", self._on_mousewheel)  # For Linux
        self.bind("<Button-5>", self._on_mousewheel)  # For Linux

    def _on_mousewheel(self, event):
        # Scroll the canvas on mouse wheel
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
