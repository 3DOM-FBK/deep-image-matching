import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk


class MatcherApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Deep Image Matcher")

        self.strategy = self.create_combobox(
            "Matching strategy:",
            ["bruteforce", "sequential", "retrieval", "custom_pairs"],
        )

        self.image_dir = self.create_folder_button("Images directory")

        self.out_dir = self.create_folder_button("Output directory")

        self.pair_file = self.create_file_button(
            "If matching strategy == 'custom_pairs':", "Choose pairs file"
        )

        self.overlap = self.create_int_entry(
            "If matching strategy == 'sequential', insert image overlap:"
        )

        self.max_features = self.create_int_entry(
            "Max number of local features per image:"
        )

        self.local_feat = self.create_combobox(
            "Choose local features:",
            [
                "superglue",
                "lightglue",
                "loftr",
                "ALIKE",
                "superpoint",
                "KeyNetAffNetHardNet",
                "DISK",
                "ORB",
            ],
        )

        self.error_label = tk.Label(master, text="", fg="red")
        self.error_label.pack()

        self.submit_button = tk.Button(master, text="Submit", command=self.on_submit)
        self.submit_button.pack()

    def on_submit(self):
        strategy = self.strategy.get()
        image_dir = self.image_dir.get()
        out_dir = self.out_dir.get()
        pair_file = self.pair_file.get()
        image_overlap = self.overlap.get()
        local_feat = self.local_feat.get()
        max_features = self.max_features.get()

        print("image_dir:", image_dir)
        print("out_dir:", out_dir)
        print("strategy:", strategy)
        print("pair_file:", pair_file)
        print("image overlap:", image_overlap)
        print("local_feat", local_feat)
        print("max_features:", max_features)

        self.master.quit()

    def get_values(self):
        strategy = self.strategy.get()
        image_dir = self.image_dir.get()
        out_dir = self.out_dir.get()
        pair_file = self.pair_file.get()
        image_overlap = self.overlap.get()
        local_feat = self.local_feat.get()
        max_features = self.max_features.get()

        if not pair_file:
            pair_file = None
        else:
            pair_file = Path(pair_file)

        return (
            strategy,
            Path(image_dir),
            Path(out_dir),
            pair_file,
            int(image_overlap),
            local_feat,
            int(max_features),
        )

    def create_combobox(self, label_text, values):
        label = tk.Label(self.master, text=label_text)
        label.pack()
        combobox = ttk.Combobox(self.master, values=values)
        combobox.pack()
        return combobox

    def create_folder_button(self, text):
        var = tk.StringVar()
        folder_button = tk.Button(
            self.master, text=text, command=lambda: self.choose_folder(var)
        )
        folder_button.pack()
        folder_label = tk.Label(self.master, textvariable=var)
        folder_label.pack()
        return var

    def create_file_button(self, label_text, text):
        label = tk.Label(self.master, text=label_text)
        label.pack()
        var = tk.StringVar()
        file_button = tk.Button(
            self.master, text=text, command=lambda: self.choose_file(var)
        )
        file_button.pack()
        file_label = tk.Label(self.master, textvariable=var)
        file_label.pack()
        return var

    def choose_folder(self, var):
        folder = filedialog.askdirectory()
        var.set(folder)

    def choose_file(self, var):
        file = filedialog.askopenfilename()
        var.set(file)

    def create_int_entry(self, label_text):
        label = tk.Label(self.master, text=label_text)
        label.pack()
        int_entry = tk.Entry(self.master)
        int_entry.pack()
        return int_entry


def gui():
    root = tk.Tk()
    app = MatcherApp(root)
    root.mainloop()
    return app.get_values()


if __name__ == "__main__":
    gui()
