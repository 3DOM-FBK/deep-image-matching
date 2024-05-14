import logging
import shutil
from pathlib import Path
from pprint import pprint

from .config import confs

logger = logging.getLogger("dim")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except:
    logger.warning("Not possible to import tkinter")


class MatcherApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Deep Image Matcher")

        additional_text_label = tk.Label(
            master,
            text="\nMultiview matching with deep-learning and hand-crafted local features\n",
        )
        additional_text_label.pack()

        self.image_dir = self.create_folder_button("Images directory")

        self.out_dir = self.create_folder_button("Output directory")

        self.config = self.create_combobox("Choose available matching configuration:", list(confs.keys()))

        self.strategy = self.create_combobox(
            "Matching strategy:",
            ["bruteforce", "sequential", "retrieval", "custom_pairs"],
        )

        self.pair_file = self.create_file_button("If matching strategy == 'custom_pairs':", "Choose pairs file")

        self.overlap = self.create_int_entry("If matching strategy == 'sequential', insert image overlap:")

        additional_text_label = tk.Label(
            master,
            text="\nTry to rotate upright images (useful for not rotation invariant local features). \n Features are extracted on upright images, but rotatated accordenly to be used with the original images",
        )
        additional_text_label.pack()
        self.use_custom = tk.BooleanVar()
        self.use_custom.set(False)
        self.upright = tk.Checkbutton(
            master,
            text="Upright",
            variable=self.use_custom,
            # command=self.toggle_custom_pairs,
        )
        self.upright.pack()

        self.error_label = tk.Label(master, text="", fg="red")
        self.error_label.pack()

        self.submit_button = tk.Button(master, text="Submit", command=self.on_submit)
        self.submit_button.pack()

    # def toggle_custom_pairs(self):
    #    state = "normal" if self.use_custom.get() else "disabled"
    #    self.pair_file["state"] = state

    def on_submit(self):
        args = {
            "image_dir": Path(self.image_dir.get()),
            "out_dir": Path(self.out_dir.get()),
            "config": self.config.get(),
            "strategy": self.strategy.get(),
            "pair_file": self.pair_file.get(),
            "image_overlap": self.overlap.get(),
            "upright": self.use_custom.get(),
        }
        pprint(args)

        self.master.quit()

    def get_values(self):
        args = {
            "image_dir": Path(self.image_dir.get()),
            "out_dir": Path(self.out_dir.get()),
            "config": self.config.get(),
            "strategy": self.strategy.get(),
            "pair_file": self.pair_file.get(),
            "image_overlap": self.overlap.get(),
            "upright": self.use_custom.get(),
        }

        if not args["image_dir"].exists() or not args["image_dir"].is_dir():
            msg = f"Directory {args['image_dir']} does not exist"
            messagebox.showerror("Error", msg)
            raise ValueError(msg)

        if not args["pair_file"]:
            args["pair_file"] = None
        else:
            args["pair_file"] = Path(args["pair_file"])
            if not args["pair_file"].exists():
                msg = f"File {args['pair_file']} does not exist"
                messagebox.showerror("Error", msg)
                raise ValueError(msg)

        if args["out_dir"].exists():
            answer = messagebox.askokcancel(
                "Warning",
                f"Directory {args['out_dir']} already exists, previous results will be overwritten. Do you wish to proceed?",
            )
            if not answer:
                return None
            else:
                shutil.rmtree(args["out_dir"])
        args["out_dir"].mkdir(parents=True, exist_ok=True)

        if args["strategy"] == "sequential":
            if not args["image_overlap"]:
                msg = "Image overlap is required when strategy is set to sequential"
                messagebox.showerror("Error", msg)
                raise ValueError(msg)
            else:
                args["image_overlap"] = int(args["image_overlap"])

        # if not args["max_features"]:
        #    self.error_label[
        #        "text"
        #    ] = "Invalid max number of local features per image. Using the default value."
        # else:
        #    args["max_features"] = int(args["max_features"])

        return args

    def create_combobox(self, label_text, values):
        label = tk.Label(self.master, text=label_text)
        label.pack()
        combobox = ttk.Combobox(self.master, values=values, width=40)
        combobox.pack()
        return combobox

    def create_folder_button(self, text):
        var = tk.StringVar()
        folder_button = tk.Button(self.master, text=text, command=lambda: self.choose_folder(var))
        folder_button.pack()
        folder_label = tk.Label(self.master, textvariable=var)
        folder_label.pack()
        return var

    def create_file_button(self, label_text, text):
        label = tk.Label(self.master, text=label_text)
        label.pack()
        var = tk.StringVar()
        file_button = tk.Button(self.master, text=text, command=lambda: self.choose_file(var))
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
