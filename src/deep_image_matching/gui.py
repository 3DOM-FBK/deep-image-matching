import tkinter as tk
from tkinter import ttk

def on_submit():
    print(f"User chose: {combo.get()}")
    window.quit()

if __name__ == "__main__":
    # Create the main window
    window = tk.Tk()
    window.title("Deep-image-matching")

    # Create a label
    label = tk.Label(window, text="Choose an option:")
    label.pack()

    # Create a dropdown menu
    options = ["Option 1", "Option 2", "Option 3"]
    combo = ttk.Combobox(window, values=options)
    combo.pack()

    # Create a submit button
    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack()

    # Start the main loop
    window.mainloop()