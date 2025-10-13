import tkinter as tk
from tkinter import ttk, messagebox
import importlib.util
import os

GAN_FOLDER = "."  # Current folder

class GANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Clinical Document Generator")
        self.root.geometry("600x500")

        self.gan_files = self.find_gan_files()
        self.selected_gan = None
        self.gan_module = None
        self.constraint_entries = {}

        self.build_gui()

    def find_gan_files(self):
        return [f for f in os.listdir(GAN_FOLDER) if f.startswith("gan_") and f.endswith(".py")]

    def build_gui(self):
        ttk.Label(self.root, text="Select GAN Model:").pack(pady=5)

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.X, padx=10)

        self.gan_listbox = tk.Listbox(frame, height=5)
        self.gan_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.gan_listbox.bind("<<ListboxSelect>>", self.on_gan_select)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.gan_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gan_listbox.config(yscrollcommand=scrollbar.set)

        for gan_file in self.gan_files:
            self.gan_listbox.insert(tk.END, gan_file)

        self.constraints_frame = ttk.LabelFrame(self.root, text="Constraints")
        self.constraints_frame.pack(fill=tk.X, padx=10, pady=10)

        self.generate_button = ttk.Button(self.root, text="Generate Document", command=self.generate_document)
        self.generate_button.pack(pady=10)

        self.output_box = tk.Text(self.root, height=10, wrap=tk.WORD)
        self.output_box.pack(fill=tk.BOTH, padx=10, pady=10)

    def on_gan_select(self, event):
        selection = self.gan_listbox.curselection()
        if not selection:
            return

        selected_file = self.gan_files[selection[0]]
        self.selected_gan = selected_file
        self.load_gan_module(selected_file)
        self.show_constraints()

    def load_gan_module(self, filename):
        path = os.path.join(GAN_FOLDER, filename)
        spec = importlib.util.spec_from_file_location("gan_module", path)
        self.gan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gan_module)

    def show_constraints(self):
        for widget in self.constraints_frame.winfo_children():
            widget.destroy()
        self.constraint_entries.clear()

        if hasattr(self.gan_module, "get_constraints"):
            constraints = self.gan_module.get_constraints()
            for constraint in constraints:
                row = ttk.Frame(self.constraints_frame)
                row.pack(fill=tk.X, pady=2)

                label = ttk.Label(row, text=constraint + ":")
                label.pack(side=tk.LEFT, padx=5)

                entry = ttk.Entry(row)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                self.constraint_entries[constraint] = entry

    def validate_constraints(self):
        valid = True
        for name, entry in self.constraint_entries.items():
            value = entry.get().strip()
            if "%" in name:
                try:
                    val = float(value)
                    if not (0 <= val <= 100):
                        raise ValueError
                except ValueError:
                    messagebox.showerror("Invalid Input", f"{name} must be a number between 0 and 100.")
                    valid = False
        return valid

    def generate_document(self):
        if not self.gan_module or not hasattr(self.gan_module, "generate_document"):
            messagebox.showerror("Error", "No GAN module selected or loaded.")
            return

        if not self.validate_constraints():
            return

        constraints = {name: entry.get().strip() for name, entry in self.constraint_entries.items()}
        try:
            output = self.gan_module.generate_document(constraints)
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, output)
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = GANApp(root)
    root.mainloop()
