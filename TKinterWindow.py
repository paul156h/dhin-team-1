import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.simpledialog
import importlib.util
import os

GAN_FOLDER = "."  # Current folder

class GANApp:
    def is_valid_folder_name(self, name):
        # Windows forbidden characters: \ / : * ? " < > |
        invalid_chars = r'<>:"/\\|?*'
        return not any(char in name for char in invalid_chars)

    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Clinical Document Generator")
        self.root.geometry("600x500")

        self.gan_files = self.find_gan_files()
        self.selected_gan = None
        self.gan_module = None
        self.constraint_entries = {}

        self.attributions = [
            "MNIST dataset © Yann LeCun, Corinna Cortes, and Christopher J.C. Burges",
            "Synthetic data generation inspired by DCGAN architecture",
            "Model trained using PyTorch framework",
        ]

        self.build_gui()

    def find_gan_files(self):
        return [
            f for f in os.listdir(GAN_FOLDER)
            if f.endswith("GAN.py") or f.endswith("GAN.pth")
        ]

    def build_gui(self):
        ttk.Label(self.root, text="Select GAN Model or Script:").pack(pady=5)

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

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        attribution_button = ttk.Button(bottom_frame, text="View Attributions", command=self.show_attributions)
        attribution_button.pack(side=tk.LEFT)

        manual_load_button = ttk.Button(bottom_frame, text="Manually Load GAN", command=self.manual_load_gan)
        manual_load_button.pack(side=tk.LEFT, padx=5)

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

        if filename.endswith("GAN.py"):
            spec = importlib.util.spec_from_file_location("gan_module", path)
            if spec and spec.loader:
                self.gan_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.gan_module)
            else:
                messagebox.showerror("Load Error", f"Could not load Python module: {filename}")
                self.gan_module = None

        elif filename.endswith("GAN.pth"):
            self.gan_module = self.load_fallback_gan(path)

        else:
            messagebox.showerror("Unsupported File", f"{filename} is not a recognized GAN script or model.")
            self.gan_module = None

    def load_fallback_gan(self, model_path):
        import torch
        import numpy as np
        from PIL import Image

        class Generator(torch.nn.Module):
            def __init__(self, latent_dim=100, output_dim=784):
                super().__init__()
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, 256),
                    torch.nn.ReLU(True),
                    torch.nn.Linear(256, 512),
                    torch.nn.ReLU(True),
                    torch.nn.Linear(512, output_dim),
                    torch.nn.Tanh()
                )

            def forward(self, z):
                return self.model(z)

        class FallbackGAN:
            def __init__(self):
                self.latent_dim = 100
                self.generator = Generator(self.latent_dim)
                self.generator.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.generator.eval()

            def get_constraints(self):
                return [f"Digit {i} %" for i in range(10)]

            def generate_document(self, constraints, output_folder):
                ratios = [float(constraints.get(f"Digit {i} %", 0)) for i in range(10)]
                total = sum(ratios)
                if total == 0:
                    raise ValueError("At least one digit ratio must be greater than 0.")
                probabilities = [r / total for r in ratios]
                digit_labels = np.random.choice(range(10), size=100, p=probabilities)

                images = []
                for i, label in enumerate(digit_labels):
                    z = torch.randn(1, self.latent_dim)
                    img = self.generator(z).detach().numpy().reshape(28, 28)
                    img = ((img + 1) * 127.5).astype(np.uint8)
                    images.append((label, Image.fromarray(img)))

                grid = Image.new("L", (280, 280))
                for idx, (_, img) in enumerate(images[:100]):
                    x = (idx % 10) * 28
                    y = (idx // 10) * 28
                    grid.paste(img, (x, y))

                image_path = os.path.join(output_folder, "generated_digits.jpg")
                grid.save(image_path)

                counts = {str(i): digit_labels.tolist().count(i) for i in range(10)}
                summary = "Synthetic Digit Generation Summary:\n"
                for digit, count in counts.items():
                    summary += f"Digit {digit}: {count} samples\n"
                summary += f"\nImage saved as: {image_path}"
                return summary

        return FallbackGAN()

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
        total = 0

        for name, entry in self.constraint_entries.items():
            value = entry.get().strip()
            if "%" in name:
                try:
                    val = float(value)
                    if not (0 <= val <= 100):
                        raise ValueError
                    total += val
                except ValueError:
                    messagebox.showerror("Invalid Input", f"{name} must be a number between 0 and 100.")
                    valid = False

        if total > 100:
            messagebox.showerror("Invalid Input", "Total percentage across all constraints must not exceed 100%.")
            valid = False

        return valid
    #monke brain make the generate_document work after like 15 renditions
    def generate_document(self):
        if not self.gan_module or not hasattr(self.gan_module, "generate_document"):
            messagebox.showerror("Error", "No GAN module selected or loaded.")
            return

        if not self.validate_constraints():
            return

        constraints = {name: entry.get().strip() for name, entry in self.constraint_entries.items()}

        folder_path = filedialog.askdirectory(title="Select Destination Folder")
        if not folder_path:
            return

        subfolder_name = tkinter.simpledialog.askstring("Folder Name", "Enter name for output folder:")
        if not subfolder_name:
            return

        if not self.is_valid_folder_name(subfolder_name):
            messagebox.showerror("Invalid Folder Name", "Folder name contains invalid characters:\n\\ / : * ? \" < > |")
            return


        output_folder = os.path.join(folder_path, subfolder_name)
        os.makedirs(output_folder, exist_ok=True)

        try:
            output = self.gan_module.generate_document(constraints, output_folder)

            txt_path = os.path.join(output_folder, "constraints.txt")
            with open(txt_path, "w") as f:
                for key, value in constraints.items():
                    f.write(f"{key}: {value}\n")

            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, f"{output}\n\nSaved to: {output_folder}")
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))

    def show_attributions(self):
        window = tk.Toplevel(self.root)
        window.title("Dataset Attributions")
        window.geometry("500x300")

        frame = ttk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for attribution in self.attributions:
            label = ttk.Label(frame, text=attribution, wraplength=480, justify=tk.LEFT)
            label.pack(anchor="w", pady=2)

    def manual_load_gan(self):
        file_path = filedialog.askopenfilename(
            title="Select GAN Python File",
            filetypes=[("Python Files", "*.py")]
        )
        if not file_path:
            return

        try:
            spec = importlib.util.spec_from_file_location("gan_module", file_path)
            self.gan_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.gan_module)
            self.selected_gan = os.path.basename(file_path)
            self.show_constraints()
            messagebox.showinfo("Success", f"Loaded GAN module: {self.selected_gan}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load GAN module:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GANApp(root)
    root.mainloop()
