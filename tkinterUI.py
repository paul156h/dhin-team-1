import os
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import csv, io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(SCRIPT_DIR, "models")

# -------------------------------
# Generic Loader
# -------------------------------
class TransformerGenerator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.obj = torch.load(self.model_path, map_location="cpu")
        self.config = self.obj.get("config", {})
        self.state_dict = self.obj.get("state_dict", {})
        self.model = self._load_model()

    def _load_model(self):
        cls = self.config.get("model_class", "SimpleTransformer")
        h = self.config.get("model_hparams", {})
        vocab = self.config.get("vocab", [])
        vocab_size = len(vocab) if vocab else int(self.config.get("training_info", {}).get("vocab_size", 0))

        if cls == "SimpleTransformer":
            # Import here to avoid circular issues
            import torch.nn as nn
            class SimpleTransformer(nn.Module):
                def __init__(self, vocab_size, d_model, nhead, num_layers, max_length):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, d_model))
                    self.transformer = nn.Transformer(
                        d_model=d_model,
                        nhead=nhead,
                        num_encoder_layers=num_layers,
                        num_decoder_layers=num_layers
                    )
                    self.fc_out = nn.Linear(d_model, vocab_size)

                def forward(self, src, tgt):
                    src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
                    tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
                    out = self.transformer(src_emb.transpose(0,1), tgt_emb.transpose(0,1))
                    return self.fc_out(out).transpose(0,1)

            model = SimpleTransformer(
                vocab_size,
                d_model=h.get("d_model", 128),
                nhead=h.get("nhead", 8),
                num_layers=h.get("num_layers", 4),
                max_length=h.get("max_length", 128)
            )
        else:
            raise RuntimeError(f"Unknown model class {cls}")

        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def get_constraints(self):
        return self.config.get("parameters", {})

    def get_training_info(self):
        return self.config.get("training_info", {})

    def generate_text(self, seed_text="", max_new_tokens=200):
        if not self.config.get("vocab"):
            return ""

        device = next(self.model.parameters()).device
        stoi = {ch: i for i, ch in enumerate(self.config["vocab"])}
        itos = {i: ch for i, ch in enumerate(self.config["vocab"])}
        # Encode seed
        ids = [stoi.get(ch, 0) for ch in seed_text][:self.config["model_hparams"]["max_length"]]
        if not ids:
            ids = [0]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        y = x.clone()
        generated = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(x, y)  # [B, T, vocab]
                next_logits = logits[:, -1, :]  # last step
                next_id = torch.argmax(next_logits, dim=-1)  # greedy decode
            y = torch.cat([y, next_id.unsqueeze(1)], dim=1)
            generated.append(itos[int(next_id)])
            # stop at newline (end of row)
            if generated[-1] == "\n":
                break
            if y.size(1) >= self.config["model_hparams"]["max_length"]:
                break

        return seed_text + "".join(generated)



# -------------------------------
# Tkinter UI
# -------------------------------
class TransformerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transformer Document Generator")
        self.root.geometry("950x800")
        self.models = self.find_models()
        self.model_instance = None
        self.constraint_entries = {}
        self.generated_csv_text = ""
        self.build_ui()

    def find_models(self):
        if not os.path.exists(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        return [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pth")]

    def build_ui(self):
        # Model selection
        ttk.Label(self.root, text="Available Models").pack(pady=5)
        self.model_list = tk.Listbox(self.root, height=6)
        self.model_list.pack(fill=tk.X, padx=10)
        self.model_list.bind("<<ListboxSelect>>", self.load_model)
        for model in self.models:
            self.model_list.insert(tk.END, model)
        ttk.Button(self.root, text="Manual Select", command=self.manual_select).pack(pady=5)

        # Parameters
        ttk.Label(self.root, text="Editable Parameters").pack(pady=5)
        self.constraints_frame = ttk.LabelFrame(self.root, text="Parameters")
        self.constraints_frame.pack(fill=tk.X, padx=10, pady=5)

        # Sample length
        sample_frame = ttk.Frame(self.root)
        sample_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(sample_frame, text="CSV Sample Length").pack(side=tk.LEFT, padx=5)
        self.sample_length_entry = ttk.Entry(sample_frame, width=8)
        self.sample_length_entry.insert(0, "5")
        self.sample_length_entry.pack(side=tk.LEFT, padx=5)

        # Training info
        ttk.Label(self.root, text="Training Info").pack(pady=5)
        self.training_info_box = tk.Text(self.root, height=6, state="disabled")
        self.training_info_box.pack(fill=tk.X, padx=10, pady=5)

        # Generate button
        ttk.Button(self.root, text="Generate", command=self.generate_output).pack(pady=5)

        # Status window
        ttk.Label(self.root, text="Status").pack(pady=5)
        self.status_box = tk.Text(self.root, height=6)
        self.status_box.pack(fill=tk.X, padx=10, pady=5)

        # Output preview
        ttk.Label(self.root, text="CSV Preview").pack(pady=5)
        self.output_box = tk.Text(self.root, height=12)
        self.output_box.pack(fill=tk.BOTH, padx=10, pady=10)

        # Table button
        ttk.Button(self.root, text="Show Table", command=self.show_table_window).pack(pady=5)

        # Conversion
        ttk.Button(self.root, text="Convert to Document", command=self.convert_placeholder).pack(pady=5)

        # Manual Mode
        ttk.Button(self.root, text="Switch to Manual Mode", command=self.switch_manual_mode).pack(pady=5)

    def manual_select(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            self.model_instance = TransformerGenerator(path)
            self.show_constraints()
            self.show_training_info()

    def load_model(self, event):
        selection = self.model_list.curselection()
        if not selection: return
        model_file = self.models[selection[0]]
        self.model_instance = TransformerGenerator(os.path.join(MODEL_FOLDER, model_file))
        self.show_constraints()
        self.show_training_info()

    def show_constraints(self):
        for widget in self.constraints_frame.winfo_children():
            widget.destroy()
        self.constraint_entries.clear()
        if not self.model_instance: return
        constraints = self.model_instance.get_constraints()
        for name, default in constraints.items():
            row = ttk.Frame(self.constraints_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=name).pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(row)
            entry.insert(0, str(default))
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.constraint_entries[name] = entry

    def show_training_info(self):
        self.training_info_box.config(state="normal")
        self.training_info_box.delete("1.0", tk.END)
        info = self.model_instance.get_training_info()
        if info:
            for k, v in info.items():
                self.training_info_box.insert(tk.END, f"{k}: {v}\n")
        else:
            self.training_info_box.insert(tk.END, "No training info available.\n")
        self.training_info_box.config(state="disabled")

    def validate_constraints(self):
        constraints = {}
        for name, widget in self.constraint_entries.items():
            val = widget.get().strip()
            if not val:
                messagebox.showerror("Invalid Input", f"{name} cannot be empty.")
                return None
            constraints[name] = val
        return constraints

    def generate_output(self):
        if not self.model_instance:
            messagebox.showerror("No Model", "Please select a model first.")
            return
        constraints = self.validate_constraints()
        if constraints is None: return
        self.status_box.insert(tk.END, "Generating...\n")
        self.root.update_idletasks()
        try:
            num_rows = max(1, int(self.sample_length_entry.get()))
        except ValueError:
            num_rows = 1
        generated_lines = []
        for _ in range(num_rows):
            line = self.model_instance.generate_text(seed_text="", max_new_tokens=256)
            generated_lines.append(line.strip())
        self.generated_csv_text = "\n".join(generated_lines)
        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, self.generated_csv_text)
        self.status_box.insert(tk.END, "Generation complete.\n")

    def show_table_window(self):
        if not self.generated_csv_text.strip():
            messagebox.showinfo("No Data", "No generated CSV to display yet.")
            return

        reader = csv.reader(io.StringIO(self.generated_csv_text))
        parsed_rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not parsed_rows:
            messagebox.showinfo("No Data", "Generated CSV appears empty.")
            return

        # Decide headers: if first row looks like headers, use them
        first_row = parsed_rows[0]
        has_headers = all(cell.strip() for cell in first_row)

        if has_headers:
            headers = first_row
            data_rows = parsed_rows[1:]
        else:
            headers = [f"Column {i+1}" for i in range(len(first_row))]
            data_rows = parsed_rows

        table_win = tk.Toplevel(self.root)
        table_win.title("Generated CSV Table")
        table_win.geometry("800x500")

        tree = ttk.Treeview(table_win, columns=headers, show="headings")
        for h in headers:
            tree.heading(h, text=h)
            tree.column(h, width=max(120, int(700 / max(1, len(headers)))))

        for row in data_rows:
            values = (row + [""] * (len(headers) - len(row)))[:len(headers)]
            tree.insert("", tk.END, values=values)

        vsb = ttk.Scrollbar(table_win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(table_win, text="Close", command=table_win.destroy).pack(pady=5)

    def convert_placeholder(self):
        # Save the generated CSV to disk for sanity check
        if not self.generated_csv_text.strip():
            messagebox.showinfo("No Data", "No generated CSV to save.")
            return
        filename = os.path.join(SCRIPT_DIR, "generated_output.csv")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.generated_csv_text)
            self.status_box.insert(tk.END, f"CSV saved to {filename}\n")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save CSV: {e}")

    def switch_manual_mode(self):
        # Launch manualMode.py script
        try:
            subprocess.Popen(["python", os.path.join(SCRIPT_DIR, "manualMode.py")])
            self.status_box.insert(tk.END, "Switched to Manual Mode.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch manual mode: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TransformerApp(root)
    root.mainloop()
