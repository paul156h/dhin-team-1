# Root directory and model folder setup
# Ensures the app always looks for .pth models relative to this script.

import os, sys, subprocess, csv, io, torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(SCRIPT_DIR, "models")

# Wrapper for loading a saved Transformer model and generating text from it.
# This class abstracts away model reconstruction and character-level decoding.
# -------------------------------
# Loader
# -------------------------------
class TransformerGenerator:
    def __init__(self, model_path):
        # Load checkpoint and extract config/state_dict needed to rebuild the model.
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obj = torch.load(self.model_path, map_location=self.device)
        self.config = self.obj.get("config", {})
        self.state_dict = self.obj.get("state_dict", {})
        self.model = self._load_model()

    # Reconstructs the exact model architecture used during training.
    # The architecture is defined inline so the .pth file doesn't need the class definition.
    def _load_model(self):
        h = self.config.get("model_hparams", {})
        vocab = self.config.get("vocab", [])
        vocab_size = len(vocab)

        if vocab_size == 0:
            raise ValueError("Model checkpoint is missing vocab in config.")

        model_type = self.config.get("model_type", "legacy_transformer")

        class LegacyTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_layers, max_length):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, d_model))
                self.transformer = nn.Transformer(
                    d_model=d_model, nhead=nhead,
                    num_encoder_layers=num_layers,
                    num_decoder_layers=num_layers
                )
                self.fc_out = nn.Linear(d_model, vocab_size)
            def forward(self, src, tgt):
                src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
                tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
                out = self.transformer(src_emb.transpose(0,1), tgt_emb.transpose(0,1))
                return self.fc_out(out).transpose(0,1)

        class CharTransformerDecoder(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_layers, max_length, pad_id=0, dropout=0.1):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
                self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, d_model))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc_out = nn.Linear(d_model, vocab_size)
                self.pad_id = pad_id

            def forward(self, x):
                seq_len = x.size(1)
                emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                    diagonal=1,
                )
                pad_mask = x.eq(self.pad_id)
                out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=pad_mask)
                return self.fc_out(out)

        if model_type == "char_transformer_decoder":
            model = CharTransformerDecoder(
                vocab_size=vocab_size,
                d_model=h.get("d_model", 128),
                nhead=h.get("nhead", 4),
                num_layers=h.get("num_layers", 3),
                max_length=h.get("max_length", self.config.get("max_length", 256)),
                pad_id=h.get("pad_id", self.config.get("pad_id", 0)),
                dropout=h.get("dropout", 0.1),
            )
        else:
            model = LegacyTransformer(
                vocab_size,
                d_model=h.get("d_model", 64),
                nhead=h.get("nhead", 4),
                num_layers=h.get("num_layers", 2),
                max_length=h.get("max_length", 256),
            )

        if self.state_dict:
            model.load_state_dict(self.state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def get_headers(self):
        return self.config.get("headers") if self.config.get("has_headers") else None
    def get_binary_columns(self):
        return self.config.get("binary_columns", {})
    def generate_text(self, seed_text="", max_new_tokens=200):
        # Greedy character-level generation loop.
        # Continues until newline or max length is reached.
        if not self.config.get("vocab"):
            return ""
        device = self.device
        stoi = {ch: i for i, ch in enumerate(self.config["vocab"])}
        itos = {i: ch for i, ch in enumerate(self.config["vocab"])}
        max_length = self.config["max_length"]
        pad_id = int(self.config.get("pad_id", 0))
        model_type = self.config.get("model_type", "legacy_transformer")
        ids = [stoi.get(ch, 0) for ch in seed_text][:max_length] or [0]
        y = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        generated = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                if model_type == "char_transformer_decoder":
                    logits = self.model(y)
                else:
                    logits = self.model(y, y)
                if 0 <= pad_id < logits.size(-1):
                    logits[:, -1, pad_id] = float("-inf")
                next_id = torch.argmax(logits[:,-1,:], dim=-1)
            if int(next_id) == pad_id:
                break
            y = torch.cat([y, next_id.unsqueeze(1)], dim=1)
            generated.append(itos[int(next_id)])
            if generated[-1] == "\n" or y.size(1) >= max_length:
                break
        return seed_text + "".join(generated)

# Tkinter GUI wrapper for selecting models, adjusting constraints,
# generating synthetic CSV rows, and previewing/exporting results.
# -------------------------------
# Tkinter UI
# -------------------------------
class TransformerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Dataset Generator")
        self.root.geometry("950x800")
        self.models = self.find_models()
        self.model_instance = None
        self.constraint_entries = {}
        self.generated_csv_text = ""
        self.build_ui()

    def find_models(self):
        # Scan the models/ directory for available .pth model files.
        if not os.path.exists(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        return [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pth")]

    def build_ui(self):
        # Model selection
        ttk.Label(self.root, text="Available Models").pack(pady=5)
        self.model_list = tk.Listbox(self.root, height=6)
        self.model_list.pack(fill=tk.X, padx=10)
        self.model_list.bind("<<ListboxSelect>>", self.load_model)
        for m in self.models: self.model_list.insert(tk.END, m)
        ttk.Button(self.root, text="Manual Select", command=self.manual_select).pack(pady=5)

        # Constraints
        ttk.Label(self.root, text="Editable Parameters").pack(pady=5)
        self.constraints_frame = ttk.LabelFrame(self.root, text="Parameters")
        self.constraints_frame.pack(fill=tk.X, padx=10, pady=5)

        # Sample length
        sample_frame = ttk.Frame(self.root)
        sample_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(sample_frame, text="CSV Sample Length").pack(side=tk.LEFT, padx=5)
        self.sample_length_entry = ttk.Entry(sample_frame, width=8)
        self.sample_length_entry.insert(0,"5")
        self.sample_length_entry.pack(side=tk.LEFT, padx=5)

        # Generate button
        ttk.Button(self.root, text="Generate", command=self.generate_output).pack(pady=5)

        # Status
        ttk.Label(self.root, text="Status").pack(pady=5)
        self.status_box = tk.Text(self.root, height=6)
        self.status_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Output preview
        ttk.Label(self.root, text="CSV Preview").pack(pady=5)
        self.output_box = tk.Text(self.root, height=12)
        self.output_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bottom buttons
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        ttk.Button(bottom_frame, text="Show Table", command=self.show_table_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Export Output", command=self.convert_placeholder).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Manual Mode", command=self.switch_manual_mode).pack(side=tk.LEFT, padx=5)

    def manual_select(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model","*.pth")])
        if path:
            self.model_instance = TransformerGenerator(path)
            self.show_constraints()

    def load_model(self, event):
        sel = self.model_list.curselection()
        if not sel: return
        model_file = self.models[sel[0]]
        self.model_instance = TransformerGenerator(os.path.join(MODEL_FOLDER, model_file))
        self.show_constraints()

    def show_constraints(self):
        # Dynamically build sliders for binary columns detected during training.
        # These values are not yet applied to generation but reserved for future constraint logic.
        for w in self.constraints_frame.winfo_children(): w.destroy()
        self.constraint_entries.clear()
        if not self.model_instance: return
        binary_cols = self.model_instance.get_binary_columns()
        for col, values in binary_cols.items():
            row = ttk.Frame(self.constraints_frame); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{col} ({values[0]} vs {values[1]})").pack(side=tk.LEFT, padx=5)
            val_label = ttk.Label(row, text="50%"); val_label.pack(side=tk.RIGHT, padx=5)
            scale = ttk.Scale(row, from_=0,to=100,orient="horizontal"); scale.set(50)
            def update_label(event=None,lbl=val_label,sc=scale): lbl.config(text=f"{int(sc.get())}%")
            scale.bind("<Motion>", update_label); scale.bind("<ButtonRelease-1>", update_label)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.constraint_entries[col] = scale

    def validate_constraints(self):
        return {name: float(widget.get())/100.0 for name, widget in self.constraint_entries.items()}

    def generate_output(self):
        # Generate N synthetic CSV rows by repeatedly calling the model's text generator.
        if not self.model_instance:
            messagebox.showerror("No Model", "Please select a model first.")
            return
        constraints = self.validate_constraints()
        self.status_box.insert(tk.END, "Generating...\n")
        self.root.update_idletasks()
        try:
            num_rows = max(1, int(self.sample_length_entry.get()))
        except ValueError:
            num_rows = 1

        generated_lines = []
        for _ in range(num_rows):
            candidate = self.model_instance.generate_text(seed_text="", max_new_tokens=256).strip()
            # In a more advanced version, you could apply constraints here
            generated_lines.append(candidate)

        self.generated_csv_text = "\n".join(generated_lines)
        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, self.generated_csv_text)
        self.status_box.insert(tk.END, "Generation complete.\n")

    def show_table_window(self):
        # Parse generated CSV text and display it in a scrollable table view.
        text = self.generated_csv_text.strip()
        parsed_rows = [row for row in csv.reader(io.StringIO(text))] if text else [["(no data)"]]

        headers = self.model_instance.get_headers()
        if headers:
            data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []
        else:
            max_cols = max(len(row) for row in parsed_rows)
            headers = [f"Column {i+1}" for i in range(max_cols)]
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
        # Save the generated CSV text to a file in the script directory.
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
        # Launch an external script for manual generation mode.
        try:
            subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, "manualMode.py")])
            self.status_box.insert(tk.END, "Switched to Manual Mode.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch manual mode: {e}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = TransformerApp(root)
    root.mainloop()
