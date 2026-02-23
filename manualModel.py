import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import random
import uuid
import datetime
import json
import os

# --- Scrollable Frame Helper ---
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# --- Main App ---
class SyntheticDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Data Generator")
        self.root.geometry("900x700")

        self.columns = {}       # {name: {"type": type, "values": [], "rules": []}}
        self.associations = []  # list of dicts
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        main_frame = ScrollableFrame(root)
        main_frame.pack(fill="both", expand=True)
        self.sf = main_frame.scrollable_frame

        self.build_ui()

    def build_ui(self):
        # Column Section
        col_frame = ttk.LabelFrame(self.sf, text="Columns")
        col_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(col_frame, text="Column Name").grid(row=0, column=0)
        self.col_name_entry = ttk.Entry(col_frame)
        self.col_name_entry.grid(row=0, column=1)

        ttk.Label(col_frame, text="Type").grid(row=0, column=2)
        self.col_type = ttk.Combobox(col_frame, values=["Text","Value","Time","Date","Code"])
        self.col_type.grid(row=0, column=3)

        ttk.Button(col_frame, text="Add Column", command=self.add_column).grid(row=0, column=4)

        self.col_list = tk.Listbox(col_frame, height=5)
        self.col_list.grid(row=1, column=0, columnspan=5, sticky="ew")

        # Values/Rules Section
        val_frame = ttk.LabelFrame(self.sf, text="Values and Rules")
        val_frame.pack(fill="x", padx=10, pady=5)

        self.val_entry = ttk.Entry(val_frame)
        self.val_entry.pack(fill="x", padx=5, pady=2)
        ttk.Button(val_frame, text="Add Value/Rule", command=self.add_value).pack(pady=2)

        self.val_list = tk.Listbox(val_frame, height=5)
        self.val_list.pack(fill="x")

        # Associations Section
        assoc_frame = ttk.LabelFrame(self.sf, text="Associations")
        assoc_frame.pack(fill="x", padx=10, pady=5)

        self.assoc_list = tk.Listbox(assoc_frame, height=5)
        self.assoc_list.pack(fill="x")

        ttk.Button(assoc_frame, text="Add Pattern", command=self.save_assoc).pack(side="left", padx=5)
        ttk.Button(assoc_frame, text="Remove Pattern", command=self.remove_assoc).pack(side="left", padx=5)
        ttk.Button(assoc_frame, text="Import JSON", command=self.import_patterns_json).pack(side="left", padx=5)
        ttk.Button(assoc_frame, text="Export JSON", command=self.export_patterns_json).pack(side="left", padx=5)
        ttk.Button(assoc_frame, text="Edit", command=self.edit_assoc).pack(side="left", padx=5)
        ttk.Button(assoc_frame, text="Use as Conditional", command=self.use_conditional).pack(side="left", padx=5)
        self.invert_var = tk.BooleanVar()
        ttk.Checkbutton(assoc_frame, text="Inverse (not exists)", variable=self.invert_var).pack(side="left", padx=5)

        # Generate Section
        gen_frame = ttk.LabelFrame(self.sf, text="Generate Data")
        gen_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(gen_frame, text="Number of Entries").pack(anchor="w")
        self.num_entries = ttk.Entry(gen_frame)
        self.num_entries.pack(anchor="w")

        ttk.Button(gen_frame, text="Generate", command=self.generate_data).pack(pady=5)

        # Output Table
        out_frame = ttk.LabelFrame(self.sf, text="Resulting Table")
        out_frame.pack(fill="both", padx=10, pady=5, expand=True)

        self.tree = ttk.Treeview(out_frame, show="headings")
        self.tree.pack(fill="both", expand=True)

        # Export Button
        ttk.Button(self.sf, text="Export to CSV", command=self.export_csv).pack(pady=10)

    # --- Column Logic ---
    def add_column(self):
        name = self.col_name_entry.get()
        ctype = self.col_type.get()
        if not name or not ctype:
            return
        self.columns[name] = {"type": ctype, "values": [], "rules": []}
        self.col_list.insert(tk.END, f"{name} ({ctype})")
        self.update_tree_columns()

    def add_value(self):
        sel = self.col_list.curselection()
        if not sel: 
            return
        col_name = list(self.columns.keys())[sel[0]]
        val = self.val_entry.get()
        if val:
            self.columns[col_name]["values"].append(val)
            self.val_list.insert(tk.END, f"{col_name}: {val}")

    # --- Associations Logic ---
    def save_assoc(self):
        sel = self.val_list.curselection()
        if not sel: 
            return
        values = [self.val_list.get(i) for i in sel]
        assoc = {"conditions": values, "outcomes": [], "inverse": self.invert_var.get()}
        self.associations.append(assoc)
        self.refresh_assoc_list()

    def remove_assoc(self):
        sel = self.assoc_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.associations):
            self.associations.pop(idx)
            self.refresh_assoc_list()

    def refresh_assoc_list(self):
        self.assoc_list.delete(0, tk.END)
        for assoc in self.associations:
            conditions = assoc.get("conditions", [])
            inverse = " [inverse]" if assoc.get("inverse") else ""
            preview = ", ".join(conditions)
            if len(preview) > 90:
                preview = preview[:87] + "..."
            self.assoc_list.insert(tk.END, f"{preview}{inverse}")

    def _normalize_pattern(self, raw):
        if not isinstance(raw, dict):
            return None
        conditions = raw.get("conditions", [])
        outcomes = raw.get("outcomes", [])
        inverse = bool(raw.get("inverse", False))
        if not isinstance(conditions, list):
            return None
        if not isinstance(outcomes, list):
            outcomes = []
        return {
            "conditions": [str(x) for x in conditions],
            "outcomes": [str(x) for x in outcomes],
            "inverse": inverse,
        }

    def import_patterns_json(self):
        default_file = os.path.join(self.base_dir, "manualPatterns.json")
        file = filedialog.askopenfilename(
            title="Import Patterns JSON",
            initialdir=self.base_dir,
            initialfile=os.path.basename(default_file),
            filetypes=[("JSON files", "*.json")],
        )
        if not file:
            return
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                self.associations = []
                self.refresh_assoc_list()
                messagebox.showinfo("Import", "Loaded 0 patterns (empty file).")
                return

            data = json.loads(content)
            raw_patterns = data.get("patterns", []) if isinstance(data, dict) else data
            if not isinstance(raw_patterns, list):
                raise ValueError("JSON must be a list or an object with a 'patterns' list.")

            imported = []
            for item in raw_patterns:
                normalized = self._normalize_pattern(item)
                if normalized is not None:
                    imported.append(normalized)

            self.associations = imported
            self.refresh_assoc_list()
            messagebox.showinfo("Import", f"Imported {len(imported)} patterns.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import patterns: {e}")

    def export_patterns_json(self):
        default_file = os.path.join(self.base_dir, "manualPatterns.json")
        file = filedialog.asksaveasfilename(
            title="Export Patterns JSON",
            initialdir=self.base_dir,
            initialfile=os.path.basename(default_file),
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not file:
            return
        try:
            payload = {"patterns": self.associations}
            with open(file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            messagebox.showinfo("Export", f"Exported {len(self.associations)} patterns to {file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export patterns: {e}")

    def edit_assoc(self):
        messagebox.showinfo("Edit", "Editing associations not yet implemented")

    def use_conditional(self):
        messagebox.showinfo("Conditional", "Conditional logic placeholder")

    # --- Data Generation ---
    def generate_data(self):
        try:
            n = int(self.num_entries.get())
        except:
            n = 10
        rows = []
        for _ in range(n):
            row = {}
            for col, meta in self.columns.items():
                if meta["type"] == "Text":
                    row[col] = random.choice(meta["values"]) if meta["values"] else ""
                elif meta["type"] == "Value":
                    row[col] = random.randint(0,100)
                elif meta["type"] == "Time":
                    row[col] = f"{random.randint(0,23)}:{random.randint(0,59):02d}"
                elif meta["type"] == "Date":
                    row[col] = str(datetime.date.today() + datetime.timedelta(days=random.randint(0,365)))
                elif meta["type"] == "Code":
                    row[col] = str(uuid.uuid4())[:8]
            rows.append(row)
        self.display_rows(rows)

    def display_rows(self, rows):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.columns.keys())
        for col in self.columns.keys():
            self.tree.heading(col, text=col)
        for row in rows:
            self.tree.insert("", tk.END, values=[row[c] for c in self.columns.keys()])

    def update_tree_columns(self):
        self.tree["columns"] = list(self.columns.keys())
        for col in self.columns.keys():
            self.tree.heading(col, text=col)

    # --- Export ---
    def export_csv(self):
        file = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files","*.csv")])
        if not file: 
            return
        try:
            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns.keys())
                for item in self.tree.get_children():
                    writer.writerow(self.tree.item(item)["values"])
            messagebox.showinfo("Export", f"Data exported to {file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

def main():
    root = tk.Tk()
    app = SyntheticDataApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
