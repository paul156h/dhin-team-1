import csv
import datetime as dt
import json
import os
import random
import re
import string
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def pseudocode_to_regex(text: str) -> str:
	phrase = (text or "").strip().lower()
	if not phrase:
		return ""

	match = re.search(r"(\d+)\s*letters?.*(\d+)\s*digits?", phrase)
	if match:
		letters = int(match.group(1))
		digits = int(match.group(2))
		return f"[A-Za-z]{{{letters}}}[0-9]{{{digits}}}"

	match = re.search(r"ending\s+in\s+([a-z0-9_\- ]+)", phrase)
	if match:
		suffix = re.escape(match.group(1).strip())
		return f".*{suffix}$"

	if "hh:mm" in phrase or ("time" in phrase and "format" in phrase):
		return r"^[0-9]{2}:[0-9]{2}$"

	return ""


def generate_from_regex(pattern: str) -> str:
	p = (pattern or "").strip()
	if not p:
		return ""

	m = re.fullmatch(r"\[A-Za-z\]\{(\d+)\}\[0-9\]\{(\d+)\}", p)
	if m:
		letters = "".join(random.choice(string.ascii_letters) for _ in range(int(m.group(1))))
		digits = "".join(random.choice(string.digits) for _ in range(int(m.group(2))))
		return f"{letters}{digits}"

	if p == r"^[0-9]{2}:[0-9]{2}$":
		return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"

	m = re.fullmatch(r"\.\*(.+)\$", p)
	if m:
		prefix = "".join(random.choice(string.ascii_letters) for _ in range(random.randint(3, 8)))
		suffix = m.group(1)
		suffix = suffix.replace("\\", "")
		return f"{prefix}{suffix}"

	return "X"


class ManualModeApp:
	COLUMN_TYPES = ["Nominal", "Ordinal", "Boolean", "Regex", "Numeric", "Date", "Time"]
	OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith", "regex"]

	def __init__(self, root: tk.Tk):
		self.root = root
		self.root.title("Manual Synthetic Data Rule Builder")
		self.root.geometry("1100x760")

		self.columns = []
		self.rules = []
		self.generated_rows = []

		self._build_ui()

	def _build_ui(self):
		main = ttk.Frame(self.root, padding=8)
		main.pack(fill=tk.BOTH, expand=True)

		left = ttk.LabelFrame(main, text="Columns")
		right = ttk.LabelFrame(main, text="Rules")
		left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
		right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

		self._build_columns_ui(left)
		self._build_rules_ui(right)
		self._build_bottom_ui(main)

	def _build_columns_ui(self, parent):
		list_frame = ttk.Frame(parent)
		list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

		self.column_list = tk.Listbox(list_frame, height=12)
		self.column_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		self.column_list.bind("<<ListboxSelect>>", self._on_column_select)

		c_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.column_list.yview)
		c_scroll.pack(side=tk.RIGHT, fill=tk.Y)
		self.column_list.config(yscrollcommand=c_scroll.set)

		form = ttk.Frame(parent)
		form.pack(fill=tk.X, padx=6, pady=(0, 6))

		ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w")
		self.col_name = ttk.Entry(form)
		self.col_name.grid(row=0, column=1, sticky="ew", padx=4, pady=2)

		ttk.Label(form, text="Type").grid(row=1, column=0, sticky="w")
		self.col_type = ttk.Combobox(form, values=self.COLUMN_TYPES, state="readonly")
		self.col_type.set(self.COLUMN_TYPES[0])
		self.col_type.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

		ttk.Label(form, text="Config").grid(row=2, column=0, sticky="nw")
		self.col_config = tk.Text(form, height=5)
		self.col_config.grid(row=2, column=1, sticky="ew", padx=4, pady=2)

		helper = ttk.Frame(form)
		helper.grid(row=3, column=1, sticky="ew", padx=4, pady=2)
		ttk.Label(helper, text="Regex helper").pack(anchor="w")
		self.regex_help_entry = ttk.Entry(helper)
		self.regex_help_entry.pack(fill=tk.X, pady=(2, 2))
		ttk.Button(helper, text="Convert Pseudocode → Regex", command=self._convert_regex_helper).pack(anchor="e")

		btn_row = ttk.Frame(parent)
		btn_row.pack(fill=tk.X, padx=6, pady=(0, 6))
		ttk.Button(btn_row, text="Add Column", command=self._add_column).pack(side=tk.LEFT)
		ttk.Button(btn_row, text="Update Column", command=self._update_column).pack(side=tk.LEFT, padx=4)
		ttk.Button(btn_row, text="Remove Column", command=self._remove_column).pack(side=tk.LEFT)

		form.columnconfigure(1, weight=1)

	def _build_rules_ui(self, parent):
		list_frame = ttk.Frame(parent)
		list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

		self.rule_list = tk.Listbox(list_frame, height=12)
		self.rule_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		self.rule_list.bind("<<ListboxSelect>>", self._on_rule_select)

		r_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.rule_list.yview)
		r_scroll.pack(side=tk.RIGHT, fill=tk.Y)
		self.rule_list.config(yscrollcommand=r_scroll.set)

		form = ttk.Frame(parent)
		form.pack(fill=tk.X, padx=6, pady=(0, 6))

		ttk.Label(form, text="Rule name").grid(row=0, column=0, sticky="w")
		self.rule_name = ttk.Entry(form)
		self.rule_name.grid(row=0, column=1, sticky="ew", padx=4, pady=2)

		ttk.Label(form, text="Conditions").grid(row=1, column=0, sticky="nw")
		self.rule_conditions = tk.Text(form, height=5)
		self.rule_conditions.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

		ttk.Label(
			form,
			text="Format: Column|operator|value (one per line). Example: Day|==|Tuesday",
		).grid(row=2, column=1, sticky="w", padx=4)

		ttk.Label(form, text="Condition logic").grid(row=3, column=0, sticky="w")
		self.rule_logic = ttk.Combobox(form, values=["AND", "OR"], state="readonly")
		self.rule_logic.set("AND")
		self.rule_logic.grid(row=3, column=1, sticky="w", padx=4, pady=2)

		ttk.Label(form, text="Action column").grid(row=4, column=0, sticky="w")
		self.rule_target_col = ttk.Entry(form)
		self.rule_target_col.grid(row=4, column=1, sticky="ew", padx=4, pady=2)

		ttk.Label(form, text="Action value").grid(row=5, column=0, sticky="w")
		self.rule_target_val = ttk.Entry(form)
		self.rule_target_val.grid(row=5, column=1, sticky="ew", padx=4, pady=2)

		self.rule_stop_var = tk.BooleanVar(value=False)
		ttk.Checkbutton(form, text="Stop after this rule matches", variable=self.rule_stop_var).grid(
			row=6, column=1, sticky="w", padx=4
		)

		btn_row = ttk.Frame(parent)
		btn_row.pack(fill=tk.X, padx=6, pady=(0, 6))
		ttk.Button(btn_row, text="Add Rule", command=self._add_rule).pack(side=tk.LEFT)
		ttk.Button(btn_row, text="Update Rule", command=self._update_rule).pack(side=tk.LEFT, padx=4)
		ttk.Button(btn_row, text="Remove Rule", command=self._remove_rule).pack(side=tk.LEFT)
		ttk.Button(btn_row, text="Move Up", command=lambda: self._move_rule(-1)).pack(side=tk.LEFT, padx=8)
		ttk.Button(btn_row, text="Move Down", command=lambda: self._move_rule(1)).pack(side=tk.LEFT)

		form.columnconfigure(1, weight=1)

	def _build_bottom_ui(self, parent):
		controls = ttk.LabelFrame(parent, text="Generate / Save")
		controls.pack(fill=tk.X, pady=8)

		top = ttk.Frame(controls)
		top.pack(fill=tk.X, padx=6, pady=6)

		ttk.Label(top, text="Rows").pack(side=tk.LEFT)
		self.row_count = ttk.Entry(top, width=10)
		self.row_count.insert(0, "100")
		self.row_count.pack(side=tk.LEFT, padx=(4, 12))

		ttk.Button(top, text="Generate", command=self._generate).pack(side=tk.LEFT)
		ttk.Button(top, text="Export CSV", command=self._export_csv).pack(side=tk.LEFT, padx=4)
		ttk.Button(top, text="Save Schema", command=self._save_schema).pack(side=tk.LEFT, padx=4)
		ttk.Button(top, text="Load Schema", command=self._load_schema).pack(side=tk.LEFT, padx=4)

		self.status = tk.Text(controls, height=6)
		self.status.pack(fill=tk.X, padx=6, pady=(0, 6))

		preview_frame = ttk.Frame(parent)
		preview_frame.pack(fill=tk.BOTH, expand=True)

		self.preview = ttk.Treeview(preview_frame, show="headings")
		self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		p_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview.yview)
		p_scroll.pack(side=tk.RIGHT, fill=tk.Y)
		self.preview.config(yscrollcommand=p_scroll.set)

	def _convert_regex_helper(self):
		output = pseudocode_to_regex(self.regex_help_entry.get())
		if not output:
			messagebox.showinfo("Regex Helper", "Could not map that phrase to a regex pattern.")
			return
		self.col_type.set("Regex")
		self.col_config.delete("1.0", tk.END)
		self.col_config.insert(tk.END, output)

	def _parse_column_form(self):
		name = self.col_name.get().strip()
		ctype = self.col_type.get().strip()
		config = self.col_config.get("1.0", tk.END).strip()
		if not name:
			raise ValueError("Column name is required.")
		if ctype not in self.COLUMN_TYPES:
			raise ValueError("Column type is invalid.")

		return {"name": name, "type": ctype, "config": config}

	def _add_column(self):
		try:
			col = self._parse_column_form()
		except ValueError as err:
			messagebox.showerror("Column", str(err))
			return
		if any(existing["name"] == col["name"] for existing in self.columns):
			messagebox.showerror("Column", "Column name already exists.")
			return
		self.columns.append(col)
		self._refresh_column_list()
		self._log(f"Added column: {col['name']}")

	def _update_column(self):
		selected = self.column_list.curselection()
		if not selected:
			return
		idx = selected[0]
		try:
			col = self._parse_column_form()
		except ValueError as err:
			messagebox.showerror("Column", str(err))
			return

		duplicate = [i for i, x in enumerate(self.columns) if x["name"] == col["name"] and i != idx]
		if duplicate:
			messagebox.showerror("Column", "Column name already exists.")
			return

		old_name = self.columns[idx]["name"]
		self.columns[idx] = col
		if old_name != col["name"]:
			for rule in self.rules:
				for condition in rule["conditions"]:
					if condition["column"] == old_name:
						condition["column"] = col["name"]
				if rule["target_column"] == old_name:
					rule["target_column"] = col["name"]
			self._refresh_rule_list()

		self._refresh_column_list(selected_index=idx)
		self._log(f"Updated column: {col['name']}")

	def _remove_column(self):
		selected = self.column_list.curselection()
		if not selected:
			return
		idx = selected[0]
		name = self.columns[idx]["name"]
		self.columns.pop(idx)
		self.rules = [
			rule
			for rule in self.rules
			if rule["target_column"] != name
			and all(condition["column"] != name for condition in rule["conditions"])
		]
		self._refresh_column_list()
		self._refresh_rule_list()
		self._log(f"Removed column: {name}")

	def _on_column_select(self, _event=None):
		selected = self.column_list.curselection()
		if not selected:
			return
		col = self.columns[selected[0]]
		self.col_name.delete(0, tk.END)
		self.col_name.insert(0, col["name"])
		self.col_type.set(col["type"])
		self.col_config.delete("1.0", tk.END)
		self.col_config.insert(tk.END, col["config"])

	def _refresh_column_list(self, selected_index=None):
		self.column_list.delete(0, tk.END)
		for col in self.columns:
			short_cfg = col["config"].replace("\n", " ")
			if len(short_cfg) > 40:
				short_cfg = short_cfg[:37] + "..."
			label = f"{col['name']} [{col['type']}]"
			if short_cfg:
				label += f" - {short_cfg}"
			self.column_list.insert(tk.END, label)

		if selected_index is not None and self.columns:
			selected_index = max(0, min(selected_index, len(self.columns) - 1))
			self.column_list.selection_set(selected_index)

	def _parse_conditions(self, text: str):
		conditions = []
		lines = [line.strip() for line in text.splitlines() if line.strip()]
		for line in lines:
			parts = [part.strip() for part in line.split("|", 2)]
			if len(parts) != 3:
				raise ValueError(f"Invalid condition format: {line}")
			column, operator, value = parts
			if operator not in self.OPERATORS:
				raise ValueError(f"Invalid operator: {operator}")
			if not any(c["name"] == column for c in self.columns):
				raise ValueError(f"Condition column not found: {column}")
			conditions.append({"column": column, "operator": operator, "value": value})
		return conditions

	def _parse_rule_form(self):
		name = self.rule_name.get().strip() or f"Rule {len(self.rules) + 1}"
		logic = self.rule_logic.get().strip() or "AND"
		target_col = self.rule_target_col.get().strip()
		target_val = self.rule_target_val.get().strip()
		conditions_text = self.rule_conditions.get("1.0", tk.END)

		if not target_col:
			raise ValueError("Action column is required.")
		if not any(c["name"] == target_col for c in self.columns):
			raise ValueError("Action column does not exist.")

		conditions = self._parse_conditions(conditions_text)
		if not conditions:
			raise ValueError("At least one condition is required.")

		return {
			"name": name,
			"logic": "OR" if logic.upper() == "OR" else "AND",
			"conditions": conditions,
			"target_column": target_col,
			"target_value": target_val,
			"stop_on_match": bool(self.rule_stop_var.get()),
		}

	def _add_rule(self):
		try:
			rule = self._parse_rule_form()
		except ValueError as err:
			messagebox.showerror("Rule", str(err))
			return
		self.rules.append(rule)
		self._refresh_rule_list()
		self._log(f"Added rule: {rule['name']}")

	def _update_rule(self):
		selected = self.rule_list.curselection()
		if not selected:
			return
		idx = selected[0]
		try:
			rule = self._parse_rule_form()
		except ValueError as err:
			messagebox.showerror("Rule", str(err))
			return
		self.rules[idx] = rule
		self._refresh_rule_list(selected_index=idx)
		self._log(f"Updated rule: {rule['name']}")

	def _remove_rule(self):
		selected = self.rule_list.curselection()
		if not selected:
			return
		idx = selected[0]
		name = self.rules[idx]["name"]
		self.rules.pop(idx)
		self._refresh_rule_list()
		self._log(f"Removed rule: {name}")

	def _move_rule(self, delta: int):
		selected = self.rule_list.curselection()
		if not selected:
			return
		idx = selected[0]
		new_idx = idx + delta
		if new_idx < 0 or new_idx >= len(self.rules):
			return
		self.rules[idx], self.rules[new_idx] = self.rules[new_idx], self.rules[idx]
		self._refresh_rule_list(selected_index=new_idx)
		self._log("Reordered rules.")

	def _on_rule_select(self, _event=None):
		selected = self.rule_list.curselection()
		if not selected:
			return
		rule = self.rules[selected[0]]
		self.rule_name.delete(0, tk.END)
		self.rule_name.insert(0, rule["name"])
		self.rule_logic.set(rule["logic"])
		self.rule_target_col.delete(0, tk.END)
		self.rule_target_col.insert(0, rule["target_column"])
		self.rule_target_val.delete(0, tk.END)
		self.rule_target_val.insert(0, rule["target_value"])
		self.rule_stop_var.set(rule.get("stop_on_match", False))

		self.rule_conditions.delete("1.0", tk.END)
		for condition in rule["conditions"]:
			self.rule_conditions.insert(
				tk.END,
				f"{condition['column']}|{condition['operator']}|{condition['value']}\n",
			)

	def _refresh_rule_list(self, selected_index=None):
		self.rule_list.delete(0, tk.END)
		for i, rule in enumerate(self.rules, start=1):
			stop_text = " stop" if rule.get("stop_on_match") else ""
			self.rule_list.insert(
				tk.END,
				f"{i}. {rule['name']} ({rule['logic']}, {len(rule['conditions'])} conditions,{stop_text.strip() or ' continue'})",
			)

		if selected_index is not None and self.rules:
			selected_index = max(0, min(selected_index, len(self.rules) - 1))
			self.rule_list.selection_set(selected_index)

	def _column_value(self, column):
		ctype = column["type"]
		cfg = column.get("config", "")

		if ctype in ("Nominal", "Ordinal"):
			options = [x.strip() for x in cfg.split(",") if x.strip()]
			if not options:
				return ""
			return random.choice(options)

		if ctype == "Boolean":
			options = [x.strip() for x in cfg.split(",") if x.strip()]
			if len(options) >= 2:
				return random.choice(options[:2])
			return random.choice(["True", "False"])

		if ctype == "Regex":
			return generate_from_regex(cfg)

		if ctype == "Numeric":
			parts = [p.strip() for p in cfg.split(",") if p.strip()]
			value_type = "int"
			low = 0.0
			high = 100.0
			if parts:
				if parts[0].lower() in ("int", "float"):
					value_type = parts[0].lower()
					parts = parts[1:]
				if len(parts) >= 2:
					low = float(parts[0])
					high = float(parts[1])
			if high < low:
				low, high = high, low
			if value_type == "float":
				return round(random.uniform(low, high), 2)
			return random.randint(int(low), int(high))

		if ctype == "Date":
			days_back = 365
			if cfg.strip().isdigit():
				days_back = max(1, int(cfg.strip()))
			date_value = dt.date.today() - dt.timedelta(days=random.randint(0, days_back))
			return date_value.isoformat()

		if ctype == "Time":
			return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"

		return ""

	@staticmethod
	def _to_number(value):
		try:
			if isinstance(value, (int, float)):
				return float(value)
			return float(str(value).strip())
		except Exception:
			return None

	def _condition_match(self, row, condition):
		left = row.get(condition["column"], "")
		op = condition["operator"]
		right = condition["value"]

		if op in (">", ">=", "<", "<="):
			left_num = self._to_number(left)
			right_num = self._to_number(right)
			if left_num is None or right_num is None:
				return False
			if op == ">":
				return left_num > right_num
			if op == ">=":
				return left_num >= right_num
			if op == "<":
				return left_num < right_num
			return left_num <= right_num

		left_text = str(left)
		if op == "==":
			return left_text == right
		if op == "!=":
			return left_text != right
		if op == "contains":
			return right in left_text
		if op == "startswith":
			return left_text.startswith(right)
		if op == "endswith":
			return left_text.endswith(right)
		if op == "regex":
			try:
				return re.search(right, left_text) is not None
			except re.error:
				return False
		return False

	def _rule_match(self, row, rule):
		evaluations = [self._condition_match(row, cond) for cond in rule["conditions"]]
		if rule["logic"] == "OR":
			return any(evaluations)
		return all(evaluations)

	def _generate(self):
		if not self.columns:
			messagebox.showerror("Generate", "Add at least one column first.")
			return

		try:
			count = int(self.row_count.get().strip())
		except ValueError:
			messagebox.showerror("Generate", "Rows must be a whole number.")
			return

		count = max(1, count)
		rows = []
		for _ in range(count):
			row = {col["name"]: self._column_value(col) for col in self.columns}
			for rule in self.rules:
				if self._rule_match(row, rule):
					row[rule["target_column"]] = rule["target_value"]
					if rule.get("stop_on_match"):
						break
			rows.append(row)

		self.generated_rows = rows
		self._render_preview()
		self._log(f"Generated {len(rows)} rows.")

	def _render_preview(self):
		self.preview.delete(*self.preview.get_children())
		headers = [c["name"] for c in self.columns]
		self.preview["columns"] = headers
		for col in headers:
			self.preview.heading(col, text=col)
			self.preview.column(col, width=max(120, int(980 / max(1, len(headers)))))

		for row in self.generated_rows[:1000]:
			self.preview.insert("", tk.END, values=[row.get(c, "") for c in headers])

	def _export_csv(self):
		if not self.generated_rows:
			messagebox.showinfo("Export", "Generate data first.")
			return

		path = filedialog.asksaveasfilename(
			title="Export CSV",
			defaultextension=".csv",
			initialdir=SCRIPT_DIR,
			filetypes=[("CSV files", "*.csv")],
		)
		if not path:
			return

		headers = [c["name"] for c in self.columns]
		with open(path, "w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=headers)
			writer.writeheader()
			writer.writerows(self.generated_rows)
		self._log(f"Exported CSV: {path}")

	def _save_schema(self):
		path = filedialog.asksaveasfilename(
			title="Save schema",
			defaultextension=".json",
			initialdir=SCRIPT_DIR,
			filetypes=[("JSON files", "*.json")],
		)
		if not path:
			return

		payload = {"columns": self.columns, "rules": self.rules}
		with open(path, "w", encoding="utf-8") as handle:
			json.dump(payload, handle, indent=2)
		self._log(f"Saved schema: {path}")

	def _load_schema(self):
		path = filedialog.askopenfilename(
			title="Load schema",
			initialdir=SCRIPT_DIR,
			filetypes=[("JSON files", "*.json")],
		)
		if not path:
			return

		try:
			with open(path, "r", encoding="utf-8") as handle:
				payload = json.load(handle)
			columns = payload.get("columns", [])
			rules = payload.get("rules", [])

			if not isinstance(columns, list) or not isinstance(rules, list):
				raise ValueError("Schema is malformed.")

			self.columns = []
			for col in columns:
				if not isinstance(col, dict):
					continue
				name = str(col.get("name", "")).strip()
				ctype = str(col.get("type", "")).strip()
				config = str(col.get("config", ""))
				if name and ctype in self.COLUMN_TYPES:
					self.columns.append({"name": name, "type": ctype, "config": config})

			self.rules = []
			valid_names = {c["name"] for c in self.columns}
			for rule in rules:
				if not isinstance(rule, dict):
					continue
				target_col = str(rule.get("target_column", "")).strip()
				if target_col not in valid_names:
					continue

				parsed_conditions = []
				for cond in rule.get("conditions", []):
					if not isinstance(cond, dict):
						continue
					col_name = str(cond.get("column", "")).strip()
					op = str(cond.get("operator", "")).strip()
					val = str(cond.get("value", ""))
					if col_name in valid_names and op in self.OPERATORS:
						parsed_conditions.append({"column": col_name, "operator": op, "value": val})

				if not parsed_conditions:
					continue

				self.rules.append(
					{
						"name": str(rule.get("name", "Rule")).strip() or "Rule",
						"logic": "OR" if str(rule.get("logic", "AND")).upper() == "OR" else "AND",
						"conditions": parsed_conditions,
						"target_column": target_col,
						"target_value": str(rule.get("target_value", "")),
						"stop_on_match": bool(rule.get("stop_on_match", False)),
					}
				)

			self.generated_rows = []
			self._render_preview()
			self._refresh_column_list()
			self._refresh_rule_list()
			self._log(f"Loaded schema: {path}")
		except Exception as err:
			messagebox.showerror("Load Schema", f"Could not load schema:\n{err}")

	def _log(self, text: str):
		self.status.insert(tk.END, f"{text}\n")
		self.status.see(tk.END)


def main():
	root = tk.Tk()
	ManualModeApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
