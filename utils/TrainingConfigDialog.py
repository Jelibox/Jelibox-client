"""
Training Configuration Dialog
Handles model selection, hyperparameter configuration, and model removal.
"""
import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox


class TrainingConfigDialog:
    """Dialog konfigurasi training: deteksi existing model, pilih base model,
    atur input size / epoch / batch."""

    # ─────────────────────────────────────────────────────────────
    # Constructor
    # ─────────────────────────────────────────────────────────────

    def __init__(self, parent, default_epoch=100, default_batch=16,
                 model_path=None, model_type="detect", model_folder=None):
        self.result        = None
        self.model_path    = model_path
        self.model_type    = model_type
        self.model_folder  = model_folder
        self.existing_info = None          # diisi oleh _load_existing_model_info

        # Cek existing model SEBELUM membangun UI
        if model_path and os.path.exists(model_path):
            self._load_existing_model_info(model_path)

        # ── Theme colors ──────────────────────────────────────
        from .theme import (C_BASE, C_CARD, C_TXT1, C_TXT2, C_TXT3, C_ACCENT, C_ON_ACCENT,
                            C_BORDER, C_RED, C_ON_RED)
        self.bg_dark      = C_BASE
        self.bg_secondary = C_CARD
        self.fg_light     = C_TXT1
        self.fg_muted     = C_TXT2
        self.fg_hint      = C_TXT3
        self.accent_blue  = C_ACCENT
        self.accent_green = C_ACCENT
        self.on_accent    = C_ON_ACCENT
        self.border       = C_BORDER
        self.danger       = C_RED
        self.on_danger    = C_ON_RED

        # ── Window ────────────────────────────────────────────
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Training Configuration")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.configure(bg=self.bg_dark)

        # ── Build semua widget ────────────────────────────────
        self._build_title_bar()

        content = tk.Frame(self.dialog, bg=self.bg_dark, padx=40, pady=10)
        content.pack(fill=tk.BOTH, expand=True)

        self._build_model_section(content)
        self._build_separator(content)
        self._build_imgsz_row(content, default_imgsz=640)
        self._build_spinbox_row(content, "Epochs:",     "epoch_var", default_epoch, 1,   1000, "(1–1000)")
        self._build_spinbox_row(content, "Batch Size:", "batch_var", default_batch, 1,   128,  "(1–128)")
        self._build_spinbox_row(content, "Freeze:",     "freeze_var", 0,            0,   100,  "layers (0 = none, 10 = backbone)")
        self._build_float_row(content,   "LR (lr0):",   "lr0_var",    0.001,        "AdamW start LR (1e-5 – 0.1)")
        self._build_float_row(content,   "Weight Decay:", "wd_var",   0.0005,       "AdamW weight decay (0 – 0.1)")
        self._build_hint_label(content)
        self._build_action_buttons(content)

        # ── Keyboard shortcuts ────────────────────────────────
        self.dialog.bind('<Return>', lambda e: self.start())
        self.dialog.bind('<Escape>', lambda e: self.cancel())

        # ── Auto-size lalu center ke parent ──────────────────
        self.dialog.update_idletasks()
        w = self.dialog.winfo_reqwidth()
        h = self.dialog.winfo_reqheight()
        px = parent.winfo_x() + (parent.winfo_width()  // 2) - (w // 2)
        py = parent.winfo_y() + (parent.winfo_height() // 2) - (h // 2)
        self.dialog.geometry(f"{w}x{h}+{px}+{py}")

    # ─────────────────────────────────────────────────────────────
    # Model info loader
    # ─────────────────────────────────────────────────────────────

    def _load_existing_model_info(self, model_path):
        """Baca nama & imgsz dari {model_folder}/train_run/args.yaml."""
        args_yaml = os.path.join(self.model_folder, "train_run", "args.yaml")

        # Jika folder train_run / args.yaml tidak ada → anggap tidak ada existing model
        if not os.path.exists(args_yaml):
            self.existing_info = None
            return

        try:
            import yaml

            with open(args_yaml, "r") as f:
                train_args = yaml.safe_load(f)

            imgsz      = train_args.get("imgsz", 640)
            model_name = train_args.get("model", "Unknown")

            # Ambil nama pendek, misal "yolov8x.pt" → "yolov8x"
            model_name = os.path.basename(model_name).replace(".pt", "")

            self.existing_info = {"name": model_name, "imgsz": imgsz}

        except Exception as e:
            print(f"[WARN] Could not read args.yaml: {e}")
            self.existing_info = None

    # ─────────────────────────────────────────────────────────────
    # UI builders
    # ─────────────────────────────────────────────────────────────

    def _build_title_bar(self):
        frame = tk.Frame(self.dialog, bg=self.accent_blue, height=60)
        frame.pack(fill=tk.X)
        frame.pack_propagate(False)
        tk.Label(
            frame,
            text="⚙️  Training Configuration",
            font=("Segoe UI", 16, "bold"),
            bg=self.accent_blue, fg=self.on_accent
        ).pack(pady=15)

    def _build_separator(self, parent):
        tk.Frame(parent, bg=self.border, height=1).pack(fill=tk.X, pady=10)

    def _build_hint_label(self, parent):
        tk.Label(
            parent,
            text="💡  Batch size depends on GPU VRAM.\n"
                 "    More epochs = longer training time.",
            font=("Segoe UI", 9),
            bg=self.bg_dark, fg=self.fg_muted,
            justify=tk.LEFT
        ).pack(pady=(6, 2), anchor="w")

    def _build_action_buttons(self, parent):
        frame = tk.Frame(parent, bg=self.bg_dark)
        frame.pack(fill=tk.X, pady=10)

        tk.Button(
            frame, text="❌  Cancel",
            font=("Segoe UI", 11, "bold"),
            bg=self.danger, fg=self.on_danger, activebackground=self.danger,
            cursor="hand2", relief=tk.FLAT,
            command=self.cancel, width=12
        ).pack(side=tk.LEFT, padx=5, ipady=6)

        tk.Button(
            frame, text="🚀  Start Training",
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_green, fg=self.on_accent, activebackground=self.accent_green,
            cursor="hand2", relief=tk.FLAT,
            command=self.start, width=15
        ).pack(side=tk.RIGHT, padx=5, ipady=6)

    # ── Model section (existing info  –atau–  base model dropdown) ──

    def _build_model_section(self, parent):
        """Tampilkan info existing model ATAU dropdown base model."""
        frame = tk.Frame(parent, bg=self.bg_dark)
        frame.pack(fill=tk.X, pady=(4, 0))

        # Cek ulang: existing_info valid hanya jika model_path juga ada
        has_existing = (
            self.existing_info is not None
            and self.model_path is not None
            and os.path.exists(self.model_path)
        )

        if has_existing:
            self._build_existing_model_panel(frame)
        else:
            self._build_base_model_dropdown(frame)

    def _build_existing_model_panel(self, parent):
        """Panel info ketika model existing ditemukan."""
        panel = tk.Frame(parent, bg=self.bg_secondary, padx=14, pady=10)
        panel.pack(fill=tk.X)

        # Header
        tk.Label(
            panel,
            text="✅Existing Model Detected",
            font=("Segoe UI", 10, "bold"),
            bg=self.bg_secondary, fg=self.accent_green
        ).pack(anchor="w")
        
        if self.existing_info['name'] == "modelAssistant":
            model_display_name = "The YOLO model type is not specified in args.yml"
        else:
            model_display_name = self.existing_info['name']

        # Detail info
        details = (
            f"  Model      :  {model_display_name}\n"
            f"  Input Size :  {self.existing_info['imgsz']} px"
        )
        tk.Label(
            panel,
            text=details,
            font=("Segoe UI", 10),
            bg=self.bg_secondary, fg=self.fg_light,
            justify=tk.LEFT
        ).pack(anchor="w", pady=(6, 4))

        tk.Label(
            panel,
            text="Training will CONTINUE from this model.",
            font=("Segoe UI", 9, "italic"),
            bg=self.bg_secondary, fg=self.fg_muted
        ).pack(anchor="w")

        # Tombol Remove
        tk.Button(
            panel,
            text="🗑  Remove Model",
            font=("Segoe UI", 9, "bold"),
            bg=self.danger, fg=self.on_danger, activebackground=self.danger,
            cursor="hand2", relief=tk.FLAT,
            command=self._confirm_remove_model
        ).pack(anchor="e", pady=(8, 2))

    def _build_base_model_dropdown(self, parent):
        """Dropdown pilih base model ketika tidak ada model existing."""
        row = tk.Frame(parent, bg=self.bg_dark)
        row.pack(fill=tk.X, pady=6)

        tk.Label(
            row,
            text="Base Model:",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_dark, fg=self.fg_light,
            width=12, anchor="w"
        ).pack(side=tk.LEFT)

        models = self._get_base_model_list()
        self.base_model_var = tk.StringVar(value=models[0])

        combo = ttk.Combobox(
            row,
            textvariable=self.base_model_var,
            values=models,
            state="readonly",
            font=("Segoe UI", 10),
            width=22
        )
        combo.pack(side=tk.LEFT, padx=10)

        task_label = "seg" if self.model_type == "seg" else "detect"
        tk.Label(
            row,
            text=f"(task: {task_label})",
            font=("Segoe UI", 9),
            bg=self.bg_dark, fg=self.fg_hint
        ).pack(side=tk.LEFT)

    def _get_base_model_list(self):
        """Kembalikan daftar model sesuai model_type."""
        if self.model_type == "seg":
            return [
                "yolov8n-seg.pt",  "yolov8s-seg.pt",  "yolov8m-seg.pt",
                "yolov8l-seg.pt",  "yolov8x-seg.pt",
                "yolov9c-seg.pt",  "yolov9e-seg.pt",
                "yolo11n-seg.pt",  "yolo11s-seg.pt",  "yolo11m-seg.pt",
                "yolo11l-seg.pt",  "yolo11x-seg.pt",
                "yolo26n-seg.pt",  "yolo26s-seg.pt",  "yolo26m-seg.pt",
                "yolo26l-seg.pt",  "yolo26x-seg.pt",
            ]
        else:  # detect
            return [
                "yolov8n.pt",  "yolov8s.pt",  "yolov8m.pt",  "yolov8l.pt",  "yolov8x.pt",
                "yolov9t.pt",  "yolov9s.pt",  "yolov9m.pt",  "yolov9c.pt",  "yolov9e.pt",
                "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
                "yolo11n.pt",  "yolo11s.pt",  "yolo11m.pt",  "yolo11l.pt",  "yolo11x.pt",
                "yolo12n.pt",  "yolo12s.pt",  "yolo12m.pt",  "yolo12l.pt",  "yolo12x.pt",
                "yolo26n.pt",  "yolo26s.pt",  "yolo26m.pt",  "yolo26l.pt",  "yolo26x.pt",
            ]

    # ── Input Size row ──────────────────────────────────────────

    def _build_imgsz_row(self, parent, default_imgsz=640):
        """Input size dengan default dari model existing (jika ada)."""
        if self.existing_info:
            default_imgsz = self.existing_info['imgsz']

        row = tk.Frame(parent, bg=self.bg_dark)
        row.pack(fill=tk.X, pady=6)

        tk.Label(
            row,
            text="Input Size:",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_dark, fg=self.fg_light,
            width=12, anchor="w"
        ).pack(side=tk.LEFT)

        self.imgsz_var = tk.IntVar(value=default_imgsz)
        tk.Spinbox(
            row,
            from_=32, to=1920, increment=32,
            textvariable=self.imgsz_var,
            font=("Segoe UI", 11),
            width=8,
            bg=self.bg_secondary, fg=self.fg_light,
            buttonbackground=self.bg_secondary,
            relief=tk.FLAT, insertbackground=self.fg_light
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            row,
            text="px  (multiples of 32 recommended)",
            font=("Segoe UI", 9),
            bg=self.bg_dark, fg=self.fg_hint
        ).pack(side=tk.LEFT)

    # ── Generic spinbox row ─────────────────────────────────────

    def _build_spinbox_row(self, parent, label, var_name, default,
                           from_, to, hint):
        row = tk.Frame(parent, bg=self.bg_dark)
        row.pack(fill=tk.X, pady=6)

        tk.Label(
            row,
            text=label,
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_dark, fg=self.fg_light,
            width=12, anchor="w"
        ).pack(side=tk.LEFT)

        var = tk.IntVar(value=default)
        setattr(self, var_name, var)

        tk.Spinbox(
            row,
            from_=from_, to=to,
            textvariable=var,
            font=("Segoe UI", 11),
            width=8,
            bg=self.bg_secondary, fg=self.fg_light,
            buttonbackground=self.bg_secondary,
            relief=tk.FLAT, insertbackground=self.fg_light
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            row,
            text=hint,
            font=("Segoe UI", 9),
            bg=self.bg_dark, fg=self.fg_hint
        ).pack(side=tk.LEFT)

    # ── Float entry row ─────────────────────────────────────────

    def _build_float_row(self, parent, label, var_name, default, hint):
        row = tk.Frame(parent, bg=self.bg_dark)
        row.pack(fill=tk.X, pady=6)

        tk.Label(
            row,
            text=label,
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_dark, fg=self.fg_light,
            width=12, anchor="w"
        ).pack(side=tk.LEFT)

        var = tk.StringVar(value=str(default))
        setattr(self, var_name, var)

        tk.Entry(
            row,
            textvariable=var,
            font=("Segoe UI", 11),
            width=10,
            bg=self.bg_secondary, fg=self.fg_light,
            relief=tk.FLAT, insertbackground=self.fg_light
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            row,
            text=hint,
            font=("Segoe UI", 9),
            bg=self.bg_dark, fg=self.fg_hint
        ).pack(side=tk.LEFT)

    # ─────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────

    def _confirm_remove_model(self):
        """Konfirmasi lalu hapus file model existing + folder train_run."""
        train_run_dir = os.path.join(self.model_folder, "train_run")

        confirmed = messagebox.askyesno(
            "Confirm Remove",
            f"Are you sure you want to DELETE the existing model?\n"
            f"\n"
            "This action CANNOT be undone.",
            parent=self.dialog
        )
        if not confirmed:
            return

        errors = []

        # Hapus file model
        try:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
        except Exception as e:
            errors.append(f"Model file: {e}")

        # Hapus folder train_run
        try:
            if os.path.exists(train_run_dir):
                shutil.rmtree(train_run_dir)
        except Exception as e:
            errors.append(f"train_run folder: {e}")

        if errors:
            messagebox.showerror(
                "Error",
                "Some items could not be deleted:\n\n" + "\n".join(errors),
                parent=self.dialog
            )
            return

        messagebox.showinfo(
            "Model Removed",
            "Existing model and training data deleted.\n\n"
            "Please click Train again to select a base model.",
            parent=self.dialog
        )
        self.result = {"action": "removed"}
        self.dialog.destroy()

    def start(self):
        """Validasi input lalu kumpulkan result."""
        try:
            epoch = self.epoch_var.get()
            batch = self.batch_var.get()
            imgsz = self.imgsz_var.get()
            freeze = self.freeze_var.get()
            try:
                lr0 = float(self.lr0_var.get())
                weight_decay = float(self.wd_var.get())
            except ValueError:
                messagebox.showwarning("Invalid Input", "LR and Weight Decay must be numbers!", parent=self.dialog)
                return

            if not (0 <= freeze <= 100):
                messagebox.showwarning("Invalid Input", "Freeze must be between 0–100 layers!", parent=self.dialog)
                return
            if not (1e-5 <= lr0 <= 0.1):
                messagebox.showwarning("Invalid Input", "LR (lr0) must be between 0.00001–0.1!", parent=self.dialog)
                return
            if not (0 <= weight_decay <= 0.1):
                messagebox.showwarning("Invalid Input", "Weight decay must be between 0–0.1!", parent=self.dialog)
                return
            if not (1 <= epoch <= 1000):
                messagebox.showwarning("Invalid Input", "Epochs must be between 1–1000!", parent=self.dialog)
                return
            if not (1 <= batch <= 128):
                messagebox.showwarning("Invalid Input", "Batch size must be between 1–128!", parent=self.dialog)
                return
            if imgsz < 32 or imgsz % 32 != 0:
                messagebox.showwarning("Invalid Input", "Input size must be ≥ 32 and a multiple of 32!", parent=self.dialog)
                return

            self.result = {
                'epoch': epoch,
                'batch': batch,
                'imgsz': imgsz,
                'freeze': freeze,
                'lr0': lr0,
                'weight_decay': weight_decay,
            }

            # base_model hanya relevan jika tidak ada existing model
            if not self.existing_info:
                self.result['base_model'] = self.base_model_var.get()

            self.dialog.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}", parent=self.dialog)

    def cancel(self):
        self.result = None
        self.dialog.destroy()

    def show(self):
        """Tampilkan dialog, tunggu, kembalikan result dict atau None."""
        self.dialog.wait_window()
        return self.result
