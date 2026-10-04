"""
Label Assistant window: choose what powers the G (infer) shortcut.

  - YOLO-World   : zero-shot, driven by text prompts that are translated into the
                   workspace's own classes (e.g. "white horse" -> horse)
  - Trained model: the workspace's own models/<ws>/modelAssistant.pt

Settings are stored per workspace in configs/<workspace>.json.
Import this module only after config.load_workspace() has run.
"""
import copy
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

from . import workspace_config as wcfg
from .config import class_manager, workspaceName
from .inferenceObjectDetection import custom_model_available
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT,
                    C_GREEN, C_AMBER, C_RED, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT, C_ON_RED, C_DANGER_BG, C_DANGER_FG)

MAX_VISIBLE_ROWS = 5
ROW_HEIGHT = 40


class LabelAssistantDialog:
    def __init__(self, parent):
        self.parent = parent
        self.cfg = copy.deepcopy(wcfg.get_assistant(workspaceName))
        self.classes = list(class_manager.get_classes())
        self.custom_ok = custom_model_available()
        self.rows = []          # [{'frame','prompt','cls','entry','combo','x'}]

        provider = self.cfg["provider"]
        if provider == wcfg.PROVIDER_CUSTOM and not self.custom_ok:
            provider = wcfg.PROVIDER_YOLO_WORLD
        self.provider_var = tk.StringVar(value=provider)
        self.model_var = tk.StringVar(value=self.cfg["yolo_world"]["model"])
        self.conf_var = tk.DoubleVar(value=self.cfg["confidence"])

        self.win = tk.Toplevel(parent)
        self.win.title("Label Assistant")
        self.win.configure(bg=C_BASE)
        self.win.transient(parent)
        self.win.resizable(False, False)

        self._style()
        self._build()
        for t in self.cfg["yolo_world"]["target_classes"]:
            self._add_row(t["prompt"], t["map_to"])
        if not self.rows:
            self._add_row()
        self._apply_provider_state()

        self.win.update_idletasks()
        w, h = 660, self.win.winfo_reqheight()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (w // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (h // 2)
        self.win.geometry(f"{w}x{h}+{max(x, 0)}+{max(y, 0)}")
        self.win.bind('<Escape>', lambda e: self.win.destroy())
        self.win.grab_set()
        self.win.focus_force()

    # ---------------------------------------------------------- styling
    def _style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Jelibox.TCombobox',
                        fieldbackground=C_CARD, background=C_CARD2,
                        foreground=C_TXT1, selectbackground=C_CARD,
                        selectforeground=C_TXT1, bordercolor=C_BORDER,
                        darkcolor=C_CARD2, lightcolor=C_CARD2,
                        arrowcolor=C_ACCENT, padding=4)
        style.map('Jelibox.TCombobox',
                  fieldbackground=[('disabled', C_PANEL), ('readonly', C_CARD)],
                  foreground=[('disabled', C_TXT3)],
                  arrowcolor=[('disabled', C_TXT3)])
        self.win.option_add('*TCombobox*Listbox.background', C_CARD)
        self.win.option_add('*TCombobox*Listbox.foreground', C_TXT1)
        self.win.option_add('*TCombobox*Listbox.selectBackground', C_ACCENT)
        self.win.option_add('*TCombobox*Listbox.selectForeground', C_ON_ACCENT)

    # ------------------------------------------------------------ layout
    def _build(self):
        header = tk.Frame(self.win, bg=C_PANEL, height=54)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        try:
            logo = Image.open("assets/jelibox.png").resize((32, 32))
            self._logo = ImageTk.PhotoImage(logo)
            tk.Label(header, image=self._logo, bg=C_PANEL).pack(side=tk.LEFT, padx=(14, 8))
        except Exception:
            pass
        tk.Label(header, text="LABEL ASSISTANT", bg=C_PANEL, fg=C_TXT1,
                 font=('Segoe UI', 12, 'bold')).pack(side=tk.LEFT)
        tk.Frame(self.win, bg=C_ACCENT, height=2).pack(fill=tk.X)

        body = tk.Frame(self.win, bg=C_BASE)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(14, 8))

        tk.Label(body, text="Choose the model that annotates for you when you press G.",
                 bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 10))

        self._build_yolo_world_card(body)
        self._build_custom_card(body)
        self._build_confidence(body)

        footer = tk.Frame(self.win, bg=C_BASE)
        footer.pack(fill=tk.X, padx=20, pady=(4, 16))
        tk.Button(footer, text="Save", command=self._save, bg=C_ACCENT, fg=C_ON_ACCENT,
                  font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, cursor='hand2',
                  activebackground=C_ACCENT, borderwidth=0
                  ).pack(side=tk.RIGHT, ipadx=22, ipady=4)
        tk.Button(footer, text="Cancel", command=self.win.destroy, bg=C_CARD2, fg=C_TXT1,
                  font=('Segoe UI', 9), relief=tk.FLAT, cursor='hand2',
                  activebackground=C_CARD2, activeforeground=C_TXT1, borderwidth=0
                  ).pack(side=tk.RIGHT, padx=8, ipadx=16, ipady=4)

    def _card(self, parent):
        return tk.Frame(parent, bg=C_CARD, highlightbackground=C_BORDER, highlightthickness=1)

    def _radio(self, parent, text, value, state=tk.NORMAL):
        return tk.Radiobutton(
            parent, text=text, variable=self.provider_var, value=value,
            command=self._apply_provider_state, state=state,
            bg=C_CARD, fg=C_TXT1, selectcolor=C_BASE, activebackground=C_CARD,
            activeforeground=C_ACCENT, disabledforeground=C_TXT3,
            font=('Segoe UI', 10, 'bold'), cursor='hand2', highlightthickness=0, bd=0)

    def _build_yolo_world_card(self, parent):
        self.yw_card = self._card(parent)
        self.yw_card.pack(fill=tk.X, pady=(0, 10))
        inner = tk.Frame(self.yw_card, bg=C_CARD)
        inner.pack(fill=tk.X, padx=14, pady=10)

        self._radio(inner, "YOLO-World", wcfg.PROVIDER_YOLO_WORLD).pack(anchor='w')
        tk.Label(inner, text="Zero-shot: describe what to look for, no training needed.",
                 bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 8)).pack(anchor='w', padx=22)

        self.yw_body = tk.Frame(inner, bg=C_CARD)
        self.yw_body.pack(fill=tk.X, padx=22, pady=(10, 0))

        model_row = tk.Frame(self.yw_body, bg=C_CARD)
        model_row.pack(fill=tk.X)
        tk.Label(model_row, text="Model", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            model_row, textvariable=self.model_var, values=wcfg.YOLO_WORLD_MODELS,
            state='readonly', font=('Segoe UI', 10), width=22, style='Jelibox.TCombobox')
        self.model_combo.pack(side=tk.LEFT, padx=10)

        tk.Frame(self.yw_body, bg=C_BORDER, height=1).pack(fill=tk.X, pady=10)

        head = tk.Frame(self.yw_body, bg=C_CARD)
        head.pack(fill=tk.X)
        tk.Label(head, text="TARGET CLASSES", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8, 'bold')).pack(side=tk.LEFT)
        self.add_btn = tk.Button(
            head, text="＋  Add target class", command=lambda: self._add_row(focus=True),
            bg=C_ACCENT, fg=C_ON_ACCENT, font=('Segoe UI', 8, 'bold'), relief=tk.FLAT,
            cursor='hand2', activebackground=C_ACCENT, borderwidth=0)
        self.add_btn.pack(side=tk.RIGHT, ipadx=8, ipady=2)

        cols = tk.Frame(self.yw_body, bg=C_CARD)
        cols.pack(fill=tk.X, pady=(8, 2))
        tk.Label(cols, text="Search for (letters & spaces)", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8)).place(x=0, y=0)
        tk.Label(cols, text="Save as workspace class", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8)).place(x=268, y=0)
        cols.configure(height=20)

        list_wrap = tk.Frame(self.yw_body, bg=C_CARD)
        list_wrap.pack(fill=tk.X)
        self.list_canvas = tk.Canvas(list_wrap, bg=C_CARD, highlightthickness=0,
                                     height=ROW_HEIGHT)
        self.list_scroll = tk.Scrollbar(list_wrap, orient='vertical',
                                        command=self.list_canvas.yview,
                                        bg=C_BORDER, troughcolor=C_CARD, relief=tk.FLAT, width=8)
        self.list_canvas.configure(yscrollcommand=self.list_scroll.set)
        self.list_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rows_frame = tk.Frame(self.list_canvas, bg=C_CARD)
        self.rows_window = self.list_canvas.create_window((0, 0), window=self.rows_frame, anchor='nw')
        self.rows_frame.bind('<Configure>', self._on_rows_resize)
        self.list_canvas.bind('<Configure>',
                              lambda e: self.list_canvas.itemconfig(self.rows_window, width=e.width))
        self.list_canvas.bind('<Enter>', lambda e: self.list_canvas.bind_all('<MouseWheel>', self._on_wheel))
        self.list_canvas.bind('<Leave>', lambda e: self.list_canvas.unbind_all('<MouseWheel>'))

        self.empty_hint = tk.Label(
            self.yw_body, text="", bg=C_CARD, fg=C_AMBER, font=('Segoe UI', 8), anchor='w')
        self.empty_hint.pack(fill=tk.X, pady=(4, 0))

    def _build_custom_card(self, parent):
        self.custom_card = self._card(parent)
        self.custom_card.pack(fill=tk.X, pady=(0, 10))
        inner = tk.Frame(self.custom_card, bg=C_CARD)
        inner.pack(fill=tk.X, padx=14, pady=10)

        self._radio(inner, "My trained model", wcfg.PROVIDER_CUSTOM,
                    state=tk.NORMAL if self.custom_ok else tk.DISABLED).pack(anchor='w')
        if self.custom_ok:
            tk.Label(inner, text="Uses the model you trained in this workspace.",
                     bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 8)).pack(anchor='w', padx=22)
        else:
            tk.Label(inner, text="Your model was not found in this workspace.",
                     bg=C_CARD, fg=C_AMBER, font=('Segoe UI', 8)).pack(anchor='w', padx=22)

    def _build_confidence(self, parent):
        row = tk.Frame(parent, bg=C_BASE)
        row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(row, text="Confidence threshold", bg=C_BASE, fg=C_TXT2,
                 font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.conf_text = tk.Label(row, text="", bg=C_BASE, fg=C_ACCENT,
                                  font=('Segoe UI', 9, 'bold'), width=5)
        self.conf_text.pack(side=tk.RIGHT)
        tk.Scale(row, variable=self.conf_var, from_=0.05, to=0.95, resolution=0.05,
                 orient='horizontal', showvalue=False, command=self._on_conf,
                 bg=C_BASE, fg=C_TXT1, troughcolor=C_CARD, highlightthickness=0,
                 activebackground=C_ACCENT, bd=0, length=300
                 ).pack(side=tk.RIGHT, padx=10)
        self._on_conf()

    def _on_conf(self, _=None):
        self.conf_text.config(text=f"{self.conf_var.get():.2f}")

    # ------------------------------------------------------- target rows
    def _add_row(self, prompt="", map_to="", focus=False):
        frame = tk.Frame(self.rows_frame, bg=C_CARD, height=ROW_HEIGHT)
        frame.pack(fill=tk.X)
        frame.pack_propagate(False)

        prompt_var = tk.StringVar(value=prompt)
        cls_var = tk.StringVar(value=map_to if map_to in self.classes else "")

        vcmd = (self.win.register(
            lambda proposed: wcfg.is_valid_prompt_text(proposed)), '%P')
        entry = tk.Entry(frame, textvariable=prompt_var, width=26, font=('Segoe UI', 10),
                         bg=C_CARD2, fg=C_TXT1, insertbackground=C_ACCENT, relief=tk.FLAT,
                         highlightthickness=1, highlightbackground=C_BORDER,
                         highlightcolor=C_ACCENT, disabledbackground=C_PANEL,
                         disabledforeground=C_TXT3, validate='key', validatecommand=vcmd)
        entry.pack(side=tk.LEFT, ipady=4, pady=6)

        arrow = tk.Label(frame, text="→", bg=C_CARD, fg=C_ACCENT, font=('Segoe UI', 11, 'bold'))
        arrow.pack(side=tk.LEFT, padx=8)

        combo = ttk.Combobox(frame, textvariable=cls_var, values=self.classes,
                             state='readonly', font=('Segoe UI', 10), width=18,
                             style='Jelibox.TCombobox')
        combo.pack(side=tk.LEFT, pady=6)

        row = {'frame': frame, 'prompt': prompt_var, 'cls': cls_var,
               'entry': entry, 'combo': combo}
        x = tk.Button(frame, text="✕", command=lambda r=row: self._remove_row(r),
                      bg=C_CARD, fg=C_DANGER_FG, font=('Segoe UI', 10, 'bold'), relief=tk.FLAT,
                      cursor='hand2', activebackground=C_RED, activeforeground=C_ON_RED,
                      borderwidth=0, width=3)
        x.pack(side=tk.RIGHT, padx=(6, 4), pady=6)
        row['x'] = x
        self.rows.append(row)

        self._update_rows_layout()
        if focus:
            entry.focus_set()
            self.list_canvas.update_idletasks()
            self.list_canvas.yview_moveto(1.0)

    def _remove_row(self, row):
        row['frame'].destroy()
        self.rows.remove(row)
        self._update_rows_layout()

    def _update_rows_layout(self):
        visible = max(1, min(len(self.rows), MAX_VISIBLE_ROWS))
        self.list_canvas.configure(height=visible * ROW_HEIGHT)
        if len(self.rows) > MAX_VISIBLE_ROWS:
            self.list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.list_scroll.pack_forget()
        self.empty_hint.config(
            text="" if self.rows else "No target class yet - click “Add target class”.")
        self.win.update_idletasks()
        self.win.geometry(f"660x{self.win.winfo_reqheight()}")

    def _on_rows_resize(self, _):
        self.list_canvas.configure(scrollregion=self.list_canvas.bbox('all'))

    def _on_wheel(self, event):
        if len(self.rows) > MAX_VISIBLE_ROWS:
            self.list_canvas.yview_scroll(int(-event.delta / 120), 'units')

    # ----------------------------------------------------- radio -> state
    def _apply_provider_state(self):
        yw_on = self.provider_var.get() == wcfg.PROVIDER_YOLO_WORLD
        state = tk.NORMAL if yw_on else tk.DISABLED
        self.model_combo.config(state='readonly' if yw_on else 'disabled')
        self.add_btn.config(state=state, bg=C_ACCENT if yw_on else C_CARD2)
        for r in self.rows:
            r['entry'].config(state=state)
            r['combo'].config(state='readonly' if yw_on else 'disabled')
            r['x'].config(state=state)
        self.yw_card.config(highlightbackground=C_ACCENT if yw_on else C_BORDER)
        self.custom_card.config(highlightbackground=C_BORDER if yw_on else C_ACCENT)

    # -------------------------------------------------------------- save
    def _collect_targets(self):
        """Returns (targets, error). Fully empty rows are ignored."""
        return wcfg.validate_targets((r['prompt'].get(), r['cls'].get()) for r in self.rows)

    def _save(self):
        provider = self.provider_var.get()
        targets, error = self._collect_targets()
        if provider == wcfg.PROVIDER_YOLO_WORLD:
            if error:
                messagebox.showwarning("Label Assistant", error, parent=self.win)
                return
            if not targets:
                messagebox.showwarning(
                    "Label Assistant",
                    "YOLO-World can't work without a target class.\n\n"
                    "Your settings will be saved, but pressing G will not annotate anything "
                    "until you add one. You can keep annotating manually in the meantime.",
                    parent=self.win)
        # Keep whatever rows are valid even when the trained model is selected.
        self.cfg["provider"] = provider
        self.cfg["confidence"] = round(float(self.conf_var.get()), 2)
        self.cfg["yolo_world"]["model"] = self.model_var.get()
        self.cfg["yolo_world"]["target_classes"] = targets if targets is not None else \
            self.cfg["yolo_world"]["target_classes"]
        wcfg.set_assistant(workspaceName, self.cfg)
        print(f"[Assistant] Saved: {provider}, conf={self.cfg['confidence']}, "
              f"{len(self.cfg['yolo_world']['target_classes'])} target class(es)")
        self.win.destroy()


def open_label_assistant(parent):
    dlg = LabelAssistantDialog(parent)
    parent.wait_window(dlg.win)
