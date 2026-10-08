"""
Label Assistant window: choose what powers the G (infer) shortcut.

  - YOLO-World   : zero-shot, driven by text prompts that are translated into the
                   workspace's own classes (e.g. "white horse" -> horse)
  - Trained model: the workspace's own models/<ws>/modelAssistant.pt, or any Ultralytics model chosen with
                   "Browse custom model" (its class names are mapped to workspace classes)

Settings are stored per workspace in configs/<workspace>.json.
Import this module only after config.load_workspace() has run.
"""
import copy
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageTk

from . import workspace_config as wcfg
from . import assistant_providers as providers
from .dialog_scroll import ScrollBody
from .config import class_manager, workspaceName, model_folder, BASE_DIR
from .inferenceObjectDetection import custom_model_available, custom_model_file
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT,
                    C_GREEN, C_AMBER, C_RED, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT, C_ON_RED, C_DANGER_BG, C_DANGER_FG)

SKIP = "(skip)"
MAX_VISIBLE_ROWS = 5
ROW_HEIGHT = 40


class LabelAssistantDialog:
    def __init__(self, parent):
        self.parent = parent
        self.cfg = copy.deepcopy(wcfg.get_assistant(workspaceName))
        self.classes = list(class_manager.get_classes())
        self.custom_ok = custom_model_available()
        self.model_path = self.cfg["custom_model"]["path"]        # a browsed model, or "" for the workspace's own
        self.model_names = []
        self.map_vars = {}
        if self.model_path and os.path.isfile(self.model_path):
            try:
                self.model_names = [n for n in providers.model_class_names(self.model_path) if n]
            except providers.AssistantError:
                pass
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

        self.scroll.fit(self.win, 660, parent)
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

        footer = tk.Frame(self.win, bg=C_BASE)            # packed first: a tall body can never push it off the screen
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(4, 16))
        self.scroll = ScrollBody(self.win, C_BASE)
        self.scroll.pack(fill=tk.BOTH, expand=True, padx=20, pady=(14, 8))
        body = self.scroll.inner

        tk.Label(body, text="Choose the model that annotates for you when you press G.",
                 bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 10))

        self._build_yolo_world_card(body)
        self._build_custom_card(body)
        self._build_confidence(body)

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
        self.scroll.nested.append(self.list_canvas)         # the target list keeps the wheel while it is over it

        self.empty_hint = tk.Label(
            self.yw_body, text="", bg=C_CARD, fg=C_AMBER, font=('Segoe UI', 8), anchor='w')
        self.empty_hint.pack(fill=tk.X, pady=(4, 0))

    def _build_custom_card(self, parent):
        self.custom_card = self._card(parent)
        self.custom_card.pack(fill=tk.X, pady=(0, 10))
        inner = tk.Frame(self.custom_card, bg=C_CARD)
        inner.pack(fill=tk.X, padx=14, pady=10)

        self.custom_radio = self._radio(inner, "My trained model", wcfg.PROVIDER_CUSTOM)
        self.custom_radio.pack(anchor='w')
        self.custom_info = tk.Label(inner, text="", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 8),
                                    anchor='w', justify=tk.LEFT, wraplength=580)
        self.custom_info.pack(anchor='w', padx=22)

        row = tk.Frame(inner, bg=C_CARD)
        row.pack(fill=tk.X, padx=22, pady=(8, 0))
        tk.Button(row, text="Browse custom model ...", command=self._browse_model, bg=C_ACCENT, fg=C_ON_ACCENT,
                  font=('Segoe UI', 8, 'bold'), relief=tk.FLAT, cursor='hand2', activebackground=C_ACCENT,
                  borderwidth=0).pack(side=tk.LEFT, ipadx=8, ipady=2)
        self.clear_btn = tk.Button(row, text="Use the workspace's own model", command=self._clear_model, bg=C_CARD2,
                                   fg=C_TXT1, font=('Segoe UI', 8), relief=tk.FLAT, cursor='hand2',
                                   activebackground=C_CARD2, activeforeground=C_TXT1, borderwidth=0)
        self.clear_btn.pack(side=tk.LEFT, padx=8, ipadx=8, ipady=2)

        self.model_map_frame = tk.Frame(inner, bg=C_CARD)
        self.model_map_frame.pack(fill=tk.X, padx=22, pady=(8, 0))
        self._refresh_custom_card()

    # ------------------------------------------------- browsed custom model
    def _refresh_custom_card(self):
        """Info line, radio state and the class-mapping table of the custom-model card."""
        browsed = bool(self.model_path)
        usable = (os.path.isfile(self.model_path) if browsed else custom_model_available())
        self.custom_ok = usable
        self.custom_radio.config(state=tk.NORMAL if usable else tk.DISABLED)
        self.clear_btn.config(state=tk.NORMAL if browsed else tk.DISABLED)
        if browsed and usable:
            self.custom_info.config(fg=C_GREEN, text=f"Custom model: {self.model_path}")
        elif browsed:
            self.custom_info.config(fg=C_AMBER, text=f"The chosen model was not found: {self.model_path}")
        elif usable:
            self.custom_info.config(fg=C_TXT2, text="Uses the model you trained in this workspace (classes are "
                                                   "matched by position in your class list).")
        else:
            self.custom_info.config(fg=C_AMBER, text="Your model was not found in this workspace. "
                                                    "Browse to choose any YOLO model (.pt).")
        self._render_model_map()

    def _render_model_map(self):
        for child in self.model_map_frame.winfo_children():
            child.destroy()
        self.map_vars = {}
        if not (self.model_path and self.model_names):
            self._fit()
            return
        tk.Label(self.model_map_frame, text="MODEL CLASS  →  WORKSPACE CLASS", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))
        auto = providers.map_head_classes(self.model_names, self.cfg["custom_model"]["class_map"], self.classes)
        for r, name in enumerate(self.model_names, 1):
            tk.Label(self.model_map_frame, text=name, bg=C_CARD, fg=C_TXT1, font=('Segoe UI', 10), width=24,
                     anchor='w').grid(row=r, column=0, pady=2, sticky='w')
            tk.Label(self.model_map_frame, text="→", bg=C_CARD, fg=C_ACCENT,
                     font=('Segoe UI', 11, 'bold')).grid(row=r, column=1, padx=8)
            var = tk.StringVar(value=auto.get(name) or SKIP)
            ttk.Combobox(self.model_map_frame, textvariable=var, values=[SKIP] + self.classes, state='readonly',
                         font=('Segoe UI', 10), width=22, style='Jelibox.TCombobox').grid(row=r, column=2, pady=2,
                                                                                         sticky='w')
            self.map_vars[name] = var
        self._fit()

    def _fit(self):
        if self.win.winfo_exists():
            self.scroll.fit(self.win, 660)

    def _browse_model(self):
        start = model_folder if os.path.isdir(model_folder) else BASE_DIR
        path = filedialog.askopenfilename(
            parent=self.win, title="Choose a custom YOLO model", initialdir=start,
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        self.win.config(cursor='watch')
        self.win.update_idletasks()
        try:
            names = [n for n in providers.model_class_names(path) if n]
        except providers.AssistantError as exc:
            messagebox.showwarning("Label Assistant", str(exc), parent=self.win)
            return
        finally:
            self.win.config(cursor='')
        self.model_path, self.model_names = os.path.normpath(path), names
        self.cfg["custom_model"]["class_map"] = {}               # a new model: map by name again
        self.provider_var.set(wcfg.PROVIDER_CUSTOM)
        self._refresh_custom_card()
        self._apply_provider_state()

    def _clear_model(self):
        self.model_path, self.model_names = "", []
        self.cfg["custom_model"]["class_map"] = {}
        self._refresh_custom_card()
        if self.provider_var.get() == wcfg.PROVIDER_CUSTOM and not self.custom_ok:
            self.provider_var.set(wcfg.PROVIDER_YOLO_WORLD)
        self._apply_provider_state()

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
        self._fit()

    def _on_rows_resize(self, _):
        self.list_canvas.configure(scrollregion=self.list_canvas.bbox('all'))

    def _on_wheel(self, event):
        if len(self.rows) > MAX_VISIBLE_ROWS:
            self.list_canvas.yview_scroll(int(-event.delta / 120), 'units')
        else:
            self.scroll.scroll_by(event.delta)

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
        class_map = {n: v.get() for n, v in self.map_vars.items() if v.get() != SKIP}
        if provider == wcfg.PROVIDER_CUSTOM and self.model_path:
            if not os.path.isfile(self.model_path):
                messagebox.showwarning("Label Assistant", f"The chosen model was not found:\n{self.model_path}",
                                       parent=self.win)
                return
            if self.model_names and not class_map:
                messagebox.showwarning("Label Assistant", "Map at least one model class to a workspace class.",
                                       parent=self.win)
                return
        self.cfg["custom_model"] = {"path": self.model_path, "class_map": class_map}
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
    """Open the settings window of the mode chosen on the front menu."""
    from . import assistant_modes
    mode = assistant_modes.get_mode()
    if mode == assistant_modes.MODE_LOCATE:
        from .ModeDialogs import open_locate_anything
        open_locate_anything(parent)
    elif mode == assistant_modes.MODE_HEAD:
        from .ModeDialogs import open_custom_head
        open_custom_head(parent)
    else:
        dlg = LabelAssistantDialog(parent)
        parent.wait_window(dlg.win)
