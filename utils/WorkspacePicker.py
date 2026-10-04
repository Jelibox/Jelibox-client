"""
Workspace Picker - VS Code-style "no folder opened" screen for Jelibox.

Lists the workspaces found in datasetsInput (folders grouped by name, e.g.
weapon-1 / weapon-2 -> "weapon"). Picking a workspace expands it to show its
instances; picking an instance launches the annotation GUI for that folder
in a separate process and hides this picker until the annotation window is
closed (its "Back to Workspace" button), at which point the picker reappears.

Deliberately imports nothing from utils.config / utils.AnnotationGUI - this
screen must be safe to show before any dataset instance folder is chosen.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT,
                    C_GREEN, C_RED, C_AMBER, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT, C_ON_RED, MODE, toggle_mode)
from .workspace_manager import (list_workspaces, instance_path, count_images, DATASETS_ROOT,
                                count_images_in_folder, sanitize_workspace_name, parse_classes_input,
                                existing_classes_for_workspace, create_workspace_instance,
                                workspace_stats, delete_instance, delete_workspace)
from .dataset_import import (scan_dataset_folder, detect_dataset_format, import_dataset,
                             yolo_classes_resolved, sanitize_filename_prefix)


class WorkspacePickerApp:
    def __init__(self, root, python_exe=None, entry_script=None):
        self.root = root
        self.python_exe = python_exe or sys.executable
        self.entry_script = entry_script or os.path.abspath(sys.argv[0])

        self.process = None
        self.expanded = set()
        self.workspaces = {}

        self.root.title("Jelibox — Workspaces")
        self.root.geometry("1100x680")
        self.root.minsize(820, 520)
        self.root.configure(bg=C_BASE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_header()
        self._build_body()
        self.refresh_workspaces()

    # ----------------------------------------------------------
    #  Header (matches AnnotationGUI's header bar)
    # ----------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self.root, bg=C_PANEL, height=54)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=C_PANEL)
        brand.pack(side=tk.LEFT, padx=(14, 0))

        try:
            from PIL import Image, ImageTk
            logo_img = Image.open("assets/jelibox.png").resize((32, 32))
            self.logo = ImageTk.PhotoImage(logo_img)
            tk.Label(brand, image=self.logo, bg=C_PANEL).pack(side=tk.LEFT, pady=10)
        except Exception:
            pass

        name_stack = tk.Frame(brand, bg=C_PANEL)
        name_stack.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(name_stack, text="Jelibox", bg=C_PANEL, fg=C_TXT1,
                 font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        tk.Label(name_stack, text="WORKSPACES", bg=C_PANEL, fg=C_TXT3,
                 font=('Segoe UI', 6, 'bold')).pack(anchor='w')

        tk.Frame(header, bg=C_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=10, padx=10)

        self._btn(header, "⟳  Refresh", self.refresh_workspaces,
                  C_CARD2, C_TXT1).pack(side=tk.LEFT, padx=(0, 6), pady=12, ipady=5, ipadx=10)

        self._btn(header, "+  Add Workspace", lambda: self._open_add_workspace_dialog(),
                  C_ACCENT, C_ON_ACCENT, bold=True).pack(side=tk.LEFT, padx=(0, 6), pady=12, ipady=5, ipadx=10)

        self._btn(header, "⬇  Import Dataset", self._open_import_dataset_dialog,
                  C_CARD2, C_TXT1, bold=True).pack(side=tk.LEFT, pady=12, ipady=5, ipadx=10)

        self.theme_btn = self._btn(
            header, "◐  Dark" if MODE == "light" else "◐  Light", self._toggle_theme,
            C_CARD2, C_TXT1, font_size=8, bold=True)
        self.theme_btn.pack(side=tk.RIGHT, padx=14, pady=12, ipady=5, ipadx=10)

        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill=tk.X)

    def _toggle_theme(self):
        """Switch theme and relaunch the picker so every widget picks up the new colors."""
        toggle_mode()
        try:
            subprocess.Popen([self.python_exe, self.entry_script])
        except OSError as e:
            messagebox.showerror("Theme", f"Theme saved, but Jelibox could not restart:\n{e}", parent=self.root)
            return
        self.root.destroy()

    def _btn(self, parent, text, command, bg, fg, font_size=9, bold=False):
        weight = 'bold' if bold else 'normal'
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg, font=('Segoe UI', font_size, weight),
            relief=tk.FLAT, cursor='hand2',
            activebackground=bg, activeforeground=fg, borderwidth=0
        )

    # ----------------------------------------------------------
    #  Body: sidebar (explorer tree) + welcome pane
    # ----------------------------------------------------------
    def _build_body(self):
        body = tk.Frame(self.root, bg=C_BASE)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ===== Sidebar (EXPLORER) =====
        sidebar = tk.Frame(body, bg=C_PANEL, width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="EXPLORER", bg=C_PANEL, fg=C_TXT3,
                 font=('Segoe UI', 8, 'bold'), anchor='w'
                 ).pack(fill=tk.X, padx=14, pady=(14, 2))
        tk.Label(sidebar, text="📁  WORKSPACES", bg=C_PANEL, fg=C_TXT2,
                 font=('Segoe UI', 9, 'bold'), anchor='w'
                 ).pack(fill=tk.X, padx=14, pady=(4, 8))

        tree_container = tk.Frame(sidebar, bg=C_PANEL)
        tree_container.pack(fill=tk.BOTH, expand=True)

        self.tree_canvas = canvas = tk.Canvas(tree_container, bg=C_PANEL, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(tree_container, orient='vertical', command=canvas.yview,
                                  bg=C_PANEL, troughcolor=C_PANEL, activebackground=C_BORDER,
                                  highlightthickness=0, bd=0, width=10)
        self.tree_frame = tk.Frame(canvas, bg=C_PANEL)

        tree_window = canvas.create_window((0, 0), window=self.tree_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        self.tree_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(tree_window, width=e.width)
        )

        def _on_mousewheel(event):
            bbox = canvas.bbox("all")
            if bbox is None or (bbox[3] - bbox[1]) <= canvas.winfo_height():
                return  # everything already fits - nothing to scroll
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(event=None):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(event=None):
            canvas.unbind_all("<MouseWheel>")

        # Bound on the sidebar (not the canvas) so the wheel stays captured
        # while the pointer moves across the rows/labels packed inside the
        # canvas - those are separate child widgets, so hovering over them
        # fires Enter/Leave on themselves, not on the canvas. Enter/Leave on
        # an ancestor only fires when the pointer actually crosses *that*
        # widget's own boundary, so this doesn't get interrupted by moving
        # between descendants.
        sidebar.bind("<Enter>", _bind_wheel)
        sidebar.bind("<Leave>", _unbind_wheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Frame(body, bg=C_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # ===== Welcome / empty editor area =====
        self.welcome = tk.Frame(body, bg=C_BASE)
        self.welcome.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._render_welcome()

    def _render_welcome(self):
        for child in self.welcome.winfo_children():
            child.destroy()

        center = tk.Frame(self.welcome, bg=C_BASE)
        center.place(relx=0.5, rely=0.42, anchor='center')

        try:
            from PIL import Image, ImageTk
            logo_img = Image.open("assets/jelibox.png").resize((112, 112))
            self.welcome_logo = ImageTk.PhotoImage(logo_img)
            tk.Label(center, image=self.welcome_logo, bg=C_BASE).pack(pady=(0, 18))
        except Exception:
            pass

        tk.Label(center, text="Jelibox", bg=C_BASE, fg=C_TXT1,
                 font=('Segoe UI', 30, 'bold')).pack()
        tk.Label(center, text="Soft to use. Sharp on every object.", bg=C_BASE, fg=C_TXT2,
                 font=('Segoe UI', 12)).pack(pady=(4, 24))
        tk.Label(center, text="Select a workspace on the left, then pick a dataset\n"
                              "to start annotating.",
                 bg=C_BASE, fg=C_TXT3, font=('Segoe UI', 9), justify=tk.CENTER
                 ).pack()

        tk.Label(self.welcome, text=f"Datasets are read from: {DATASETS_ROOT}",
                 bg=C_BASE, fg=C_TXT3, font=('Segoe UI', 8)
                 ).pack(side=tk.BOTTOM, pady=10)

    # ----------------------------------------------------------
    #  Workspace tree
    # ----------------------------------------------------------
    def refresh_workspaces(self):
        self.workspaces = list_workspaces()
        for child in self.tree_frame.winfo_children():
            child.destroy()

        if not self.workspaces:
            tk.Label(self.tree_frame, bg=C_PANEL, fg=C_TXT3, justify=tk.LEFT,
                     wraplength=280, font=('Segoe UI', 9),
                     text="No workspaces found.\n\n"
                          "Add dataset folders under datasetsInput "
                          "(e.g. weapon-1, weapon-2) and click Refresh."
                     ).pack(anchor='w', padx=14, pady=10)
            self.tree_canvas.yview_moveto(0)
            return

        for name, instances in self.workspaces.items():
            self._add_workspace_row(name, instances)

        # The tree is fully torn down and rebuilt on every refresh (expand,
        # collapse, or Refresh click), which can leave the canvas scrolled to
        # a position that no longer matches the new content. Snap back to
        # the top rather than showing a stale/out-of-range scroll offset.
        self.tree_canvas.update_idletasks()
        self.tree_canvas.configure(scrollregion=self.tree_canvas.bbox("all"))
        self.tree_canvas.yview_moveto(0)

    def _add_workspace_row(self, name, instances):
        is_expanded = name in self.expanded

        row = tk.Frame(self.tree_frame, bg=C_PANEL, cursor='hand2')
        row.pack(fill=tk.X)

        caret = tk.Label(row, text=('▾' if is_expanded else '▸'), bg=C_PANEL,
                          fg=C_TXT2, font=('Segoe UI', 9), width=2)
        caret.pack(side=tk.LEFT, padx=(10, 0), pady=8)

        label = tk.Label(row, text=f"📁 {name}", bg=C_PANEL, fg=C_TXT1,
                          font=('Segoe UI', 10, 'bold'), anchor='w')
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)

        count = tk.Label(row, text=str(len(instances)), bg=C_PANEL, fg=C_TXT3,
                          font=('Segoe UI', 8))
        count.pack(side=tk.RIGHT, padx=(2, 10))

        add_btn = tk.Label(row, text="+", bg=C_PANEL, fg=C_TXT3,
                            font=('Segoe UI', 11, 'bold'), width=2, cursor='hand2')
        add_btn.pack(side=tk.RIGHT, padx=0)

        del_btn = tk.Label(row, text="🗑", bg=C_PANEL, fg=C_TXT3,
                            font=('Segoe UI', 9), width=2, cursor='hand2')
        del_btn.pack(side=tk.RIGHT, padx=0)

        def toggle(event=None, n=name):
            if n in self.expanded:
                self.expanded.remove(n)
            else:
                self.expanded.add(n)
            self.refresh_workspaces()

        def add_instance_click(event=None, n=name):
            self._open_add_workspace_dialog(existing_workspace=n)

        def delete_click(event=None, n=name):
            self._confirm_delete_workspace(n)

        def on_enter(event=None, widgets=(row, caret, label, count)):
            for w in widgets:
                w.config(bg=C_CARD2)
            add_btn.config(bg=C_CARD2)
            del_btn.config(bg=C_CARD2)

        def on_leave(event=None, widgets=(row, caret, label, count)):
            for w in widgets:
                w.config(bg=C_PANEL)
            add_btn.config(bg=C_PANEL)
            del_btn.config(bg=C_PANEL)

        def add_btn_enter(event=None):
            add_btn.config(bg=C_ACCENT, fg=C_ON_ACCENT)

        def add_btn_leave(event=None):
            add_btn.config(bg=C_CARD2, fg=C_TXT3)

        def del_btn_enter(event=None):
            del_btn.config(bg=C_RED, fg=C_ON_RED)

        def del_btn_leave(event=None):
            del_btn.config(bg=C_CARD2, fg=C_TXT3)

        for widget in (row, caret, label, count):
            widget.bind("<Button-1>", toggle)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        # The "+" and delete icons have their own click/hover behavior - kept
        # out of the loop above so clicking them doesn't also toggle the
        # row's expand/collapse.
        add_btn.bind("<Button-1>", add_instance_click)
        add_btn.bind("<Enter>", add_btn_enter)
        add_btn.bind("<Leave>", add_btn_leave)

        del_btn.bind("<Button-1>", delete_click)
        del_btn.bind("<Enter>", del_btn_enter)
        del_btn.bind("<Leave>", del_btn_leave)

        if is_expanded:
            for instance in instances:
                self._add_instance_row(instance)

    def _add_instance_row(self, instance_name):
        n_images = count_images(instance_name)

        row = tk.Frame(self.tree_frame, bg=C_PANEL, cursor='hand2')
        row.pack(fill=tk.X)

        label = tk.Label(row, text=f"      📄 {instance_name}", bg=C_PANEL, fg=C_TXT2,
                          font=('Segoe UI', 9), anchor='w')
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=6)

        hint = tk.Label(row, text=f"{n_images} img", bg=C_PANEL, fg=C_TXT3,
                         font=('Segoe UI', 8))
        hint.pack(side=tk.RIGHT, padx=10)

        del_btn = tk.Label(row, text="🗑", bg=C_PANEL, fg=C_TXT3,
                            font=('Segoe UI', 9), width=2, cursor='hand2')
        del_btn.pack(side=tk.RIGHT, padx=0)

        def on_enter(event=None):
            row.config(bg=C_CARD)
            label.config(bg=C_CARD, fg=C_ACCENT)
            hint.config(bg=C_CARD, fg=C_TXT2)
            del_btn.config(bg=C_CARD)

        def on_leave(event=None):
            row.config(bg=C_PANEL)
            label.config(bg=C_PANEL, fg=C_TXT2)
            hint.config(bg=C_PANEL, fg=C_TXT3)
            del_btn.config(bg=C_PANEL, fg=C_TXT3)

        def open_it(event=None, n=instance_name):
            self.open_instance(n)

        def delete_click(event=None, n=instance_name):
            self._confirm_delete_instance(n)

        def del_btn_enter(event=None):
            del_btn.config(bg=C_RED, fg=C_ON_RED)

        def del_btn_leave(event=None):
            del_btn.config(bg=C_CARD, fg=C_TXT3)

        for widget in (row, label, hint):
            widget.bind("<Button-1>", open_it)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        del_btn.bind("<Button-1>", delete_click)
        del_btn.bind("<Enter>", del_btn_enter)
        del_btn.bind("<Leave>", del_btn_leave)

    # ----------------------------------------------------------
    #  Launching the annotation GUI as a separate process
    # ----------------------------------------------------------
    def open_instance(self, instance_name):
        if self.process is not None and self.process.poll() is None:
            return

        folder = instance_path(instance_name)
        if not self._launch(folder):
            return

        self.root.withdraw()
        self._poll_process()

    def _launch(self, folder):
        self._current_folder = folder
        try:
            self.process = subprocess.Popen([self.python_exe, self.entry_script, folder],
                                            env={**os.environ, "JELIBOX_PARENT": "1"})
        except OSError as e:
            messagebox.showerror("Failed to Open Workspace", str(e), parent=self.root)
            return False
        return True

    def _poll_process(self):
        if self.process is not None and self.process.poll() is None:
            self.root.after(400, self._poll_process)
            return

        if self.process is not None and self.process.returncode == 75 and getattr(self, "_current_folder", None):
            # The window asked to be relaunched (theme change): reopen the same workspace
            # without flashing the picker.
            if self._launch(self._current_folder):
                self.root.after(400, self._poll_process)
                return

        self.process = None
        self.refresh_workspaces()
        self.root.deiconify()
        self.root.lift()

    def on_close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno(
                "Close Jelibox",
                "An annotation window is still open. Close Jelibox anyway?",
                parent=self.root
            ):
                return
            try:
                self.process.terminate()
            except Exception:
                pass
        self.root.destroy()

    # ----------------------------------------------------------
    #  Add Workspace / Add Instance dialog
    # ----------------------------------------------------------
    def _open_add_workspace_dialog(self, existing_workspace=None):
        """
        existing_workspace=None  -> full "Add Workspace" form (name + folder + classes)
        existing_workspace="foo" -> "Add Instance" form (folder only, reuses foo's classes)
        """
        dialog = tk.Toplevel(self.root)
        dialog.configure(bg=C_BASE)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        width = 480
        hdr = tk.Frame(dialog, bg=C_ACCENT, height=48)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        title_text = ("➕  Add Workspace" if existing_workspace is None
                      else f"➕  Add Instance to “{existing_workspace}”")
        dialog.title("Add Workspace" if existing_workspace is None else f"Add Instance — {existing_workspace}")
        tk.Label(hdr, text=title_text, font=('Segoe UI', 12, 'bold'),
                 bg=C_ACCENT, fg=C_ON_ACCENT).pack(side=tk.LEFT, padx=16, pady=10)

        form = tk.Frame(dialog, bg=C_BASE, padx=24, pady=18)
        form.pack(fill=tk.BOTH, expand=True)

        # Width-spacer trick: a childless widget's requested size is exactly
        # its configured width/height, so this alone forces form (and thus
        # the dialog) to be at least `width` wide, while letting form's
        # height keep auto-fitting whatever fields end up packed below it.
        # (Disabling pack_propagate would lock BOTH dimensions to whatever
        # form's size happened to be at that moment - here, before any of
        # its content existed - which is what was hiding the whole form.)
        tk.Frame(form, bg=C_BASE, width=width, height=1).pack(side=tk.TOP)

        def section_label(text):
            tk.Label(form, text=text, font=('Segoe UI', 8, 'bold'),
                     bg=C_BASE, fg=C_TXT2, anchor='w').pack(fill=tk.X, pady=(14, 4))

        def hint_label(text):
            tk.Label(form, text=text, font=('Segoe UI', 8), bg=C_BASE, fg=C_TXT3,
                     anchor='w', justify=tk.LEFT, wraplength=width - 48).pack(fill=tk.X, pady=(3, 0))

        def status_label():
            lbl = tk.Label(form, text="", font=('Segoe UI', 8, 'bold'), bg=C_BASE,
                            fg=C_TXT3, anchor='w', justify=tk.LEFT, wraplength=width - 48)
            lbl.pack(fill=tk.X, pady=(3, 0))
            return lbl

        def set_status(lbl, text, color):
            lbl.config(text=text, fg=color)

        # ---- state shared with on_create() ----
        state = {'folder': None, 'image_count': 0}

        # ======== NAME (only for a brand-new workspace) ========
        name_var = tk.StringVar()
        name_status = None
        if existing_workspace is None:
            section_label("WORKSPACE NAME")
            name_entry = tk.Entry(form, textvariable=name_var, font=('Segoe UI', 10),
                                   bg=C_CARD2, fg=C_TXT1, insertbackground=C_ACCENT,
                                   relief=tk.FLAT)
            name_entry.pack(fill=tk.X, ipady=6)
            hint_label('Groups instances together - "weapon-1" and "weapon-2" both '
                       'belong to the "weapon" workspace.')
            name_status = status_label()
            name_entry.focus_set()

            def on_name_change(*_):
                raw = name_var.get().strip()
                if not raw:
                    set_status(name_status, "", C_TXT3)
                    render_classes_section(None)
                    return
                existing = existing_classes_for_workspace(raw) if raw.replace("_", "").replace("-", "").isalnum() else None
                if existing is not None:
                    preview = ", ".join(existing) if existing else "(no classes yet)"
                    set_status(name_status,
                              f"ℹ Existing workspace - a new instance will be added, "
                              f"reusing its classes: {preview}",
                              C_ACCENT)
                else:
                    set_status(name_status, "✓ New workspace", C_GREEN)
                render_classes_section(existing)

            # render_classes_section is defined further down (classes section
            # is built after this), but only ever called once typing starts -
            # by then the whole dialog has finished building.
            name_var.trace_add('write', on_name_change)

        # ======== IMAGE FOLDER ========
        section_label("IMAGE FOLDER")
        folder_row = tk.Frame(form, bg=C_BASE)
        folder_row.pack(fill=tk.X)
        folder_var = tk.StringVar(value="(no folder selected)")
        folder_display = tk.Label(folder_row, textvariable=folder_var, bg=C_CARD2, fg=C_TXT2,
                                   font=('Segoe UI', 9), anchor='w', padx=8, pady=7)
        folder_display.pack(side=tk.LEFT, fill=tk.X, expand=True)
        folder_status = status_label()

        def browse_folder():
            chosen = filedialog.askdirectory(title="Select folder with images", parent=dialog)
            if not chosen:
                return
            state['folder'] = chosen
            folder_var.set(chosen)
            if not os.path.isdir(chosen):
                set_status(folder_status, "✗ Not a folder.", C_RED)
                state['image_count'] = 0
                return
            n = count_images_in_folder(chosen)
            state['image_count'] = n
            if n == 0:
                set_status(folder_status,
                          "✗ No images found in this folder. Pick a folder that "
                          "directly contains .jpg/.png/... files.", C_RED)
            else:
                set_status(folder_status, f"✓ {n} image(s) found", C_GREEN)

        self._btn(folder_row, "📁  Browse...", browse_folder, C_CARD2, C_TXT1,
                  font_size=9).pack(side=tk.LEFT, padx=(8, 0), ipady=6, ipadx=8)

        # ======== CLASSES ========
        # A workspace's classes are shared across every one of its instances,
        # so this section switches between an editable entry (brand-new
        # workspace) and a read-only preview (existing workspace) - reacting
        # live to the name field when adding via the global "Add Workspace"
        # button, since the typed name might match one that already exists.
        classes_var = tk.StringVar()
        classes_ui = {'status': None, 'shown_key': object()}  # sentinel: nothing rendered yet

        section_label("CLASSES")
        classes_section = tk.Frame(form, bg=C_BASE)
        classes_section.pack(fill=tk.X)

        def render_classes_section(reuse_classes):
            # Only rebuild when the mode/content actually changes - typing in
            # the name field fires this on every keystroke, and tearing down
            # a live Entry each time would drop cursor position/focus even
            # though the typed classes text itself (bound via classes_var)
            # survives the rebuild.
            key = tuple(reuse_classes) if reuse_classes is not None else None
            if classes_ui['shown_key'] == key:
                return
            classes_ui['shown_key'] = key

            for child in classes_section.winfo_children():
                child.destroy()
            classes_ui['status'] = None

            if reuse_classes is not None:
                preview = ", ".join(reuse_classes) if reuse_classes else "(no classes defined yet)"
                tk.Label(classes_section, text=preview, font=('Segoe UI', 9, 'bold'),
                         bg=C_BASE, fg=C_TXT1, anchor='w', wraplength=width - 48,
                         justify=tk.LEFT).pack(fill=tk.X, ipady=4)
                tk.Label(classes_section,
                         text="This workspace already has these classes - shared across "
                              "every instance, so they can't be changed here.",
                         font=('Segoe UI', 8), bg=C_BASE, fg=C_TXT3, anchor='w',
                         wraplength=width - 48, justify=tk.LEFT).pack(fill=tk.X, pady=(3, 0))
            else:
                entry = tk.Entry(classes_section, textvariable=classes_var, font=('Segoe UI', 10),
                                  bg=C_CARD2, fg=C_TXT1, insertbackground=C_ACCENT, relief=tk.FLAT)
                entry.pack(fill=tk.X, ipady=6)
                tk.Label(classes_section,
                         text="Comma-separated names inside curly braces, e.g. {cat, dog, rabbit}.",
                         font=('Segoe UI', 8), bg=C_BASE, fg=C_TXT3, anchor='w',
                         wraplength=width - 48, justify=tk.LEFT).pack(fill=tk.X, pady=(3, 0))
                status = tk.Label(classes_section, text="", font=('Segoe UI', 8, 'bold'),
                                   bg=C_BASE, fg=C_TXT3, anchor='w', wraplength=width - 48,
                                   justify=tk.LEFT)
                status.pack(fill=tk.X, pady=(3, 0))
                classes_ui['status'] = status

        render_classes_section(
            existing_classes_for_workspace(existing_workspace) if existing_workspace is not None else None
        )

        # ======== Progress view (shown in place of the form while copying) ========
        progress_frame = tk.Frame(dialog, bg=C_BASE, padx=24, pady=24)
        progress_status_var = tk.StringVar(value="Copying images...")
        tk.Label(progress_frame, textvariable=progress_status_var, bg=C_BASE, fg=C_TXT1,
                 font=('Segoe UI', 10)).pack(pady=(10, 14))
        bar_w = width - 48
        bar_bg = tk.Frame(progress_frame, bg=C_PANEL, width=bar_w, height=8)
        bar_bg.pack()
        bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=C_ACCENT, width=0, height=8)
        bar_fill.place(x=0, y=0, relheight=1)

        def on_copy_progress(done, total):
            progress_status_var.set(f"Copying images... {done}/{total}")
            frac = (done / total) if total else 1.0
            bar_fill.place(width=int(bar_w * frac))
            dialog.update()

        # ======== Buttons ========
        btn_row = tk.Frame(form, bg=C_BASE)
        btn_row.pack(fill=tk.X, pady=(18, 0))

        self._btn(btn_row, "✕  Cancel", dialog.destroy, C_CARD2, C_TXT2,
                  font_size=10).pack(side=tk.LEFT, ipady=8, ipadx=12)

        create_label = "＋  Add Instance" if existing_workspace else "＋  Create Workspace"

        def on_create():
            # ---- validate name ----
            target_name = existing_workspace
            classes_to_write = None

            if existing_workspace is None:
                try:
                    target_name = sanitize_workspace_name(name_var.get())
                except ValueError as e:
                    set_status(name_status, f"✗ {e}", C_RED)
                    return

            # ---- validate folder ----
            if not state['folder']:
                set_status(folder_status, "✗ Choose a folder first.", C_RED)
                return
            if not os.path.isdir(state['folder']):
                set_status(folder_status, "✗ Not a folder.", C_RED)
                return
            if state['image_count'] == 0:
                set_status(folder_status, "✗ No images found in this folder.", C_RED)
                return

            # ---- validate classes (only when this will define a new workspace) ----
            # Not a hard blocker: an empty or malformed entry just means the
            # workspace is created without predefined classes - classes can
            # always be added afterwards from inside the workspace.
            reusing_existing = existing_workspace is not None or \
                existing_classes_for_workspace(target_name) is not None
            if not reusing_existing:
                raw_classes = classes_var.get().strip()
                if raw_classes:
                    try:
                        classes_to_write = parse_classes_input(raw_classes)
                    except ValueError:
                        proceed = messagebox.askyesno(
                            "Classes Format Not Recognized",
                            "Your Classes input doesn't match the expected format "
                            "({cat, dog, rabbit}).\n\n"
                            "You can still create this workspace now and add classes "
                            "to it afterwards. Continue without setting classes?",
                            parent=dialog
                        )
                        if not proceed:
                            return
                        classes_to_write = None

            # ---- do the copy, with progress ----
            form.pack_forget()
            progress_frame.pack(fill=tk.BOTH, expand=True)
            dialog.update()

            try:
                instance_name, n = create_workspace_instance(
                    target_name, state['folder'],
                    classes=classes_to_write, progress_cb=on_copy_progress
                )
            except (ValueError, OSError) as e:
                dialog.destroy()
                messagebox.showerror("Couldn't Add Workspace", str(e), parent=self.root)
                return

            dialog.destroy()
            self.expanded.add(target_name)
            self.refresh_workspaces()
            messagebox.showinfo(
                "Workspace Ready",
                f'"{instance_name}" created with {n} image(s).',
                parent=self.root
            )

        self._btn(btn_row, create_label, on_create, C_ACCENT, C_ON_ACCENT,
                  font_size=10, bold=True).pack(side=tk.RIGHT, ipady=8, ipadx=16)

        dialog.bind('<Escape>', lambda e: dialog.destroy())

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

    # ----------------------------------------------------------
    #  Shared progress popup (used by delete, which can touch a lot of files)
    # ----------------------------------------------------------
    def _show_progress_popup(self, title):
        """Small centered popup with a status line + progress bar.
        Returns (popup, update_fn) where update_fn(done, total, label=None)
        advances it - call popup.destroy() when the work is finished."""
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.configure(bg=C_BASE)
        popup.transient(self.root)
        popup.grab_set()
        popup.resizable(False, False)
        popup.protocol("WM_DELETE_WINDOW", lambda: None)  # can't cancel mid-delete

        width = 420
        tk.Frame(popup, bg=C_BASE, width=width, height=1).pack(side=tk.TOP)

        status_var = tk.StringVar(value=title)
        tk.Label(popup, textvariable=status_var, bg=C_BASE, fg=C_TXT1,
                 font=('Segoe UI', 10)).pack(pady=(20, 14))

        bar_w = width - 48
        bar_bg = tk.Frame(popup, bg=C_PANEL, width=bar_w, height=6)
        bar_bg.pack(pady=(0, 20))
        bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=C_RED, width=0, height=6)
        bar_fill.place(x=0, y=0, relheight=1)

        def update_fn(done, total, label=None):
            if label:
                status_var.set(label)
            frac = (done / total) if total else 1.0
            bar_fill.place(width=int(bar_w * frac))
            popup.update()

        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")
        popup.update()

        return popup, update_fn

    # ----------------------------------------------------------
    #  Delete instance
    # ----------------------------------------------------------
    def _confirm_delete_instance(self, instance_name):
        n_images = count_images(instance_name)
        proceed = messagebox.askyesno(
            "Delete Instance",
            f'Permanently delete "{instance_name}"?\n\n'
            f'This removes {n_images} image(s) from datasetsInput, plus any VOC '
            f'and YOLO annotation data that belongs only to those images.\n\n'
            f'This cannot be undone.',
            icon='warning',
            parent=self.root
        )
        if not proceed:
            return

        popup, update = self._show_progress_popup(f'Deleting "{instance_name}"...')
        try:
            delete_instance(instance_name, progress_cb=update)
        except (ValueError, OSError) as e:
            popup.destroy()
            messagebox.showerror("Delete Failed", str(e), parent=self.root)
            return

        popup.destroy()
        self.refresh_workspaces()

    # ----------------------------------------------------------
    #  Delete workspace (all instances + shared annotation data)
    # ----------------------------------------------------------
    def _confirm_delete_workspace(self, workspace_name):
        stats = workspace_stats(workspace_name)

        dialog = tk.Toplevel(self.root)
        dialog.configure(bg=C_BASE)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.title(f"Delete Workspace — {workspace_name}")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        width = 460
        hdr = tk.Frame(dialog, bg=C_RED, height=48)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"⚠  Delete Workspace “{workspace_name}”",
                 font=('Segoe UI', 12, 'bold'), bg=C_RED, fg=C_ON_RED
                 ).pack(side=tk.LEFT, padx=16, pady=10)

        body = tk.Frame(dialog, bg=C_BASE, padx=24, pady=18)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Frame(body, bg=C_BASE, width=width - 48, height=1).pack(side=tk.TOP)

        tk.Label(body, text="This permanently deletes:", font=('Segoe UI', 9, 'bold'),
                 bg=C_BASE, fg=C_TXT1, anchor='w').pack(fill=tk.X, pady=(0, 8))

        bullets = [
            f"{len(stats['instances'])} instance(s): {', '.join(stats['instances'])}",
            f"{stats['image_count']} image(s) from datasetsInput",
            f"{stats['xml_count']} VOC XML annotation(s)",
            f"{stats['label_count']} YOLO label(s)",
        ]
        if stats['has_config']:
            bullets.append("its class configuration")

        for item in bullets:
            tk.Label(body, text=f"•  {item}", font=('Segoe UI', 9), bg=C_BASE, fg=C_TXT2,
                     anchor='w', wraplength=width - 48, justify=tk.LEFT).pack(fill=tk.X, pady=1)

        tk.Label(body, text="Trained models and exported datasets are not affected.",
                 font=('Segoe UI', 8), bg=C_BASE, fg=C_TXT3, anchor='w',
                 wraplength=width - 48, justify=tk.LEFT).pack(fill=tk.X, pady=(10, 0))

        tk.Label(body, text="This cannot be undone.", font=('Segoe UI', 9, 'bold'),
                 bg=C_BASE, fg=C_RED, anchor='w').pack(fill=tk.X, pady=(10, 14))

        tk.Label(body, text=f'Type "{workspace_name}" to confirm:', font=('Segoe UI', 8, 'bold'),
                 bg=C_BASE, fg=C_TXT2, anchor='w').pack(fill=tk.X, pady=(0, 4))

        confirm_var = tk.StringVar()
        confirm_entry = tk.Entry(body, textvariable=confirm_var, font=('Segoe UI', 10),
                                  bg=C_CARD2, fg=C_TXT1, insertbackground=C_ACCENT, relief=tk.FLAT)
        confirm_entry.pack(fill=tk.X, ipady=6)
        confirm_entry.focus_set()

        btn_row = tk.Frame(body, bg=C_BASE)
        btn_row.pack(fill=tk.X, pady=(18, 0))

        self._btn(btn_row, "Cancel", dialog.destroy, C_CARD2, C_TXT2,
                  font_size=10).pack(side=tk.LEFT, ipady=8, ipadx=12)

        delete_btn = self._btn(btn_row, "🗑  Delete Forever", lambda: None,
                               C_CARD2, C_TXT3, font_size=10, bold=True)
        delete_btn.config(state=tk.DISABLED, cursor='')
        delete_btn.pack(side=tk.RIGHT, ipady=8, ipadx=16)

        def on_confirm_change(*_):
            if confirm_var.get() == workspace_name:
                delete_btn.config(state=tk.NORMAL, bg=C_RED, fg=C_ON_RED,
                                  activebackground=C_RED, activeforeground=C_ON_RED,
                                  cursor='hand2')
            else:
                delete_btn.config(state=tk.DISABLED, bg=C_CARD2, fg=C_TXT3,
                                  activebackground=C_CARD2, activeforeground=C_TXT3,
                                  cursor='')

        confirm_var.trace_add('write', on_confirm_change)

        def do_delete():
            if confirm_var.get() != workspace_name:
                return
            dialog.destroy()

            popup, update = self._show_progress_popup(f'Deleting "{workspace_name}"...')
            try:
                delete_workspace(workspace_name, progress_cb=update)
            except OSError as e:
                popup.destroy()
                messagebox.showerror("Delete Failed", str(e), parent=self.root)
                return

            popup.destroy()
            self.expanded.discard(workspace_name)
            self.refresh_workspaces()
            messagebox.showinfo("Workspace Deleted",
                                f'"{workspace_name}" has been deleted.', parent=self.root)

        delete_btn.config(command=do_delete)
        confirm_entry.bind('<Return>', lambda e: do_delete())
        dialog.bind('<Escape>', lambda e: dialog.destroy())

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

    # ----------------------------------------------------------
    #  Import Dataset (any folder layout, VOC / YOLO / COCO / images-only)
    # ----------------------------------------------------------
    def _open_import_dataset_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.configure(bg=C_BASE)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.title("Import Dataset")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        width = 480
        hdr = tk.Frame(dialog, bg=C_ACCENT, height=48)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬇  Import Dataset", font=('Segoe UI', 12, 'bold'),
                 bg=C_ACCENT, fg=C_ON_ACCENT).pack(side=tk.LEFT, padx=16, pady=10)

        form = tk.Frame(dialog, bg=C_BASE, padx=24, pady=18)
        form.pack(fill=tk.BOTH, expand=True)
        tk.Frame(form, bg=C_BASE, width=width, height=1).pack(side=tk.TOP)

        def section_label(text):
            tk.Label(form, text=text, font=('Segoe UI', 8, 'bold'),
                     bg=C_BASE, fg=C_TXT2, anchor='w').pack(fill=tk.X, pady=(14, 4))

        def hint_label(text):
            tk.Label(form, text=text, font=('Segoe UI', 8), bg=C_BASE, fg=C_TXT3,
                     anchor='w', justify=tk.LEFT, wraplength=width - 48).pack(fill=tk.X, pady=(3, 0))

        def status_label():
            lbl = tk.Label(form, text="", font=('Segoe UI', 8, 'bold'), bg=C_BASE,
                            fg=C_TXT3, anchor='w', justify=tk.LEFT, wraplength=width - 48)
            lbl.pack(fill=tk.X, pady=(3, 0))
            return lbl

        def set_status(lbl, text, color):
            lbl.config(text=text, fg=color)

        state = {'folder': None, 'scan': None, 'format': 'unset', 'yolo_unnamed_classes': False}

        # ======== WORKSPACE NAME ========
        section_label("WORKSPACE NAME")
        name_var = tk.StringVar()
        name_entry = tk.Entry(form, textvariable=name_var, font=('Segoe UI', 10),
                              bg=C_CARD2, fg=C_TXT1, insertbackground=C_ACCENT, relief=tk.FLAT)
        name_entry.pack(fill=tk.X, ipady=6)
        hint_label('Groups instances together - "weapon-1" and "weapon-2" both '
                   'belong to the "weapon" workspace.')
        name_status = status_label()
        name_entry.focus_set()

        def on_name_change(*_):
            raw = name_var.get().strip()
            if not raw:
                set_status(name_status, "", C_TXT3)
                return
            existing = existing_classes_for_workspace(raw) if raw.replace("_", "").replace("-", "").isalnum() else None
            if existing is not None:
                preview = ", ".join(existing) if existing else "(no classes yet)"
                set_status(name_status,
                          f"ℹ Existing workspace - a new instance will be added, "
                          f"reusing its classes: {preview}", C_ACCENT)
            else:
                set_status(name_status, "✓ New workspace", C_GREEN)

        name_var.trace_add('write', on_name_change)

        # ======== DATASET FOLDER ========
        section_label("DATASET FOLDER")
        folder_row = tk.Frame(form, bg=C_BASE)
        folder_row.pack(fill=tk.X)
        folder_var = tk.StringVar(value="(no folder selected)")
        folder_display = tk.Label(folder_row, textvariable=folder_var, bg=C_CARD2, fg=C_TXT2,
                                   font=('Segoe UI', 9), anchor='w', padx=8, pady=7)
        folder_display.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scan_status = status_label()
        hint_label("Every subfolder is searched for images plus one annotation format - "
                   "Pascal VOC (.xml), YOLO (.txt), or COCO (.json).")

        # ======== FILENAME PREFIX (optional) ========
        section_label("FILENAME PREFIX (OPTIONAL)")
        prefix_var = tk.StringVar()
        prefix_entry = tk.Entry(form, textvariable=prefix_var, font=('Segoe UI', 10),
                                bg=C_CARD2, fg=C_TXT1, insertbackground=C_ACCENT, relief=tk.FLAT)
        prefix_entry.pack(fill=tk.X, ipady=6)
        prefix_status = status_label()
        hint_label('Renames every imported image (and its annotation) to keep them in '
                   'order - e.g. "test-" becomes test-1, test-2, test-3, ... '
                   'Leave blank to keep the original filenames.')

        def on_prefix_change(*_):
            raw = prefix_var.get().strip()
            if not raw:
                set_status(prefix_status, "", C_TXT3)
                return
            try:
                sanitize_filename_prefix(raw)
            except ValueError as e:
                set_status(prefix_status, f"✗ {e}", C_RED)
                return
            set_status(prefix_status, f"✓ Files will be renamed {raw}1, {raw}2, {raw}3, ...", C_GREEN)

        prefix_var.trace_add('write', on_prefix_change)

        def browse_folder():
            chosen = filedialog.askdirectory(title="Select dataset folder", parent=dialog)
            if not chosen:
                return
            state['folder'] = chosen
            state['scan'] = None
            state['format'] = 'unset'
            state['yolo_unnamed_classes'] = False
            folder_var.set(chosen)
            set_status(scan_status, "Scanning...", C_TXT3)
            dialog.update()

            scan = scan_dataset_folder(chosen)
            state['scan'] = scan
            n_images = len(scan['images'])

            if n_images == 0:
                state['format'] = None
                set_status(scan_status,
                          "✗ No images found in this folder (every subfolder was searched).", C_RED)
                return

            try:
                fmt = detect_dataset_format(scan)
            except ValueError as e:
                state['format'] = 'conflict'
                set_status(scan_status, f"✗ {e}", C_RED)
                return

            state['format'] = fmt
            if fmt is None:
                set_status(scan_status,
                          f"✓ {n_images} image(s) found - no annotations detected, "
                          f"will import as unannotated.", C_GREEN)
            elif fmt == 'yolo' and not yolo_classes_resolved(scan):
                state['yolo_unnamed_classes'] = True
                fmt_count = len(scan['yolo'])
                set_status(scan_status,
                          f"⚠ {n_images} image(s) and {fmt_count} YOLO annotation(s) found, "
                          f"but no class name file (classes.txt / obj.names / data.yaml) was "
                          f"found alongside them - classes will import as class_0, class_1, "
                          f"etc. instead of their real names. Add that file to the dataset "
                          f"folder and re-browse if you'd rather fix this first, or continue "
                          f"and rename classes later.", C_AMBER)
            else:
                fmt_label = {'voc': 'Pascal VOC', 'yolo': 'YOLO', 'coco': 'COCO'}[fmt]
                fmt_count = len(scan[fmt])
                set_status(scan_status,
                          f"✓ {n_images} image(s) and {fmt_count} {fmt_label} annotation(s) found.",
                          C_GREEN)

        self._btn(folder_row, "📁  Browse...", browse_folder, C_CARD2, C_TXT1,
                  font_size=9).pack(side=tk.LEFT, padx=(8, 0), ipady=6, ipadx=8)

        # ======== Progress view ========
        progress_frame = tk.Frame(dialog, bg=C_BASE, padx=24, pady=24)
        progress_status_var = tk.StringVar(value="Importing...")
        tk.Label(progress_frame, textvariable=progress_status_var, bg=C_BASE, fg=C_TXT1,
                 font=('Segoe UI', 10)).pack(pady=(10, 14))
        bar_w = width - 48
        bar_bg = tk.Frame(progress_frame, bg=C_PANEL, width=bar_w, height=8)
        bar_bg.pack()
        bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=C_ACCENT, width=0, height=8)
        bar_fill.place(x=0, y=0, relheight=1)

        def on_import_progress(done, total, label=None):
            if label:
                progress_status_var.set(label)
            frac = (done / total) if total else 1.0
            bar_fill.place(width=int(bar_w * frac))
            dialog.update()

        # ======== Buttons ========
        btn_row = tk.Frame(form, bg=C_BASE)
        btn_row.pack(fill=tk.X, pady=(18, 0))

        self._btn(btn_row, "✕  Cancel", dialog.destroy, C_CARD2, C_TXT2,
                  font_size=10).pack(side=tk.LEFT, ipady=8, ipadx=12)

        def on_import():
            try:
                target_name = sanitize_workspace_name(name_var.get())
            except ValueError as e:
                set_status(name_status, f"✗ {e}", C_RED)
                return

            if not state['folder'] or state['scan'] is None:
                set_status(scan_status, "✗ Choose a dataset folder first.", C_RED)
                return
            if state['format'] == 'conflict':
                return  # the specific error is already shown under the folder field
            if len(state['scan']['images']) == 0:
                set_status(scan_status, "✗ No images found in this folder.", C_RED)
                return

            try:
                prefix = sanitize_filename_prefix(prefix_var.get())
            except ValueError as e:
                set_status(prefix_status, f"✗ {e}", C_RED)
                return

            if state.get('yolo_unnamed_classes'):
                proceed = messagebox.askyesno(
                    "Class Names Not Found",
                    "No class name file (classes.txt, obj.names, or data.yaml) was found "
                    "for this YOLO dataset, so its classes will be imported as generic "
                    "names - class_0, class_1, and so on - instead of their real names.\n\n"
                    "You can rename them later from inside the workspace. Continue the "
                    "import now, or cancel and add that file to the dataset folder first?",
                    parent=dialog
                )
                if not proceed:
                    return

            form.pack_forget()
            progress_frame.pack(fill=tk.BOTH, expand=True)
            dialog.update()

            try:
                result = import_dataset(state['folder'], target_name, prefix=prefix,
                                        progress_cb=on_import_progress)
            except (ValueError, OSError) as e:
                dialog.destroy()
                messagebox.showerror("Import Failed", str(e), parent=self.root)
                return

            dialog.destroy()
            self.expanded.add(target_name)
            self.refresh_workspaces()

            classes_text = ", ".join(result['classes']) if result['classes'] else "(none)"
            messagebox.showinfo(
                "Dataset Imported",
                f'"{result["instance_name"]}" created with {result["image_count"]} image(s).\n\n'
                f'Annotation format: {result["format"] or "none"}\n'
                f'Annotated images: {result["annotated_count"]}\n'
                f'Classes: {classes_text}',
                parent=self.root
            )

        self._btn(btn_row, "⬇  Import Dataset", on_import, C_ACCENT, C_ON_ACCENT,
                  font_size=10, bold=True).pack(side=tk.RIGHT, ipady=8, ipadx=16)

        dialog.bind('<Escape>', lambda e: dialog.destroy())

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
