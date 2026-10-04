"""
Server settings: the servers this Jelibox knows about, plus this installation's device ID.

Groundwork only - it edits the local list and shows the status of each entry. Testing the
connection and requesting access arrive with the connection features; nothing here touches
the network.
"""
import tkinter as tk
from tkinter import messagebox

from .theme import (C_BASE, C_PANEL, C_CARD2, C_ACCENT, C_GREEN, C_RED, C_AMBER,
                    C_TXT1, C_TXT2, C_TXT3, C_ON_ACCENT, C_ON_RED)
from .collab import identity, servers

_STATUS_TEXT = {
    servers.CONNECTED: ("Connected", C_GREEN),
    servers.PENDING: ("Waiting for approval", C_AMBER),
    servers.NOT_CONNECTED: ("Not connected", C_TXT3),
}


class ServerSettingsDialog:
    WIDTH = 520

    def __init__(self, parent):
        self.parent = parent
        self.editing = None            # id of the entry loaded into the form, None = adding a new one

        self.win = tk.Toplevel(parent)
        self.win.title("Server settings")
        self.win.configure(bg=C_BASE)
        self.win.transient(parent)
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)
        self.win.bind("<Escape>", lambda e: self.win.destroy())

        hdr = tk.Frame(self.win, bg=C_ACCENT, height=48)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚙  Server settings", font=('Segoe UI', 12, 'bold'),
                 bg=C_ACCENT, fg=C_ON_ACCENT).pack(side=tk.LEFT, padx=16, pady=10)

        body = tk.Frame(self.win, bg=C_BASE, padx=24, pady=16)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Frame(body, bg=C_BASE, width=self.WIDTH, height=1).pack(side=tk.TOP)
        self.body = body

        self._section("SERVERS")
        self.list_frame = tk.Frame(body, bg=C_BASE)
        self.list_frame.pack(fill=tk.X)

        self.form_title = self._section("ADD A SERVER")
        self.name_var, self.url_var = tk.StringVar(), tk.StringVar()
        self.name_entry = self._entry("Name (optional)", self.name_var)
        self.url_entry = self._entry("Address, e.g. 192.168.1.10 or jelibox.local", self.url_var)
        self.status = tk.Label(body, text="", font=('Segoe UI', 8, 'bold'), bg=C_BASE, fg=C_TXT3,
                               anchor='w', justify=tk.LEFT, wraplength=self.WIDTH - 8)
        self.status.pack(fill=tk.X, pady=(4, 0))

        row = tk.Frame(body, bg=C_BASE)
        row.pack(fill=tk.X, pady=(10, 0))
        self.cancel_btn = self._btn(row, "Cancel edit", self._reset_form, C_CARD2, C_TXT2)
        self.save_btn = self._btn(row, "＋  Add server", self._save, C_ACCENT, C_ON_ACCENT, bold=True)
        self.save_btn.pack(side=tk.RIGHT, ipady=6, ipadx=14)

        self._section("THIS DEVICE")
        id_row = tk.Frame(body, bg=C_BASE)
        id_row.pack(fill=tk.X)
        self.device_id = identity.get_device_id()
        tk.Label(id_row, text=self.device_id, bg=C_CARD2, fg=C_TXT2, font=('Consolas', 9),
                 anchor='w', padx=8, pady=7).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._btn(id_row, "Copy", self._copy_device_id, C_CARD2, C_TXT1
                  ).pack(side=tk.LEFT, padx=(8, 0), ipady=6, ipadx=10)
        tk.Label(body, wraplength=self.WIDTH - 8, justify=tk.LEFT, anchor='w', bg=C_BASE, fg=C_TXT3,
                 font=('Segoe UI', 8),
                 text="Servers recognise this installation by this ID. Jelibox does not contact any "
                      "server yet - connecting arrives in a later version."
                 ).pack(fill=tk.X, pady=(6, 0))

        self.refresh()
        self.win.update_idletasks()
        x = self.parent.winfo_rootx() + max(0, (self.parent.winfo_width() - self.win.winfo_width()) // 2)
        y = self.parent.winfo_rooty() + 60
        self.win.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------ widgets
    def _section(self, text):
        lbl = tk.Label(self.body, text=text, font=('Segoe UI', 8, 'bold'), bg=C_BASE, fg=C_TXT2, anchor='w')
        lbl.pack(fill=tk.X, pady=(14, 4))
        return lbl

    def _entry(self, hint, var):
        tk.Label(self.body, text=hint, font=('Segoe UI', 8), bg=C_BASE, fg=C_TXT3, anchor='w').pack(fill=tk.X)
        entry = tk.Entry(self.body, textvariable=var, font=('Segoe UI', 10), bg=C_CARD2, fg=C_TXT1,
                         insertbackground=C_ACCENT, relief=tk.FLAT)
        entry.pack(fill=tk.X, ipady=6, pady=(2, 6))
        return entry

    @staticmethod
    def _btn(parent, text, command, bg, fg, bold=False):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, relief=tk.FLAT, cursor='hand2',
                         font=('Segoe UI', 9, 'bold' if bold else 'normal'), borderwidth=0,
                         activebackground=bg, activeforeground=fg)

    # ------------------------------------------------------------ list
    def refresh(self):
        for child in self.list_frame.winfo_children():
            child.destroy()
        entries = servers.list_servers()
        if not entries:
            tk.Label(self.list_frame, text="No servers yet.", bg=C_BASE, fg=C_TXT3,
                     font=('Segoe UI', 9), anchor='w').pack(fill=tk.X, pady=4)
        for entry in entries:
            self._row(entry)

    def _row(self, entry):
        row = tk.Frame(self.list_frame, bg=C_PANEL)
        row.pack(fill=tk.X, pady=2)
        text = tk.Frame(row, bg=C_PANEL)
        text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=6)
        tk.Label(text, text=entry["name"], bg=C_PANEL, fg=C_TXT1, font=('Segoe UI', 10, 'bold'),
                 anchor='w').pack(fill=tk.X)
        label, color = _STATUS_TEXT[servers.state(entry["id"])]
        tk.Label(text, text=f'{entry["url"]}   ·   {label}', bg=C_PANEL, fg=color, font=('Segoe UI', 8),
                 anchor='w').pack(fill=tk.X)
        self._btn(row, "Remove", lambda e=entry: self._remove(e), C_PANEL, C_RED
                  ).pack(side=tk.RIGHT, padx=(0, 8), ipady=4, ipadx=6)
        self._btn(row, "Edit", lambda e=entry: self._edit(e), C_PANEL, C_ACCENT
                  ).pack(side=tk.RIGHT, ipady=4, ipadx=6)

    # ------------------------------------------------------------ actions
    def _set_status(self, text, color):
        self.status.config(text=text, fg=color)

    def _reset_form(self):
        self.editing = None
        self.name_var.set("")
        self.url_var.set("")
        self.form_title.config(text="ADD A SERVER")
        self.save_btn.config(text="＋  Add server")
        self.cancel_btn.pack_forget()
        self._set_status("", C_TXT3)

    def _edit(self, entry):
        self.editing = entry["id"]
        self.name_var.set(entry["name"])
        self.url_var.set(entry["url"])
        self.form_title.config(text="EDIT SERVER")
        self.save_btn.config(text="Save changes")
        self.cancel_btn.pack(side=tk.LEFT, ipady=6, ipadx=12)
        self._set_status("", C_TXT3)
        self.url_entry.focus_set()

    def _save(self):
        try:
            if self.editing is None:
                servers.add_server(self.name_var.get(), self.url_var.get())
            else:
                servers.update_server(self.editing, name=self.name_var.get(), url=self.url_var.get())
        except ValueError as e:
            self._set_status(f"✗ {e}", C_RED)
            return
        self._reset_form()
        self.refresh()
        self._set_status("✓ Saved.", C_GREEN)

    def _remove(self, entry):
        if not messagebox.askyesno(
                "Remove server",
                f'Remove "{entry["name"]}"?\n\nThis also deletes the access key stored for it on this '
                f'computer. You would have to request access again.',
                icon='warning', parent=self.win):
            return
        servers.remove_server(entry["id"])
        if self.editing == entry["id"]:
            self._reset_form()
        self.refresh()

    def _copy_device_id(self):
        self.win.clipboard_clear()
        self.win.clipboard_append(self.device_id)
        self._set_status("✓ Device ID copied.", C_GREEN)


def open_dialog(parent):
    return ServerSettingsDialog(parent)
