"""
Main Annotation GUI Class
Handles all GUI rendering, user interaction, annotation management, and training
"""
import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import sys
import subprocess
import shutil

from .config import (CLASSLIST, state, input_folder, colorsPalette, vocdataset_folder,
                   class_manager, yolo_dataset_folder, yolo_labels_folder, model_path,
                   model_folder, export_model_folder, export_dataset_folder, workspaceName)
from .file_handler import load_annotation_local
from .polygon_manager import polygon_manager
from .inferenceObjectDetection import inference_current
from .image_manager import (repeat_last_annotations, delete_current_image, 
                          save_and_backup_bboxes)
from .TrainingConfigDialog import TrainingConfigDialog
from .LabelAssistantDialog import open_label_assistant
from . import export as export_module
import threading
import webbrowser
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import random
import time
import importlib

from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT,
                    C_PURPLE, C_ORANGE, C_GREEN, C_AMBER, C_RED, C_BLUE,
                    C_TXT1, C_TXT2, C_TXT3, C_ACCENT_TINT, C_ON_ACCENT, C_ON_RED,
                    C_DANGER_BG, C_DANGER_FG, C_STAGE, MODE, toggle_mode)

EPOCH = 5
BATCH = 4


class AnnotationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Jelibox — {os.path.basename(os.path.normpath(input_folder))}")
        self.root.geometry("1600x900")
        self.root.configure(bg=C_BASE)
        self.stream_process = None
        
        # Load images
        self.images = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.images.sort()
        
        if not self.images:
            messagebox.showerror("Error", f"No images found in {input_folder}")
            root.destroy()
            return
        
        # Variables
        self.current_img_pil = None
        self.canvas_image = None
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None
        self.bbox_rects = []
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        # === POLYGON SUPPORT ===
        self.polygon_drawing = False
        self.polygon_editing_point = False
        
        # === MASKING TOOL ===
        self.mask_mode = False
        self.mask_polygon_points = []
        self.masked_regions = []

        # === SHIFT SELECT BOX ===
        self.shift_selecting = False
        self.shift_start_x = 0
        self.shift_start_y = 0
        state.selected_bboxes = set()  # multi-select indices
        
        # === ZOOM FUNCTIONALITY ===
        self.zoom_min = 1.0  # Minimum zoom: 1x original size
        self.zoom_max = 5.0  # Maximum zoom: 5x
        self.zoom_step = 1.1  # Zoom increment per scroll
        self.canvas_container = None  # Store reference to container
        
        # Setup UI
        self.create_widgets()

        self.root.update_idletasks()
        self.root.after(100, self.initial_load)

        self.bind_shortcuts()

    # ----------------------------------------------------------
    #  Helper: create a flat styled button
    # ----------------------------------------------------------
    @staticmethod
    def _readable_fg(hex_color):
        """Dark or white text, whichever reads better on this fill."""
        h = hex_color.lstrip('#')
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
        return '#10131F' if (lum + 0.05) / 0.0553 >= 1.05 / (lum + 0.05) else '#FFFFFF'

    def _kbtn(self, parent, text, command, kind='secondary', font_size=8):
        """Hierarchy button: 'primary' (accent, the main action), 'secondary'
        (neutral, everything else) or 'danger' (destructive)."""
        if kind == 'primary':
            bg, fg, hover, ring = C_ACCENT, C_ON_ACCENT, C_ACCENT, C_ACCENT
        elif kind == 'danger':
            bg, fg, hover, ring = C_DANGER_BG, C_DANGER_FG, C_RED, C_RED
        else:
            bg, fg, hover, ring = C_CARD2, C_TXT1, C_BORDER, C_BORDER
        btn = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            font=('Segoe UI', font_size, 'bold'), relief=tk.FLAT, cursor='hand2',
            activebackground=hover, activeforeground=(C_ON_RED if kind == 'danger' else fg),
            highlightbackground=ring, highlightcolor=ring, highlightthickness=1,
            borderwidth=0)
        if kind != 'primary':
            hover_fg = C_ON_RED if kind == 'danger' else fg
            btn.bind('<Enter>', lambda e: btn['state'] == 'normal' and btn.config(bg=hover, fg=hover_fg))
            btn.bind('<Leave>', lambda e: btn['state'] == 'normal' and btn.config(bg=bg, fg=fg))
        return btn

    def _btn(self, parent, text, command, bg, fg, font_size=9, bold=False):
        """Factory for Jelibox-styled flat buttons."""
        weight = 'bold' if bold else 'normal'
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg,
            font=('Segoe UI', font_size, weight),
            relief=tk.FLAT, cursor='hand2',
            activebackground=bg, activeforeground=fg,
            borderwidth=0
        )

    # ----------------------------------------------------------
    #  Keyboard shortcuts
    # ----------------------------------------------------------
    def bind_shortcuts(self):
        """Bind all keyboard shortcuts"""
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind_all('<KeyPress>', self.handle_global_key)
        self.root.bind('r', lambda e: self.delete_selected_bbox())
        self.root.bind('s', lambda e: self.change_class_selected())
        self.root.bind('t', lambda e: self.start_training())
        self.root.bind('g', lambda e: self.run_inference())
        self.root.bind('e', lambda e: self.repeat_annotations())
        self.root.bind('b', lambda e: self.toggle_force_new_bbox())
        self.root.bind('<Delete>', lambda e: self.delete_image())
        self.root.bind('m', lambda e: self.toggle_annotation_mode())
        self.root.bind('<Escape>', lambda e: self.on_escape_pressed())
        self.root.bind('<Button-3>', lambda e: self.cancel_polygon_drawing())
        self.root.bind('<Return>', lambda e: self.on_return_pressed())
        
        for i in range(9):
            self.root.bind(str(i+1), lambda e, idx=i: self.select_class_by_number(idx))

    def handle_global_key(self, event):
        """Handle navigation keys regardless of which main widget has focus."""
        if event.widget.winfo_class() in ('Entry', 'TEntry', 'Spinbox', 'Text'):
            return

        key = event.keysym.lower()
        if key == 'a':
            self.prev_image()
            return 'break'
        if key == 'd':
            self.next_image()
            return 'break'

    def on_escape_pressed(self):
        if self.mask_mode:
            if len(self.mask_polygon_points) > 0:
                self.cancel_mask_polygon()
            else:
                self.toggle_mask_mode()
        else:
            self.close_main_gui()

    def on_return_pressed(self):
        if self.mask_mode:
            if len(self.mask_polygon_points) >= 3:
                self.finish_mask_polygon()
        elif state.annotation_mode == "polygon":
            if len(state.polygon_points_preview) >= 3:
                self.finish_polygon_drawing()

    # ----------------------------------------------------------
    #  Status toggle methods (UI state updates)
    # ----------------------------------------------------------
    def toggle_bbox_text(self):
        state.show_bbox_text = not state.show_bbox_text
        if state.show_bbox_text:
            self.text_btn.config(text="Hide label text")
        else:
            self.text_btn.config(text="Show label text")
        self.update_display()

    def update_force_label(self):
        if state.force_new_bbox:
            self.force_label.config(text="Force: ON", fg=C_AMBER, bg=C_CARD)
        else:
            self.force_label.config(text="Force: OFF", fg=C_TXT3, bg=C_CARD)

    def toggle_force_new_bbox(self):
        state.force_new_bbox = not state.force_new_bbox
        self.update_force_label()
        self.update_display()
        print(f"[GUI] Force new bbox: {'ON' if state.force_new_bbox else 'OFF'}")

    # ----------------------------------------------------------
    #  CREATE WIDGETS — main UI construction
    # ----------------------------------------------------------
    def create_widgets(self):

        # ======================================================
        # HEADER BAR  (brand + navigation + main tools)
        # ======================================================
        header = tk.Frame(self.root, bg=C_PANEL, height=54)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        # — Brand —
        logo_img = Image.open("assets/jelibox.png")
        logo_img = logo_img.resize((32, 32))  # sesuaikan ukuran
        self.logo = ImageTk.PhotoImage(logo_img)
        brand = tk.Frame(header, bg=C_PANEL)
        brand.pack(side=tk.LEFT, padx=(10, 0))

        self._kbtn(brand, "🏠", self.back_to_workspace, 'secondary', font_size=11).pack(
            side=tk.LEFT, padx=(0, 6), pady=12, ipady=2, ipadx=4)

        tk.Label(brand, image=self.logo, bg=C_PANEL).pack(side=tk.LEFT, pady=10)
        name_stack = tk.Frame(brand, bg=C_PANEL)
        name_stack.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(name_stack, text="Jelibox", bg=C_PANEL, fg=C_TXT1,
                 font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        tk.Label(name_stack, text=workspaceName.upper(), bg=C_PANEL, fg=C_TXT3,
                 font=('Segoe UI', 6, 'bold')).pack(anchor='w')

        # vertical separator
        tk.Frame(header, bg=C_BORDER, width=1).pack(side=tk.LEFT,
                                                      fill=tk.Y, pady=10, padx=10)

        # vertical separator
        tk.Frame(header, bg=C_BORDER, width=1).pack(side=tk.LEFT,
                                                      fill=tk.Y, pady=10, padx=8)

        # — Navigation —
        nav = tk.Frame(header, bg=C_PANEL)
        nav.pack(side=tk.LEFT)

        self._kbtn(nav, "◀  Prev", self.prev_image, 'secondary', font_size=9).pack(
            side=tk.LEFT, padx=(0, 2), pady=12, ipady=5, ipadx=10)

        self._kbtn(nav, "Next  ▶", self.next_image, 'secondary', font_size=9).pack(
            side=tk.LEFT, padx=(2, 0), pady=12, ipady=5, ipadx=10)

        tk.Frame(header, bg=C_BORDER, width=1).pack(side=tk.LEFT,
                                                      fill=tk.Y, pady=10, padx=10)

        # Configuration group (opens a dialog) - pinned to the top-right.
        # Packed before the quick actions so it is never the one that gets clipped.
        config_group = tk.Frame(header, bg=C_PANEL)
        config_group.pack(side=tk.RIGHT, padx=(0, 10))

        config_defs = [
            ("🧠  Label Assistant", self.show_label_assistant),
            ("🎓  Train",           self.start_training),
            ("⬆  Export Model",     self.show_export_dialog),
            ("⬆  Export Dataset",   self.show_export_dataset_dialog),
        ]
        for label, cmd in config_defs:
            self._kbtn(config_group, label, cmd, 'secondary').pack(
                side=tk.LEFT, padx=2, pady=12, ipady=5, ipadx=8)

        # Quick-action group (one click, no dialog) - next to PREV / NEXT
        tools = tk.Frame(header, bg=C_PANEL)
        tools.pack(side=tk.LEFT)

        quick_defs = [
            ("🔄  Repeat", self.repeat_annotations,     'secondary'),
            ("🤖  Infer",  self.run_inference,           'primary'),
            ("🗑  Delete", self.delete_image,            'danger'),
            ("🎥  Stream", self.launch_stream,           'secondary'),
            ("⬡  Mode",    self.toggle_annotation_mode,  'secondary'),
        ]
        for label, cmd, kind in quick_defs:
            self._kbtn(tools, label, cmd, kind).pack(
                side=tk.LEFT, padx=2, pady=12, ipady=5, ipadx=8)

        self.mask_btn = self._kbtn(tools, "⬛ ZeroFill: OFF", self.toggle_mask_mode, 'secondary')
        self.mask_btn.pack(side=tk.LEFT, padx=2, pady=12, ipady=5, ipadx=8)

        # Filler block: stretches to take whatever space is left between the
        # quick actions and the configuration group, so the header stays full
        # no matter how many buttons either group gets.
        tk.Frame(header, bg=C_CARD).pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                         padx=(10, 10), pady=12)

        # Hairline under header
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill=tk.X)

        # ======================================================
        # MAIN CONTENT AREA
        # ======================================================
        content = tk.Frame(self.root, bg=C_BASE)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ===== LEFT PANEL =====
        left_panel = tk.Frame(content, bg=C_PANEL, width=245)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        # Right border for left panel
        tk.Frame(content, bg=C_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # — Class Manager header —
        cls_hdr = tk.Frame(left_panel, bg=C_CARD, height=40)
        cls_hdr.pack(fill=tk.X)
        cls_hdr.pack_propagate(False)

        tk.Label(cls_hdr, text="📋  CLASSES", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8, 'bold')).pack(side=tk.LEFT, padx=10)

        btn_row = tk.Frame(cls_hdr, bg=C_CARD)
        btn_row.pack(side=tk.RIGHT, padx=6)

        self._kbtn(btn_row, " + ", self.add_class_dialog, 'primary', font_size=9).pack(
            side=tk.LEFT, padx=2, pady=8)

        self._kbtn(btn_row, " − ", self.delete_class_dialog, 'danger', font_size=9).pack(
            side=tk.LEFT, padx=2, pady=8)

        # — Class listbox —
        cls_box_frame = tk.Frame(left_panel, bg=C_PANEL)
        cls_box_frame.pack(fill=tk.X, padx=6, pady=(5, 0))

        cls_scroll = tk.Scrollbar(cls_box_frame, bg=C_BORDER, troughcolor=C_PANEL,
                                   relief=tk.FLAT, width=7)
        cls_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.class_listbox = tk.Listbox(
            cls_box_frame, bg=C_CARD, fg=C_TXT1,
            selectmode=tk.SINGLE, font=('Segoe UI', 9),
            yscrollcommand=cls_scroll.set, height=10,
            activestyle='none', relief=tk.FLAT,
            selectbackground=C_ACCENT, selectforeground=C_ON_ACCENT,
            highlightthickness=0, borderwidth=0
        )
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cls_scroll.config(command=self.class_listbox.yview)

        self.refresh_class_list()
        self.class_listbox.bind('<<ListboxSelect>>', self.on_class_select)

        # Divider
        tk.Frame(left_panel, bg=C_BORDER, height=1).pack(fill=tk.X, pady=(8, 0))

        # — Visibility header —
        vis_hdr = tk.Frame(left_panel, bg=C_CARD, height=40)
        vis_hdr.pack(fill=tk.X)
        vis_hdr.pack_propagate(False)

        tk.Label(vis_hdr, text="👁  VISIBILITY", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8, 'bold')).pack(side=tk.LEFT, padx=10)
        
        self.text_btn = tk.Button(
            vis_hdr,
            text="Hide label text" if state.show_bbox_text else "Show label text",
            command=self.toggle_bbox_text,
            bg=C_CARD2, fg=C_TXT1, font=('Segoe UI', 8, 'bold'), relief=tk.FLAT,
            activebackground=C_BORDER, activeforeground=C_TXT1,
            highlightbackground=C_BORDER, highlightthickness=1, borderwidth=0,
            cursor='hand2')
        self.text_btn.pack(side=tk.RIGHT, padx=(6,10), pady=6)

        vbtn_row = tk.Frame(vis_hdr, bg=C_CARD)
        vbtn_row.pack(side=tk.RIGHT, padx=6)

        # — Visibility scroll area —
        vis_canvas = tk.Canvas(left_panel, bg=C_PANEL, highlightthickness=0)
        vis_scroll = tk.Scrollbar(left_panel, orient="vertical", command=vis_canvas.yview,
                                   bg=C_BORDER, troughcolor=C_PANEL,
                                   relief=tk.FLAT, width=7)
        self.visibility_frame = tk.Frame(vis_canvas, bg=C_PANEL)
        self.visibility_frame.bind(
            "<Configure>",
            lambda e: vis_canvas.configure(scrollregion=vis_canvas.bbox("all"))
        )
        vis_canvas.create_window((0, 0), window=self.visibility_frame, anchor="nw")
        vis_canvas.configure(yscrollcommand=vis_scroll.set)
        vis_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vis_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.refresh_visibility_toggles()

        # ===== CANVAS AREA =====
        canvas_container = tk.Frame(content, bg=C_STAGE)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_container = canvas_container  # Store reference for zoom calculations

        self.canvas = tk.Canvas(canvas_container, bg=C_STAGE, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        self.canvas.bind('<Button-3>', self.on_right_click)
        self.canvas.bind('<Double-1>', self.on_double_click)
        self.canvas.bind('<MouseWheel>', self.on_canvas_scroll)  # Windows/Mac
        self.canvas.bind('<Button-4>', self.on_canvas_scroll)     # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_canvas_scroll)     # Linux scroll down

        # Left border for right panel
        tk.Frame(content, bg=C_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # ===== RIGHT PANEL =====
        right_panel = tk.Frame(content, bg=C_PANEL, width=270)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # — Info section header —
        info_hdr = tk.Frame(right_panel, bg=C_CARD, height=40)
        info_hdr.pack(fill=tk.X)
        info_hdr.pack_propagate(False)
        tk.Label(info_hdr, text="📊  INFORMATION", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8, 'bold')).pack(side=tk.LEFT, padx=10, pady=10)

        # — Info text widget —
        self.info_text = tk.Text(
            right_panel, bg=C_CARD, fg=C_ACCENT,
            font=('Consolas', 8), height=15, wrap=tk.WORD,
            relief=tk.FLAT, padx=8, pady=6,
            highlightthickness=0, borderwidth=0
        )
        self.info_text.pack(fill=tk.X, padx=6, pady=(5, 0))

        # Divider
        tk.Frame(right_panel, bg=C_BORDER, height=1).pack(fill=tk.X, pady=(6, 0))

        # — Images section header —
        img_hdr = tk.Frame(right_panel, bg=C_CARD, height=40)
        img_hdr.pack(fill=tk.X)
        img_hdr.pack_propagate(False)
        tk.Label(img_hdr, text="📁  IMAGES", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8, 'bold')).pack(side=tk.LEFT, padx=10, pady=10)

        # — Search bar —
        search_bar = tk.Frame(right_panel, bg=C_CARD2, height=30)
        search_bar.pack(fill=tk.X, padx=6, pady=(5, 2))
        search_bar.pack_propagate(False)

        tk.Label(search_bar, text="⌕", bg=C_CARD2, fg=C_TXT3,
                 font=('Segoe UI', 12)).pack(side=tk.LEFT, padx=(8, 2))

        self.image_search_var = tk.StringVar()
        self.image_search_var.trace('w', self.on_image_search_change)

        self.image_search_entry = tk.Entry(
            search_bar, textvariable=self.image_search_var,
            font=('Segoe UI', 9), bg=C_CARD2, fg=C_TXT1,
            relief=tk.FLAT, insertbackground=C_ACCENT,
            highlightthickness=0
        )
        self.image_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)

        # — Image listbox —
        img_list_wrap = tk.Frame(right_panel, bg=C_PANEL)
        img_list_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 4))

        img_scroll = tk.Scrollbar(img_list_wrap, bg=C_BORDER, troughcolor=C_PANEL,
                                   relief=tk.FLAT, width=7)
        img_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(
            img_list_wrap, bg=C_CARD, fg=C_TXT1,
            selectmode=tk.SINGLE, font=('Consolas', 8),
            yscrollcommand=img_scroll.set,
            activestyle='none', relief=tk.FLAT,
            selectbackground=C_ACCENT, selectforeground=C_ON_ACCENT,
            highlightthickness=0, borderwidth=0
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        img_scroll.config(command=self.image_listbox.yview)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        self.all_images_for_search = []
        self.refresh_image_list()

        # ======================================================
        # BOTTOM STATUS BAR
        # ======================================================
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill=tk.X)

        status_bar = tk.Frame(self.root, bg=C_PANEL, height=28)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar.pack_propagate(False)

        # Mode pill (left)
        self.statusbar_mode = tk.Label(
            status_bar, text="  📦 BBOX MODE  ",
            bg=C_ACCENT, fg=C_ON_ACCENT,
            font=('Segoe UI', 7, 'bold')
        )
        self.statusbar_mode.pack(side=tk.LEFT, padx=(8, 4), pady=4)

        # Force-new-bbox indicator (next to mode pill)
        self.force_label = tk.Label(
            status_bar, text="Force: OFF",
            bg=C_CARD, fg=C_TXT3,
            font=('Segoe UI', 7, 'bold'), padx=6
        )
        self.force_label.pack(side=tk.LEFT, padx=(0, 4), pady=4)
        self.update_force_label()

        tk.Frame(status_bar, bg=C_BORDER, width=1).pack(side=tk.LEFT,
                                                          fill=tk.Y, pady=4)

        # Progress text
        self.statusbar_progress = tk.Label(
            status_bar, text="Annotated: 0 / 0  (0%)",
            bg=C_PANEL, fg=C_TXT3,
            font=('Segoe UI', 8)
        )
        self.statusbar_progress.pack(side=tk.LEFT, padx=10)

        # Theme toggle (right) - applies on next start
        theme_btn = tk.Button(
            status_bar, text="◐ Dark" if MODE == "light" else "◐ Light",
            command=self.toggle_theme, bg=C_PANEL, fg=C_TXT2, relief=tk.FLAT,
            font=('Segoe UI', 7, 'bold'), cursor='hand2', borderwidth=0,
            activebackground=C_CARD2, activeforeground=C_TXT1)
        theme_btn.pack(side=tk.RIGHT, padx=(0, 8), pady=3, ipadx=6)

        # Keyboard hints (right)
        tk.Label(
            status_bar,
            text="A/D: Navigate  ·  B: BBox  ·  M: Mode  ·  G: Infer  ·  "
                 "T: Train  ·  Del: Remove img  ·  Esc: Exit",
            bg=C_PANEL, fg=C_TXT3, font=('Segoe UI', 7)
        ).pack(side=tk.RIGHT, padx=10)

    # ----------------------------------------------------------
    #  Class list refresh
    # ----------------------------------------------------------
    def refresh_class_list(self):
        global CLASSLIST, colorsPalette
        CLASSLIST = class_manager.get_classes()
        colorsPalette = class_manager.get_colors()

        self.class_listbox.delete(0, tk.END)
        for i, cls in enumerate(CLASSLIST):
            self.class_listbox.insert(tk.END, f"  {i+1}.  {cls}")
            if i < len(colorsPalette):
                hex_color = self.rgb_to_hex(colorsPalette[i])
                # Darken the class color for better readability in the list
                self.class_listbox.itemconfig(i, bg=hex_color, fg=self._readable_fg(hex_color))

        if CLASSLIST:
            self.class_listbox.select_set(0)
            state.current_class = CLASSLIST[0]
            print(f"[GUI] Loaded {len(CLASSLIST)} classes")

    # ----------------------------------------------------------
    #  Visibility toggles refresh
    # ----------------------------------------------------------
    def refresh_visibility_toggles(self):
        for widget in self.visibility_frame.winfo_children():
            widget.destroy()

        self.visibility_vars = {}
        self.visibility_count_labels = {}
        for cls in CLASSLIST:
            var = tk.BooleanVar(value=state.visible_class.get(cls, True))
            self.visibility_vars[cls] = var

            row = tk.Frame(self.visibility_frame, bg=C_PANEL)
            row.pack(anchor=tk.W, pady=1, fill=tk.X, padx=4)

            cb = tk.Checkbutton(
                row, text=cls, variable=var,
                bg=C_PANEL, fg=C_TXT1,
                selectcolor=C_CARD2,
                font=('Segoe UI', 9),
                command=self.update_display,
                activebackground=C_PANEL, activeforeground=C_TXT1
            )
            cb.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

            count_label = tk.Label(row, text="0", bg=C_PANEL, fg=C_ACCENT,
                                    font=('Segoe UI', 8, 'bold'), width=5, anchor='e')
            count_label.pack(side=tk.RIGHT, padx=4)
            self.visibility_count_labels[cls] = count_label

        state.visible_class = {cls: var.get() for cls, var in self.visibility_vars.items()}
        self.update_visibility_counts()

    def update_visibility_counts(self):
        counts = {cls: 0 for cls in CLASSLIST}
        for _, _, _, _, cls in state.bboxes:
            if cls in counts:
                counts[cls] += 1
        for _, cls in state.polygons:
            if cls in counts:
                counts[cls] += 1
        for cls, label in self.visibility_count_labels.items():
            label.config(text=str(counts.get(cls, 0)))

    # ----------------------------------------------------------
    #  Export Dataset UI
    # ----------------------------------------------------------
    def _find_workspace_images(self, image_exts):
        """Collect images from all indexed datasets for the active workspace."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        datasets_root = os.path.join(project_root, 'datasetsInput')
        prefix = workspaceName + '-'
        records = []

        if not os.path.isdir(datasets_root):
            return records

        folders = sorted(
            folder for folder in os.listdir(datasets_root)
            if folder.startswith(prefix) and os.path.isdir(os.path.join(datasets_root, folder))
        )
        for folder in folders:
            folder_path = os.path.join(datasets_root, folder)
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(image_exts):
                    records.append({
                        'source': os.path.join(folder_path, filename),
                        'filename': filename,
                        'base': os.path.splitext(filename)[0]
                    })
        return records

    def _read_coco_annotations(self, xml_path, image_width, image_height, categories):
        """Convert Jelibox VOC objects into COCO annotation dictionaries."""
        annotations = []
        if not os.path.exists(xml_path):
            return annotations

        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            print(f"[COCO] Could not read {xml_path}: {exc}")
            return annotations

        for obj in root.findall('object'):
            name_elem = obj.find('name')
            class_name = (name_elem.text or '').strip() if name_elem is not None else ''
            if not class_name or class_name not in categories:
                continue

            polygon = obj.find('polygon')
            segmentation = []
            points = []
            if polygon is not None:
                point_elements = polygon.findall('point')
                if point_elements:
                    for point in point_elements:
                        x_elem = point.find('x')
                        y_elem = point.find('y')
                        if x_elem is not None and y_elem is not None:
                            points.append((float(x_elem.text), float(y_elem.text)))
                else:
                    index = 1
                    while polygon.find(f'x{index}') is not None and polygon.find(f'y{index}') is not None:
                        points.append((
                            float(polygon.find(f'x{index}').text),
                            float(polygon.find(f'y{index}').text)
                        ))
                        index += 1

            if len(points) >= 3:
                points = [
                    (max(0.0, min(float(image_width), x)),
                     max(0.0, min(float(image_height), y)))
                    for x, y in points
                ]
                segmentation = [[coord for point in points for coord in point]]
                xmin = min(point[0] for point in points)
                ymin = min(point[1] for point in points)
                xmax = max(point[0] for point in points)
                ymax = max(point[1] for point in points)
                area = 0.5 * abs(sum(
                    points[index][0] * points[(index + 1) % len(points)][1] -
                    points[(index + 1) % len(points)][0] * points[index][1]
                    for index in range(len(points))
                ))
            else:
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                xmin = float(bbox.findtext('xmin', '0'))
                ymin = float(bbox.findtext('ymin', '0'))
                xmax = float(bbox.findtext('xmax', '0'))
                ymax = float(bbox.findtext('ymax', '0'))
                area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

            xmin = max(0.0, min(float(image_width), xmin))
            ymin = max(0.0, min(float(image_height), ymin))
            xmax = max(xmin, min(float(image_width), xmax))
            ymax = max(ymin, min(float(image_height), ymax))
            annotations.append({
                'category_id': categories[class_name],
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'area': area,
                'segmentation': segmentation,
                'iscrowd': 0
            })

        return annotations

    def _export_coco_dataset(self, target, train_pct, valid_pct, image_exts, progress_cb=None):
        """Export indexed workspace images and VOC annotations as COCO JSON."""
        records = self._find_workspace_images(image_exts)
        if not records:
            raise ValueError(f'No images found for workspace {workspaceName} in datasetsInput')

        random.shuffle(records)
        total = len(records)
        processed = 0
        train_count = int(round(total * (train_pct / 100.0)))
        valid_count = int(round(total * (valid_pct / 100.0)))
        if train_count + valid_count > total:
            valid_count = max(0, total - train_count)

        splits = {
            'train': records[:train_count],
            'val': records[train_count:train_count + valid_count],
            'test': records[train_count + valid_count:]
        }
        categories = {name: index + 1 for index, name in enumerate(CLASSLIST)}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        xml_root = os.path.join(project_root, 'vocdataset', workspaceName)

        for split_name, split_records in splits.items():
            image_dir = os.path.join(target, 'images', split_name)
            os.makedirs(image_dir, exist_ok=True)
            coco = {
                'info': {'description': f'Jelibox {workspaceName} dataset'},
                'licenses': [],
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': category_id, 'name': name, 'supercategory': 'object'}
                    for name, category_id in categories.items()
                ]
            }
            annotation_id = 1

            for image_id, record in enumerate(split_records, start=1):
                try:
                    with Image.open(record['source']) as image:
                        image_width, image_height = image.size
                    output_name = record['filename']
                    destination = os.path.join(image_dir, output_name)
                    if os.path.exists(destination):
                        output_name = f"{os.path.basename(os.path.dirname(record['source']))}_{record['filename']}"
                        destination = os.path.join(image_dir, output_name)
                    shutil.copy2(record['source'], destination)
                except (OSError, ValueError) as exc:
                    print(f"[COCO] Skipping {record['source']}: {exc}")
                    continue

                coco['images'].append({
                    'id': image_id,
                    'file_name': output_name,
                    'width': image_width,
                    'height': image_height
                })

                xml_path = os.path.join(xml_root, record['base'] + '.xml')
                for annotation in self._read_coco_annotations(
                        xml_path, image_width, image_height, categories):
                    annotation['id'] = annotation_id
                    annotation['image_id'] = image_id
                    coco['annotations'].append(annotation)
                    annotation_id += 1

                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f'Exporting COCO... {processed}/{total}')

            annotation_dir = os.path.join(target, 'annotations')
            os.makedirs(annotation_dir, exist_ok=True)
            json_path = os.path.join(annotation_dir, f'instances_{split_name}.json')
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(coco, json_file, indent=2)

        return target

    def _export_yolo_dataset(self, target, train_pct, valid_pct, image_exts, progress_cb=None):
        """Export the workspace's YOLO labels as a train/val/test split, pulling
        source images straight from datasetsInput (never from a copied folder)."""
        records = self._find_workspace_images(image_exts)
        images_by_base = {r['base']: r['source'] for r in records}

        label_files = [f for f in os.listdir(yolo_labels_folder) if f.endswith('.txt')] \
            if os.path.isdir(yolo_labels_folder) else []
        pairs = [(images_by_base[base], base + '.txt')
                 for base in (os.path.splitext(f)[0] for f in label_files)
                 if base in images_by_base]

        if not pairs:
            raise ValueError(f'No labeled images found for workspace {workspaceName}')

        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(round(n * (train_pct / 100.0)))
        n_valid = int(round(n * (valid_pct / 100.0)))
        if n_train + n_valid > n:
            n_valid = max(0, n - n_train)
        splits = {
            'train': pairs[:n_train],
            'val': pairs[n_train:n_train + n_valid],
            'test': pairs[n_train + n_valid:]
        }

        processed = 0
        for split_name, split_pairs in splits.items():
            out_images = os.path.join(target, split_name, 'images')
            out_labels = os.path.join(target, split_name, 'labels')
            os.makedirs(out_images, exist_ok=True)
            os.makedirs(out_labels, exist_ok=True)
            for img_src, lbl_name in split_pairs:
                try:
                    shutil.copy2(img_src, os.path.join(out_images, os.path.basename(img_src)))
                except Exception:
                    pass
                try:
                    shutil.copy2(os.path.join(yolo_labels_folder, lbl_name),
                                 os.path.join(out_labels, lbl_name))
                except Exception:
                    pass

                processed += 1
                if progress_cb:
                    progress_cb(processed, n, f'Exporting YOLO... {processed}/{n}')

        try:
            data_yml = os.path.join(target, 'data.yml')
            with open(data_yml, 'w') as fy:
                fy.write(f"train: {os.path.abspath(os.path.join(target, 'train', 'images'))}\n")
                fy.write(f"val:   {os.path.abspath(os.path.join(target, 'val', 'images'))}\n")
                fy.write(f"test:  {os.path.abspath(os.path.join(target, 'test', 'images'))}\n")
                fy.write(f"nc:    {len(CLASSLIST)}\n")
                fy.write('names:\n')
                for i, name in enumerate(CLASSLIST):
                    fy.write(f"  {i}: {name}\n")
        except Exception:
            pass

        return target

    def _export_voc_dataset(self, target, train_pct, valid_pct, image_exts, progress_cb=None):
        """Export indexed workspace images and their VOC XML annotations."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        datasets_input_root = os.path.join(project_root, 'datasetsInput')
        workspace_prefix = workspaceName

        folders = [f for f in os.listdir(datasets_input_root) if f.startswith(workspace_prefix + '-')] if os.path.isdir(datasets_input_root) else []
        images_found = []
        for folder in folders:
            full = os.path.join(datasets_input_root, folder)
            if not os.path.isdir(full):
                continue
            files = [f for f in os.listdir(full) if os.path.splitext(f)[1].lower() in image_exts]
            for f in files:
                images_found.append(os.path.join(full, f))

        if not images_found:
            raise ValueError(f'No images found for workspace {workspace_prefix} in datasetsInput')

        random.shuffle(images_found)
        n = len(images_found)
        n_train = int(round(n * (train_pct / 100.0)))
        n_valid = int(round(n * (valid_pct / 100.0)))
        if n_train + n_valid > n:
            n_valid = max(0, n - n_train)

        train_imgs = images_found[:n_train]
        valid_imgs = images_found[n_train:n_train + n_valid]
        test_imgs = images_found[n_train + n_valid:]

        processed = 0
        for split_name, imgs_list in (('train', train_imgs), ('valid', valid_imgs), ('test', test_imgs)):
            out_dir = os.path.join(target, split_name)
            os.makedirs(out_dir, exist_ok=True)
            for img_path in imgs_list:
                try:
                    shutil.copy2(img_path, os.path.join(out_dir, os.path.basename(img_path)))
                except Exception:
                    pass
                base = os.path.splitext(os.path.basename(img_path))[0]
                xml_path = os.path.join(vocdataset_folder, base + '.xml')
                if os.path.exists(xml_path):
                    try:
                        shutil.copy2(xml_path, os.path.join(out_dir, base + '.xml'))
                    except Exception:
                        pass

                processed += 1
                if progress_cb:
                    progress_cb(processed, n, f'Exporting Pascal VOC... {processed}/{n}')

        return target

    def show_export_dataset_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Dataset")
        dialog.geometry("520x350")
        dialog.configure(bg=C_BASE)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        body = tk.Frame(dialog, bg=C_CARD, padx=12, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        # Format dropdown
        tk.Label(body, text="Format:", bg=C_CARD, fg=C_TXT1).pack(anchor='w')
        fmt_var = tk.StringVar(value='YOLO')
        fmt_combo = ttk.Combobox(body, values=["YOLO", "Pascal VOC (XML)", "COCO"], state='readonly', font=('Segoe UI', 10))
        fmt_combo.set('YOLO')
        fmt_combo.pack(fill=tk.X, pady=(0,8))

        # Split slider area (two separators define train/valid/test)
        tk.Label(body, text="Split (Train / Valid / Test):", bg=C_CARD, fg=C_TXT1).pack(anchor='w')

        slider_w = 460
        slider_h = 60
        slider_canvas = tk.Canvas(body, width=slider_w, height=slider_h, bg=C_CARD, highlightthickness=0)
        slider_canvas.pack(pady=(4,8))

        # initial separators (px)
        left_x = int(slider_w * 0.7)
        right_x = int(slider_w * 0.9)

        bar_y = slider_h // 2
        bar_h = 10
        # draw bar
        slider_canvas.create_rectangle(10, bar_y - bar_h//2, slider_w - 10, bar_y + bar_h//2, fill=C_CARD2, outline='')

        handle_radius = 8
        handle1 = slider_canvas.create_oval(left_x - handle_radius, bar_y - handle_radius,
                                            left_x + handle_radius, bar_y + handle_radius,
                                            fill=C_ACCENT, outline='')
        handle2 = slider_canvas.create_oval(right_x - handle_radius, bar_y - handle_radius,
                                            right_x + handle_radius, bar_y + handle_radius,
                                            fill=C_ACCENT, outline='')

        # percent labels above centers
        train_lbl = slider_canvas.create_text(0, 12, text='', fill=C_TXT1, font=('Segoe UI', 9, 'bold'))
        valid_lbl = slider_canvas.create_text(0, 12, text='', fill=C_TXT1, font=('Segoe UI', 9, 'bold'))
        test_lbl = slider_canvas.create_text(0, 12, text='', fill=C_TXT1, font=('Segoe UI', 9, 'bold'))

        def update_labels():
            coords1 = slider_canvas.coords(handle1)
            coords2 = slider_canvas.coords(handle2)
            x1 = (coords1[0] + coords1[2]) / 2
            x2 = (coords2[0] + coords2[2]) / 2
            left = 10
            right = slider_w - 10
            total = right - left
            train_pct = round(((x1 - left) / total) * 100)
            valid_pct = round(((x2 - x1) / total) * 100)
            test_pct = 100 - train_pct - valid_pct

            # ensure bounds
            if train_pct < 0:
                train_pct = 0
            if valid_pct < 0:
                valid_pct = 0

            # positions for labels (center of ranges)
            train_cx = left + (x1 - left) / 2
            valid_cx = x1 + (x2 - x1) / 2
            test_cx = x2 + (right - x2) / 2

            slider_canvas.coords(train_lbl, train_cx, 12)
            slider_canvas.coords(valid_lbl, valid_cx, 12)
            slider_canvas.coords(test_lbl, test_cx, 12)

            slider_canvas.itemconfig(train_lbl, text=f"{train_pct}%")
            slider_canvas.itemconfig(valid_lbl, text=f"{valid_pct}%")
            slider_canvas.itemconfig(test_lbl, text=f"{test_pct}%")

            train_var.set(str(train_pct))
            valid_var.set(str(valid_pct))
            test_var.set(str(test_pct))

        # Dragging logic
        drag_data = {'item': None}
        export_state = {'active': False}

        def on_press(event):
            if export_state['active']:
                return
            item = slider_canvas.find_closest(event.x, event.y)[0]
            if item in (handle1, handle2):
                drag_data['item'] = item

        def on_release(event):
            drag_data['item'] = None

        def on_motion(event):
            item = drag_data.get('item')
            if not item:
                return
            x = event.x
            left = 10
            right = slider_w - 10
            # enforce ordering
            coords1 = slider_canvas.coords(handle1)
            coords2 = slider_canvas.coords(handle2)
            x1 = (coords1[0] + coords1[2]) / 2
            x2 = (coords2[0] + coords2[2]) / 2
            min_gap = 4
            if item == handle1:
                x = max(left, min(x, x2 - min_gap))
                slider_canvas.coords(handle1, x - handle_radius, bar_y - handle_radius, x + handle_radius, bar_y + handle_radius)
            else:
                x = min(right, max(x, x1 + min_gap))
                slider_canvas.coords(handle2, x - handle_radius, bar_y - handle_radius, x + handle_radius, bar_y + handle_radius)
            update_labels()

        slider_canvas.bind('<Button-1>', on_press)
        slider_canvas.bind('<B1-Motion>', on_motion)
        slider_canvas.bind('<ButtonRelease-1>', on_release)

        # numeric readouts / spinboxes
        readout_row = tk.Frame(body, bg=C_CARD)
        readout_row.pack(fill=tk.X)

        # Use StringVar to avoid Tcl numeric parsing errors when user types invalid numeric strings
        train_var = tk.StringVar(value=str(round((left_x - 10) / (slider_w - 20) * 100)))
        valid_var = tk.StringVar(value=str(round((right_x - left_x) / (slider_w - 20) * 100)))
        test_var = tk.StringVar(value='0')

        def safe_to_int(s):
            if s is None:
                return 0
            # keep only digits
            s2 = ''.join(ch for ch in str(s) if ch.isdigit())
            if s2 == '':
                return 0
            try:
                return int(s2)
            except Exception:
                return 0

        def on_spin_change(*a):
            t = max(0, min(100, safe_to_int(train_var.get())))
            v = max(0, min(100 - t, safe_to_int(valid_var.get())))
            train_var.set(str(t))
            valid_var.set(str(v))
            test_var.set(str(100 - t - v))
            # update handles
            left = 10
            right = slider_w - 10
            total = right - left
            new_x1 = left + (t / 100.0) * total
            new_x2 = new_x1 + (v / 100.0) * total
            slider_canvas.coords(handle1, new_x1 - handle_radius, bar_y - handle_radius, new_x1 + handle_radius, bar_y + handle_radius)
            slider_canvas.coords(handle2, new_x2 - handle_radius, bar_y - handle_radius, new_x2 + handle_radius, bar_y + handle_radius)
            update_labels()

        tk.Label(readout_row, text='Train %', bg=C_CARD, fg=C_TXT2).grid(row=0, column=0, padx=6)
        tk.Label(readout_row, text='Valid %', bg=C_CARD, fg=C_TXT2).grid(row=0, column=1, padx=6)
        tk.Label(readout_row, text='Test %', bg=C_CARD, fg=C_TXT2).grid(row=0, column=2, padx=6)

        sb_train = tk.Spinbox(readout_row, from_=0, to=100, textvariable=train_var, width=6, command=on_spin_change)
        sb_valid = tk.Spinbox(readout_row, from_=0, to=100, textvariable=valid_var, width=6, command=on_spin_change)
        sb_train.grid(row=1, column=0, padx=6)
        sb_valid.grid(row=1, column=1, padx=6)
        tk.Label(readout_row, textvariable=test_var, bg=C_CARD, fg=C_TXT1, width=6).grid(row=1, column=2, padx=6)

        # initialize labels
        update_labels()

        # Ensure typing into spinboxes updates values immediately
        try:
            train_var.trace_add('write', lambda *a: on_spin_change())
            valid_var.trace_add('write', lambda *a: on_spin_change())
        except Exception:
            train_var.trace('w', lambda *a: on_spin_change())
            valid_var.trace('w', lambda *a: on_spin_change())

        status_lbl = tk.Label(body, text='', bg=C_CARD, fg=C_TXT1)
        status_lbl.pack(fill=tk.X, pady=(8,0))

        progress_bar = ttk.Progressbar(body, orient='horizontal', mode='determinate', maximum=100)
        progress_bar.pack(fill=tk.X, pady=(4,0))

        def set_ui_locked(locked):
            export_state['active'] = locked
            state = 'disabled' if locked else 'readonly'
            fmt_combo.config(state=state)
            sb_train.config(state='disabled' if locked else 'normal')
            sb_valid.config(state='disabled' if locked else 'normal')
            export_btn.config(state='disabled' if locked else 'normal')
            close_btn.config(state='disabled' if locked else 'normal')

        def on_dialog_close():
            if export_state['active']:
                return
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        def do_export_dataset():
            if export_state['active']:
                return

            fmt = fmt_combo.get()
            t = safe_to_int(train_var.get())
            v = safe_to_int(valid_var.get())
            te = 100 - t - v
            if t + v + te != 100:
                messagebox.showerror('Invalid Split', 'Train+Valid+Test must equal 100%', parent=dialog)
                return

            image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

            set_ui_locked(True)
            progress_bar['value'] = 0
            status_lbl.config(text='Preparing export...')
            # Force an immediate repaint so the locked/0% state is visible right away,
            # instead of only appearing once the (possibly very fast) worker finishes.
            dialog.update_idletasks()

            last_pct = {'value': -1}

            def progress_cb(current, total, message=None):
                def update():
                    pct = int((current / total) * 100) if total else 0
                    progress_bar['value'] = pct
                    status_lbl.config(text=message or f'Exporting... {pct}%')
                    # Small/fast exports can finish within a single Tk event-loop
                    # iteration, so queued updates never get painted before the
                    # completion dialog takes over. Force a repaint whenever the
                    # displayed percentage actually changes so progress is visible
                    # without adding per-file overhead on large datasets.
                    if pct != last_pct['value']:
                        last_pct['value'] = pct
                        progress_bar.update_idletasks()
                self.root.after(0, update)

            def worker():
                try:
                    target = export_dataset_folder
                    # If the target exists, remove it to avoid mixing previous exports
                    try:
                        if os.path.exists(target):
                            # Safety: ensure target is inside project folder to avoid accidental deletes
                            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                            abs_target = os.path.abspath(target)
                            if abs_target.startswith(proj_root):
                                shutil.rmtree(abs_target)
                            else:
                                # If target is outside project, don't auto-delete; just create subfolder with timestamp
                                target = os.path.join(target, f'export_{int(time.time())}')
                        os.makedirs(target, exist_ok=True)
                    except Exception:
                        os.makedirs(target, exist_ok=True)

                    if fmt == 'YOLO':
                        self._export_yolo_dataset(target, t, v, image_exts, progress_cb)
                    elif 'Pascal' in fmt or 'XML' in fmt:
                        self._export_voc_dataset(target, t, v, image_exts, progress_cb)
                    elif fmt == 'COCO':
                        self._export_coco_dataset(target, t, v, image_exts, progress_cb)
                    else:
                        raise ValueError(f'Unknown format: {fmt}')

                    def on_success():
                        set_ui_locked(False)
                        progress_bar['value'] = 100
                        status_lbl.config(text=f'Export completed to: {target}')
                        messagebox.showinfo('Export Completed', f'Dataset exported to:\n{target}', parent=dialog)
                    self.root.after(0, on_success)

                except Exception as e:
                    err = str(e)
                    def on_error():
                        set_ui_locked(False)
                        progress_bar['value'] = 0
                        status_lbl.config(text='Export failed')
                        messagebox.showerror('Export Failed', f'Error during export:\n{err}', parent=dialog)
                    self.root.after(0, on_error)

            threading.Thread(target=worker, daemon=True).start()

        btn_row = tk.Frame(body, bg=C_CARD)
        btn_row.pack(fill=tk.X, pady=(8,0))
        export_btn = tk.Button(btn_row, text='Export', command=do_export_dataset, bg=C_ACCENT, fg=C_ON_ACCENT)
        export_btn.pack(side=tk.RIGHT, padx=6)
        close_btn = tk.Button(btn_row, text='Close', command=on_dialog_close, bg=C_CARD2, fg=C_TXT1)
        close_btn.pack(side=tk.RIGHT, padx=(0,6))

    # ----------------------------------------------------------
    #  Add / Delete class dialogs
    # ----------------------------------------------------------
    def add_class_dialog(self):
        global CLASSLIST
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Class")
        dialog.geometry("420x220")
        dialog.configure(bg=C_BASE)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Header
        hdr = tk.Frame(dialog, bg=C_GREEN, height=46)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="➕  Add New Class", font=('Segoe UI', 12, 'bold'),
                 bg=C_ACCENT, fg=C_ON_ACCENT).pack(side=tk.LEFT, padx=16, pady=10)

        body = tk.Frame(dialog, bg=C_BASE, padx=24, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(body, text="Class Name:", font=('Segoe UI', 10, 'bold'),
                 bg=C_BASE, fg=C_TXT1).pack(anchor='w', pady=(0, 6))

        entry = tk.Entry(body, font=('Segoe UI', 11), bg=C_CARD2, fg=C_TXT1,
                         insertbackground=C_ACCENT, relief=tk.FLAT,
                         highlightthickness=1, highlightcolor=C_ACCENT,
                         highlightbackground=C_BORDER)
        entry.pack(fill=tk.X, ipady=6)
        entry.focus()

        tk.Label(body, text="Allowed: letters, numbers, underscore, dash",
                 font=('Segoe UI', 8), bg=C_BASE, fg=C_TXT3).pack(anchor='w', pady=(4, 0))

        def on_submit():
            new_class = entry.get().strip()
            if new_class:
                success, message = class_manager.add_class(new_class)
                if success:
                    messagebox.showinfo("Success", message, parent=dialog)
                    self.refresh_class_list()
                    self.refresh_visibility_toggles()
                    self.update_display()
                    self.update_info()
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", message, parent=dialog)
                    entry.delete(0, tk.END)
                    entry.focus()

        btn_row = tk.Frame(body, bg=C_BASE)
        btn_row.pack(fill=tk.X, pady=(14, 0))

        self._btn(btn_row, "✕  Cancel", dialog.destroy,
                  C_CARD2, C_TXT2, font_size=10).pack(side=tk.LEFT, ipady=7, ipadx=14)
        self._btn(btn_row, "✓  Add Class", on_submit,
                  C_ACCENT, C_ON_ACCENT, font_size=10, bold=True).pack(
            side=tk.RIGHT, ipady=7, ipadx=16)

        entry.bind('<Return>', lambda e: on_submit())
        entry.bind('<Escape>', lambda e: dialog.destroy())
        CLASSLIST = class_manager.get_classes()

    def delete_class_dialog(self):
        global CLASSLIST
        selection = self.class_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection",
                                   "Select a class to delete first!",
                                   parent=self.root)
            return

        idx = selection[0]
        if idx >= len(CLASSLIST):
            return
        class_to_delete = CLASSLIST[idx]
        affected_count = sum(1 for bbox in state.bboxes if bbox[4] == class_to_delete)

        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Delete Class")
        dialog.geometry("460x280")
        dialog.configure(bg=C_BASE)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Header
        hdr = tk.Frame(dialog, bg=C_RED, height=46)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚠️  Delete Class Warning", font=('Segoe UI', 12, 'bold'),
                 bg=C_RED, fg=C_ON_RED).pack(side=tk.LEFT, padx=16, pady=10)

        body = tk.Frame(dialog, bg=C_BASE, padx=24, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

        msg = (f"Deleting class  '{class_to_delete}'  will:\n\n"
               f"  •  Remove it from the model dataset\n"
               f"  •  Delete {affected_count} annotation(s) in this image\n"
               f"  •  Update all label files in the dataset\n"
               f"  •  Cannot be undone!\n\n"
               f"Are you sure?")

        tk.Label(body, text=msg, bg=C_BASE, fg=C_TXT1,
                 font=('Segoe UI', 10), justify=tk.LEFT).pack(anchor='w')

        def on_confirm():
            success, message = class_manager.delete_class(class_to_delete)
            if success:
                state.bboxes = [b for b in state.bboxes if b[4] != class_to_delete]
                self.refresh_class_list()
                self.refresh_visibility_toggles()
                self.update_display()
                self.update_info()
                dialog.destroy()
                messagebox.showinfo("Success", message, parent=self.root)
            else:
                dialog.destroy()
                messagebox.showerror("Error", message, parent=self.root)

        btn_row = tk.Frame(body, bg=C_BASE)
        btn_row.pack(fill=tk.X, pady=(16, 0))

        self._btn(btn_row, "✕  Cancel", dialog.destroy,
                  C_CARD2, C_TXT2, font_size=10).pack(side=tk.LEFT, ipady=7, ipadx=14)
        self._btn(btn_row, "🗑  Yes, Delete", on_confirm,
                  C_RED, C_ON_RED, font_size=10, bold=True).pack(
            side=tk.RIGHT, ipady=7, ipadx=16)

        CLASSLIST = class_manager.get_classes()

    # ----------------------------------------------------------
    #  Utility
    # ----------------------------------------------------------
    def rgb_to_hex(self, rgb):
        return f'#{rgb[2]:02x}{rgb[1]:02x}{rgb[0]:02x}'

    # ----------------------------------------------------------
    #  Image loading & display
    # ----------------------------------------------------------
    def initial_load(self):
        self.refresh_image_list()
        self.load_current_image()
        self.update_display()

    def load_current_image(self):
        img_name = self.images[state.current_index]
        img_path = os.path.join(input_folder, img_name)

        orig = cv2.imread(img_path)
        if orig is None:
            messagebox.showerror("Error", f"Cannot load {img_name}")
            return

        state.orig_shape = orig.shape
        h, w = state.orig_shape[:2]
        state.orig_width = w
        state.orig_height = h

        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 100:
            canvas_w = 1000
        if canvas_h < 100:
            canvas_h = 700

        padding = 20
        scale_w = (canvas_w - padding * 2) / w
        scale_h = (canvas_h - padding * 2) / h
        scale = min(scale_w, scale_h, 1.0)

        state.display_scale = scale
        new_w = int(w * scale)
        new_h = int(h * scale)
        state.display_width = new_w
        state.display_height = new_h
        state.frame = cv2.resize(orig, (new_w, new_h))

        frame_rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        self.current_img_pil = Image.fromarray(frame_rgb)

        self.canvas_offset_x = (canvas_w - new_w) // 2
        self.canvas_offset_y = (canvas_h - new_h) // 2

        state.bboxes, state.polygons = load_annotation_local(img_name)
        state.polygon_points_preview = []
        state.polygon_editing_mode = False
        state.selected_polygon = None
        state.selected_polygon_point = None

        print(f"[GUI] Loaded image: {img_name}")
        print(f"      Original: {state.orig_width}x{state.orig_height}px")
        print(f"      Display:  {state.display_width}x{state.display_height}px "
              f"(scale: {state.display_scale:.3f})")
        print(f"      Annotations: {len(state.bboxes)} bboxes, {len(state.polygons)} polygons")

        self.update_info()

    def update_display(self):
        """Redraw canvas with image, bboxes, and polygons"""
        if self.current_img_pil is None:
            return

        img_copy = self.current_img_pil.copy()
        img_array = np.array(img_copy)

        # Draw all bboxes
        for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
            if not self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                continue

            class_index = class_manager.get_class_index(cls)
            if class_index == -1:
                class_index = 0
            colors = class_manager.get_colors()
            color = colors[class_index] if class_index < len(colors) else (255, 0, 0)

            if i == state.selected_bbox:
                color = (60, 60, 200)       # single select → merah
            elif i in getattr(state, 'selected_bboxes', set()):
                color = (255, 212, 0)       # multi select → cyan

            color_rgb = (color[2], color[1], color[0])
            # Scale bbox from original coordinates to display coordinates
            x1_disp = int(x1 * state.display_scale)
            y1_disp = int(y1 * state.display_scale)
            x2_disp = int(x2 * state.display_scale)
            y2_disp = int(y2 * state.display_scale)
            self.draw_bbox_outline(img_array, x1_disp, y1_disp, x2_disp, y2_disp, color_rgb)

            if i == state.selected_bbox:
                self.draw_bbox_handles(img_array, x1_disp, y1_disp, x2_disp, y2_disp)

            label = cls
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if state.show_bbox_text:
                cv2.rectangle(img_array,
                              (x1_disp, y1_disp - label_h - 8), (x1_disp + label_w + 8, y1_disp),
                              color_rgb, -1)
                cv2.putText(img_array, label, (x1_disp + 4, y1_disp - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

        # Draw all polygons
        for i, (points_orig, cls) in enumerate(state.polygons):
            if not self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                continue

            class_index = class_manager.get_class_index(cls)
            if class_index == -1:
                class_index = 0
            colors = class_manager.get_colors()
            color = colors[class_index] if class_index < len(colors) else (255, 0, 0)
            is_selected = (i == state.selected_polygon)
            color_rgb = (color[2], color[1], color[0])

            points_display = [
                (int(x * state.display_scale), int(y * state.display_scale))
                for x, y in points_orig
            ]
            img_array = polygon_manager.draw_polygon(
                img_array, points_display, color_rgb,
                thickness=2, is_selected=is_selected,
                show_points=False
            )

            if len(points_display) > 0 and state.show_bbox_text:
                pt = points_display[0]
                cv2.putText(img_array, cls, (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1, cv2.LINE_AA)

        # Draw polygon preview
        if state.polygon_points_preview and len(state.polygon_points_preview) > 0:
            preview_pts = [
                (int(x * state.display_scale), int(y * state.display_scale))
                for x, y in state.polygon_points_preview
            ]
            img_array = polygon_manager.draw_preview_polygon(img_array, preview_pts)

        # Draw mask polygon preview
        if self.mask_polygon_points and len(self.mask_polygon_points) > 0:
            mask_pts = [
                (int(x * state.display_scale), int(y * state.display_scale))
                for x, y in self.mask_polygon_points
            ]
            img_array = polygon_manager.draw_preview_polygon(
                img_array, mask_pts, color=(255, 100, 100))

        # Force bbox indicator
        if state.force_new_bbox:
            h, w = img_array.shape[:2]
            cv2.rectangle(img_array, (5, 5), (w - 5, h - 5), (200, 60, 60), 3)

        img_with_boxes = Image.fromarray(img_array)
        self.photo = ImageTk.PhotoImage(img_with_boxes)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                  anchor=tk.NW, image=self.photo)

        # self.img_label.config(
        #     text=f"{state.current_index + 1} / {len(self.images)}")

    def draw_bbox_handles(self, img_array, x1, y1, x2, y2):
        """Draw a clear resize affordance around the selected bounding box."""
        handle_size = max(6, min(10, int(9 * max(state.display_scale, 0.5))))
        edge_color = (255, 255, 255)
        accent_color = (0, 212, 255)

        # Contrast outline makes the editable edge visible over any image.
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 0), 5)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), accent_color, 2)

        handles = [
            (x1, y1, 'corner'), ((x1 + x2) // 2, y1, 'edge'), (x2, y1, 'corner'),
            (x2, (y1 + y2) // 2, 'edge'), (x2, y2, 'corner'),
            ((x1 + x2) // 2, y2, 'edge'), (x1, y2, 'corner'),
            (x1, (y1 + y2) // 2, 'edge')
        ]
        for hx, hy, handle_type in handles:
            if handle_type == 'corner':
                cv2.rectangle(
                    img_array,
                    (hx - handle_size, hy - handle_size),
                    (hx + handle_size, hy + handle_size),
                    edge_color, -1
                )
                cv2.rectangle(
                    img_array,
                    (hx - handle_size, hy - handle_size),
                    (hx + handle_size, hy + handle_size),
                    accent_color, 2
                )
            else:
                cv2.circle(img_array, (hx, hy), handle_size - 1, edge_color, -1)
                cv2.circle(img_array, (hx, hy), handle_size - 1, accent_color, 2)

    def draw_bbox_outline(self, img_array, x1, y1, x2, y2, color):
        """Draw an outlined bbox with lightweight corner accents."""
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

        corner_length = max(8, min(18, int(14 * max(state.display_scale, 0.5))))
        accent = (255, 255, 255)
        corner_segments = [
            ((x1, y1), (x1 + corner_length, y1)),
            ((x1, y1), (x1, y1 + corner_length)),
            ((x2, y1), (x2 - corner_length, y1)),
            ((x2, y1), (x2, y1 + corner_length)),
            ((x1, y2), (x1 + corner_length, y2)),
            ((x1, y2), (x1, y2 - corner_length)),
            ((x2, y2), (x2 - corner_length, y2)),
            ((x2, y2), (x2, y2 - corner_length))
        ]
        for start, end in corner_segments:
            cv2.line(img_array, start, end, accent, 2)

    # ----------------------------------------------------------
    #  Info panel update
    # ----------------------------------------------------------
    def update_info(self):
        img_name = self.images[state.current_index]

        class_counts = {cls: {'bbox': 0, 'polygon': 0} for cls in CLASSLIST}
        total_visible = 0
        total_hidden = 0

        for _, _, _, _, cls in state.bboxes:
            if cls in class_counts:
                class_counts[cls]['bbox'] += 1
                if self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                    total_visible += 1
                else:
                    total_hidden += 1

        for points, cls in state.polygons:
            if cls in class_counts:
                class_counts[cls]['polygon'] += 1
                if self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                    total_visible += 1
                else:
                    total_hidden += 1

        annotated_count = 0
        for img in self.images:
            xml_path = os.path.join(vocdataset_folder, os.path.splitext(img)[0] + ".xml")
            if os.path.exists(xml_path):
                annotated_count += 1

        progress_pct = (annotated_count / len(self.images) * 100) if self.images else 0

        # Build info text
        sep = "─" * 28
        info = f"{sep}\n"
        info += f" CURRENT IMAGE\n"
        info += f"{sep}\n"
        info += f" {state.current_index + 1} / {len(self.images)}\n"

        info += f" DATASET PROGRESS\n"
        info += f"{sep}\n"
        info += f" 📊 {annotated_count} / {len(self.images)} annotated\n"
        info += f" 📈 {progress_pct:.1f}% complete\n"
        info += f" ⏳ {len(self.images) - annotated_count} remaining\n\n"

        info += f"{sep}\n"
        info += f" CLASSES  ({len(CLASSLIST)})\n"
        info += f"{sep}\n"
        for i, cls in enumerate(CLASSLIST[:10]):
            cursor = "▸" if cls == state.current_class else " "
            info += f" {cursor} {i+1}. {cls}\n"
        if len(CLASSLIST) > 10:
            info += f"   … +{len(CLASSLIST)-10} more\n"

        info += f"{sep}\n"
        info += f" ANNOTATIONS\n"
        info += f"{sep}\n"
        total_annotations = len(state.bboxes) + len(state.polygons)
        info += f" 📦 Total:    {total_annotations}\n"
        info += f"    BBoxes:  {len(state.bboxes)}\n"
        info += f"    Polygon: {len(state.polygons)}\n"
        info += f" 👁  Visible:  {total_visible}\n"
        if total_hidden > 0:
            info += f" 🚫 Hidden:   {total_hidden}\n"
        info += "\n"

        info += f"{sep}\n"
        info += f" CURRENT IMAGE\n"
        info += f"{sep}\n"
        info += f" 📁 {img_name}\n"
        info += f" 📐 {state.orig_shape[1]}×{state.orig_shape[0]} px\n"
        info += f" 🔍 Scale: {state.display_scale:.2f}×\n"
        info += f" 🎨 Mode:  {state.annotation_mode.upper()}\n\n"

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', info)
        self.info_text.config(state=tk.DISABLED)

        if hasattr(self, 'visibility_count_labels'):
            self.update_visibility_counts()

        # Update bottom status bar progress
        if hasattr(self, 'statusbar_progress'):
            self.statusbar_progress.config(
                text=f"Annotated: {annotated_count} / {len(self.images)}  "
                     f"({progress_pct:.0f}%)"
            )

    # ----------------------------------------------------------
    #  Image list
    # ----------------------------------------------------------
    def refresh_image_list(self):
        self.all_images_for_search = self.images.copy()
        self.on_image_search_change()

    def on_image_search_change(self, *args):
        search_term = self.image_search_var.get().lower()
        if search_term.strip():
            filtered = [img for img in self.all_images_for_search
                        if search_term in img.lower()]
        else:
            filtered = self.all_images_for_search

        self.image_listbox.delete(0, tk.END)
        for img in filtered:
            self.image_listbox.insert(tk.END, img)

        current_img = self.images[state.current_index] if self.images else None
        if current_img:
            try:
                idx = filtered.index(current_img)
                self.image_listbox.select_set(idx)
                self.image_listbox.see(idx)
            except ValueError:
                pass

    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        selected_img = self.image_listbox.get(idx)
        try:
            actual_idx = self.images.index(selected_img)
        except ValueError:
            return

        self.save_current()
        state.current_index = actual_idx

        self.load_current_image()
        self.update_display()
        print(f"[GUI] Loaded image from list: {selected_img}")

    # ----------------------------------------------------------
    #  Mouse interaction
    # ----------------------------------------------------------
    def get_canvas_coords(self, event):
        if self.current_img_pil is None:
            return None, None
        x = event.x - self.canvas_offset_x
        y = event.y - self.canvas_offset_y
        img_w = self.current_img_pil.width
        img_h = self.current_img_pil.height
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        return x, y

    def on_mouse_down(self, event):
        global CLASSLIST
        CLASSLIST = class_manager.get_classes()

        if self.mask_mode:
            x, y = self.get_canvas_coords(event)
            if x is None:
                return
            x_orig = x / state.display_scale
            y_orig = y / state.display_scale
            self.add_mask_polygon_point(x_orig, y_orig)
            return

        if len(CLASSLIST) == 0:
            messagebox.showwarning("No Class",
                                   "No classes defined. Add a class before annotating.")
            return

        x, y = self.get_canvas_coords(event)
        if x is None:
            return

        state.ix, state.iy = x, y

        if state.annotation_mode == "polygon":
            self.handle_polygon_click(x, y, event.num)
        else:
            # Shift + drag → mulai select box
            if event.state & 0x1:
                state.selected_bboxes = set()
                state.selected_bbox = None
                self.shift_selecting = True
                self.shift_start_x, self.shift_start_y = x, y
                return

            # Klik biasa → clear multi-select
            state.selected_bboxes = set()
            state.selected_bbox = None
            state.resizing = False
            state.resize_mode = None

            if state.force_new_bbox:
                state.selected_bbox = None
                self.drawing = True
                self.start_x, self.start_y = x, y
                return

            # Convert display coordinates to original coordinates for collision detection
            x_orig = x / state.display_scale
            y_orig = y / state.display_scale
            clicked = []
            for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
                if x1 <= x_orig <= x2 and y1 <= y_orig <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    clicked.append((area, i))

            if clicked:
                _, state.selected_bbox = min(clicked, key=lambda a: a[0])
                x1, y1, x2, y2, _ = state.bboxes[state.selected_bbox]
                # Convert to display coordinates for handle detection
                x1_disp = int(x1 * state.display_scale)
                y1_disp = int(y1 * state.display_scale)
                x2_disp = int(x2 * state.display_scale)
                y2_disp = int(y2 * state.display_scale)
                handle_size = max(12, int(12 * max(state.display_scale, 0.5)))
                mid_x = (x1_disp + x2_disp) // 2
                mid_y = (y1_disp + y2_disp) // 2
                if abs(x - x1_disp) < handle_size and abs(y - y1_disp) < handle_size:
                    state.resizing = True
                    state.resize_mode = 'tl'
                elif abs(x - mid_x) < handle_size and abs(y - y1_disp) < handle_size:
                    state.resizing = True
                    state.resize_mode = 't'
                elif abs(x - x2_disp) < handle_size and abs(y - y1_disp) < handle_size:
                    state.resizing = True
                    state.resize_mode = 'tr'
                elif abs(x - x2_disp) < handle_size and abs(y - mid_y) < handle_size:
                    state.resizing = True
                    state.resize_mode = 'r'
                elif abs(x - x1_disp) < handle_size and abs(y - y2_disp) < handle_size:
                    state.resizing = True
                    state.resize_mode = 'bl'
                elif abs(x - mid_x) < handle_size and abs(y - y2_disp) < handle_size:
                    state.resizing = True
                    state.resize_mode = 'b'
                elif abs(x - x2_disp) < handle_size and abs(y - y2_disp) < handle_size:
                    state.resizing = True
                    state.resize_mode = 'br'
                elif abs(x - x1_disp) < handle_size and abs(y - mid_y) < handle_size:
                    state.resizing = True
                    state.resize_mode = 'l'
                else:
                    self.moving = True
                self.update_display()
                return

            self.drawing = True
            self.start_x, self.start_y = x, y

    def draw_dashed_line(self, img_array, pt1, pt2, color, dash_length=6, gap_length=4, thickness=1):
        """Draw dashed line on image array"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate total distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance == 0:
            return
        
        # Unit direction vector
        ux = (x2 - x1) / distance
        uy = (y2 - y1) / distance
        
        # Draw dashed segments
        segment_length = dash_length + gap_length
        num_segments = int(distance / segment_length) + 1
        
        for i in range(num_segments):
            start_dist = i * segment_length
            end_dist = min(start_dist + dash_length, distance)
            
            # Calculate segment endpoints
            seg_x1 = int(x1 + ux * start_dist)
            seg_y1 = int(y1 + uy * start_dist)
            seg_x2 = int(x1 + ux * end_dist)
            seg_y2 = int(y1 + uy * end_dist)
            
            cv2.line(img_array, (seg_x1, seg_y1), (seg_x2, seg_y2), color, thickness)

    def draw_crosshair(self, img_array, mouse_x, mouse_y, color=(255, 255, 255)):
        """Draw crosshair at mouse position with dashed lines"""
        h, w = img_array.shape[:2]
        
        # Vertical line (top to bottom)
        self.draw_dashed_line(img_array, (mouse_x, 0), (mouse_x, h), color, dash_length=5, gap_length=3)
        
        # Horizontal line (left to right)
        self.draw_dashed_line(img_array, (0, mouse_y), (w, mouse_y), color, dash_length=5, gap_length=3)

    def draw_dashed_rect(self, img_array, x1, y1, x2, y2, color):
        """Draw dashed rectangle for select box preview."""
        self.draw_dashed_line(img_array, (x1, y1), (x2, y1), color)
        self.draw_dashed_line(img_array, (x2, y1), (x2, y2), color)
        self.draw_dashed_line(img_array, (x2, y2), (x1, y2), color)
        self.draw_dashed_line(img_array, (x1, y2), (x1, y1), color)

    def on_mouse_drag(self, event):
        x, y = self.get_canvas_coords(event)
        if x is None:
            return
        
        if self.shift_selecting:
            img_array = np.array(self.current_img_pil)
            sx1 = min(self.shift_start_x, x)
            sy1 = min(self.shift_start_y, y)
            sx2 = max(self.shift_start_x, x)
            sy2 = max(self.shift_start_y, y)
            # Highlight bbox yang akan ke-select (preview cyan tipis)
            for i, (bx1, by1, bx2, by2, cls) in enumerate(state.bboxes):
                bx1d = int(bx1 * state.display_scale)
                by1d = int(by1 * state.display_scale)
                bx2d = int(bx2 * state.display_scale)
                by2d = int(by2 * state.display_scale)
                if bx1d < sx2 and bx2d > sx1 and by1d < sy2 and by2d > sy1:
                    cv2.rectangle(img_array, (bx1d, by1d), (bx2d, by2d), (255, 212, 0), 3)
            # Draw dashed select box
            self.draw_dashed_rect(img_array, sx1, sy1, sx2, sy2, (255, 212, 0))
            temp_img = Image.fromarray(img_array)
            self.photo = ImageTk.PhotoImage(temp_img)
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                      anchor=tk.NW, image=self.photo)
            return

        if state.annotation_mode == "polygon":
            if (state.polygon_editing_mode and
                    state.selected_polygon is not None and
                    state.selected_polygon_point is not None):
                x_orig = x / state.display_scale
                y_orig = y / state.display_scale
                polygon_points = state.polygons[state.selected_polygon][0]
                polygon_points = polygon_manager.move_polygon_point(
                    polygon_points, state.selected_polygon_point, (x_orig, y_orig))
                state.polygons[state.selected_polygon][0] = polygon_points
                self.update_display()
            else:
                if len(state.polygon_points_preview) > 0:
                    self.update_display()

        elif self.drawing:
            # Lightweight render: just current bbox preview with crosshair
            img_array = np.array(self.current_img_pil)
            
            # Draw bbox preview
            cv2.rectangle(img_array, (self.start_x, self.start_y), (x, y), (0, 200, 255), 2)
            
            # Draw crosshair at current mouse position
            self.draw_crosshair(img_array, x, y, color=(255, 255, 255))

            temp_img = Image.fromarray(img_array)
            self.photo = ImageTk.PhotoImage(temp_img)
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                      anchor=tk.NW, image=self.photo)

        elif self.moving and state.selected_bbox is not None:
            # Convert display delta to original scale delta
            dx = (x - state.ix) / state.display_scale
            dy = (y - state.iy) / state.display_scale
            state.bboxes[state.selected_bbox][0] += dx
            state.bboxes[state.selected_bbox][1] += dy
            state.bboxes[state.selected_bbox][2] += dx
            state.bboxes[state.selected_bbox][3] += dy
            state.ix, state.iy = x, y
            self.update_display()

        elif state.resizing and state.selected_bbox is not None:
            # Convert display coordinates to original coordinates
            x_orig = x / state.display_scale
            y_orig = y / state.display_scale
            x1, y1, x2, y2, cls = state.bboxes[state.selected_bbox]
            min_size = 5 / state.display_scale  # Minimum size in original coordinates
            if state.resize_mode == 'tl':
                state.bboxes[state.selected_bbox][0] = min(x_orig, x2 - min_size)
                state.bboxes[state.selected_bbox][1] = min(y_orig, y2 - min_size)
            elif state.resize_mode == 't':
                state.bboxes[state.selected_bbox][1] = min(y_orig, y2 - min_size)
            elif state.resize_mode == 'tr':
                state.bboxes[state.selected_bbox][2] = max(x_orig, x1 + min_size)
                state.bboxes[state.selected_bbox][1] = min(y_orig, y2 - min_size)
            elif state.resize_mode == 'r':
                state.bboxes[state.selected_bbox][2] = max(x_orig, x1 + min_size)
            elif state.resize_mode == 'bl':
                state.bboxes[state.selected_bbox][0] = min(x_orig, x2 - min_size)
                state.bboxes[state.selected_bbox][3] = max(y_orig, y1 + min_size)
            elif state.resize_mode == 'b':
                state.bboxes[state.selected_bbox][3] = max(y_orig, y1 + min_size)
            elif state.resize_mode == 'br':
                state.bboxes[state.selected_bbox][2] = max(x_orig, x1 + min_size)
                state.bboxes[state.selected_bbox][3] = max(y_orig, y1 + min_size)
            elif state.resize_mode == 'l':
                state.bboxes[state.selected_bbox][0] = min(x_orig, x2 - min_size)
            self.update_display()

    def on_mouse_up(self, event):
        global CLASSLIST
        CLASSLIST = class_manager.get_classes()

        if len(CLASSLIST) == 0:
            messagebox.showwarning("No Class",
                                   "No classes defined. Add a class before annotating.")
            return

        x, y = self.get_canvas_coords(event)
        if x is None:
            return
        
        if self.shift_selecting:
            self.shift_selecting = False
            sx1 = min(self.shift_start_x, x)
            sy1 = min(self.shift_start_y, y)
            sx2 = max(self.shift_start_x, x)
            sy2 = max(self.shift_start_y, y)
            state.selected_bboxes = set()
            state.selected_bbox = None
            if sx2 - sx1 > 4 and sy2 - sy1 > 4:
                for i, (bx1, by1, bx2, by2, cls) in enumerate(state.bboxes):
                    bx1d = int(bx1 * state.display_scale)
                    by1d = int(by1 * state.display_scale)
                    bx2d = int(bx2 * state.display_scale)
                    by2d = int(by2 * state.display_scale)
                    if bx1d < sx2 and bx2d > sx1 and by1d < sy2 and by2d > sy1:
                        state.selected_bboxes.add(i)
                print(f"[GUI] Select box: {len(state.selected_bboxes)} bbox(es) selected")
            self.update_display()
            self.update_info()
            return

        if state.annotation_mode == "polygon":
            state.polygon_editing_mode = False
            self.update_display()
        elif self.drawing:
            # Convert display coordinates to original image coordinates
            # self.start_x/y and x/y are in display pixel space (scaled by display_scale)
            x1_display = min(self.start_x, x)
            y1_display = min(self.start_y, y)
            x2_display = max(self.start_x, x)
            y2_display = max(self.start_y, y)
            
            # Critical: divide by current scale
            scale = max(state.display_scale, 0.001)  # Prevent division by zero
            x1_orig = int(round(x1_display / scale))
            y1_orig = int(round(y1_display / scale))
            x2_orig = int(round(x2_display / scale))
            y2_orig = int(round(y2_display / scale))
            
            # Clamp STRICTLY to valid bounds
            x1_orig = max(0, min(x1_orig, state.orig_width - 2))
            y1_orig = max(0, min(y1_orig, state.orig_height - 2))
            x2_orig = max(x1_orig + 2, min(x2_orig, state.orig_width))
            y2_orig = max(y1_orig + 2, min(y2_orig, state.orig_height))
            
            w, h = abs(x2_orig - x1_orig), abs(y2_orig - y1_orig)

            if w >= 8 and h >= 8:
                # Ensure coords are integers
                bbox = [int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig), state.current_class]
                state.bboxes.append(bbox)
                state.selected_bbox = None       
                state.selected_bboxes = set()    
                state.selected_bbox = None
                print(f"[GUI] Added bbox: {state.current_class}")
                print(f"      Display: ...")
                print(f"      Stored (original): {bbox}")
                self.update_info()
            else:
                print("[GUI] Skipped tiny bbox (<8px)")

        self.drawing = False
        self.moving = False
        state.resizing = False
        state.force_new_bbox = False
        self.update_display()
        self.update_force_label()

    def on_mouse_move(self, event):
        """Show crosshair when hovering over canvas (not drawing)"""
        if self.current_img_pil is None or self.drawing:
            return
        
        x, y = self.get_canvas_coords(event)
        if x is None:
            return
        
        # Only show crosshair if not drawing/moving/resizing and in bbox mode
        if not self.drawing and not self.moving and not state.resizing and state.annotation_mode == "bbox":
            img_array = np.array(self.current_img_pil)
            
            # Draw all existing bboxes
            for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
                if not self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                    continue
                class_index = class_manager.get_class_index(cls)
                if class_index == -1:
                    class_index = 0
                colors = class_manager.get_colors()
                color = colors[class_index] if class_index < len(colors) else (255, 0, 0)
                if i == state.selected_bbox:          
                    color = (60, 60, 200)
                elif i in getattr(state, 'selected_bboxes', set()):
                    color = (255, 212, 0)             
                color_rgb = (color[2], color[1], color[0])
                x1_disp = int(x1 * state.display_scale)
                y1_disp = int(y1 * state.display_scale)
                x2_disp = int(x2 * state.display_scale)
                y2_disp = int(y2 * state.display_scale)
                self.draw_bbox_outline(img_array, x1_disp, y1_disp, x2_disp, y2_disp, color_rgb)

                if i == state.selected_bbox:
                    self.draw_bbox_handles(img_array, x1_disp, y1_disp, x2_disp, y2_disp)
                
                # Draw class label text if enabled
                if state.show_bbox_text:
                    label = cls
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_array,
                                  (x1_disp, y1_disp - label_h - 8), (x1_disp + label_w + 8, y1_disp),
                                  color_rgb, -1)
                    cv2.putText(img_array, label, (x1_disp + 4, y1_disp - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
            
            # Draw all existing polygons
            for i, (points_orig, cls) in enumerate(state.polygons):
                if not self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                    continue
                class_index = class_manager.get_class_index(cls)
                if class_index == -1:
                    class_index = 0
                colors = class_manager.get_colors()
                color = colors[class_index] if class_index < len(colors) else (255, 0, 0)
                if i == state.selected_bbox:
                    color = (60, 60, 200)
                elif i in getattr(state, 'selected_bboxes', set()):
                    color = (255, 212, 0)
                color_rgb = (color[2], color[1], color[0])
                is_selected = (i == state.selected_polygon)
                points_display = [
                    (int(px * state.display_scale), int(py * state.display_scale))
                    for px, py in points_orig
                ]
                img_array = polygon_manager.draw_polygon(
                    img_array, points_display, color_rgb,
                    thickness=2, is_selected=is_selected,
                    show_points=False
                )
            
            # Draw crosshair at current mouse position
            self.draw_crosshair(img_array, x, y, color=(255, 255, 255))
            
            temp_img = Image.fromarray(img_array)
            self.photo = ImageTk.PhotoImage(temp_img)
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                      anchor=tk.NW, image=self.photo)

    def on_right_click(self, event):
        if self.mask_mode:
            self.cancel_mask_polygon()
            return
        if state.annotation_mode != "polygon":
            return
        x, y = self.get_canvas_coords(event)
        if x is None:
            return
        self.handle_polygon_click(x, y, button_num=3)

    def on_double_click(self, event):
        if self.mask_mode:
            if len(self.mask_polygon_points) > 0:
                self.finish_mask_polygon()
            return

        x, y = self.get_canvas_coords(event)
        if x is None:
            return

        for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.show_class_rename_dialog('bbox', i, cls)
                return

        for i, (points_orig, cls) in enumerate(state.polygons):
            points_display = [(int(px * state.display_scale), int(py * state.display_scale))
                              for px, py in points_orig]
            if polygon_manager.point_in_polygon((x, y), points_display):
                self.show_class_rename_dialog('polygon', i, cls)
                return

        if state.annotation_mode != "polygon":
            return
        if len(state.polygon_points_preview) > 0:
            self.finish_polygon_drawing()

    def on_canvas_resize(self, event):
        if self.current_img_pil is not None:
            self.canvas_offset_x = (event.width - self.current_img_pil.width) // 2
            self.canvas_offset_y = (event.height - self.current_img_pil.height) // 2
            self.update_display()

    def on_canvas_scroll(self, event):
        """Handle mouse scroll for zoom in/out with smart panning to mouse position"""
        # Abort any ongoing drawing/moving when zooming
        if self.drawing or self.moving or state.resizing:
            self.drawing = False
            self.moving = False
            state.resizing = False
            self.update_display()
            return
        
        if self.current_img_pil is None:
            return
        
        # Get current canvas dimensions
        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 100 or canvas_h < 100:
            return
        
        # Get mouse position on canvas
        mouse_x = event.x
        mouse_y = event.y
        
        # Determine zoom direction
        # event.delta > 0 for scroll up (zoom in), < 0 for scroll down (zoom out)
        # Button-4 is scroll up (Linux), Button-5 is scroll down (Linux)
        if event.num == 4 or event.delta > 0:
            zoom_factor = self.zoom_step  # Zoom in
            is_zoom_in = True
        elif event.num == 5 or event.delta < 0:
            zoom_factor = 1.0 / self.zoom_step  # Zoom out
            is_zoom_in = False
        else:
            return
        
        # Calculate new scale
        new_scale = state.display_scale * zoom_factor
        
        # Calculate min and max scale
        # Min: 1x (original size, but don't go below fitting in container)
        min_scale = min(1.0, min(
            (canvas_w - 40) / state.orig_width,
            (canvas_h - 40) / state.orig_height
        ))
        
        # Max: fit image within container, but allow up to 5x zoom
        max_scale_fit = min(
            canvas_w / state.orig_width,
            canvas_h / state.orig_height
        )
        max_scale = max(max_scale_fit, 5.0)
        
        # Clamp scale
        new_scale = max(min_scale, min(new_scale, max_scale))
        
        # If scale didn't change, return early
        if new_scale == state.display_scale:
            return
        
        # Get current image position on canvas (before zoom)
        old_img_x = self.canvas_offset_x
        old_img_y = self.canvas_offset_y
        old_scale = state.display_scale
        
        # Update display scale
        state.display_scale = new_scale
        
        # Recalculate display dimensions
        new_w = int(state.orig_width * new_scale)
        new_h = int(state.orig_height * new_scale)
        state.display_width = new_w
        state.display_height = new_h
        
        # Resize frame
        state.frame = cv2.resize(
            cv2.imread(os.path.join(input_folder, self.images[state.current_index])),
            (new_w, new_h)
        )
        
        # Update PIL image for display
        frame_rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        self.current_img_pil = Image.fromarray(frame_rgb)
        
        # === Calculate mouse position in image space (pakai scale lama) ===
        mouse_img_x = (mouse_x - old_img_x) / old_scale
        mouse_img_y = (mouse_y - old_img_y) / old_scale

        # Clamp ke batas image
        mouse_img_x = max(0, min(mouse_img_x, state.orig_width))
        mouse_img_y = max(0, min(mouse_img_y, state.orig_height))

        # === Apply ke scale baru (zoom in & zoom out sama aja) ===
        new_offset_x = mouse_x - (mouse_img_x * new_scale)
        new_offset_y = mouse_y - (mouse_img_y * new_scale)

        # Horizontal
        if new_w < canvas_w:
            # Kalau image lebih kecil → center
            new_offset_x = (canvas_w - new_w) / 2
        else:
            # Kalau lebih besar → clamp biar gak keluar
            new_offset_x = max(-(new_w - 10), min(new_offset_x, canvas_w - 10))

        # Vertical
        if new_h < canvas_h:
            new_offset_y = (canvas_h - new_h) / 2
        else:
            new_offset_y = max(-(new_h - 10), min(new_offset_y, canvas_h - 10))

        # === Apply ke canvas ===
        self.canvas_offset_x = int(new_offset_x)
        self.canvas_offset_y = int(new_offset_y)
        
        # Update display
        self.update_display()

    # ----------------------------------------------------------
    #  Polygon handling
    # ----------------------------------------------------------
    def handle_polygon_click(self, x, y, button_num):
        x_orig = x / state.display_scale
        y_orig = y / state.display_scale

        for poly_idx, (points_orig, cls) in enumerate(state.polygons):
            points_display = [(int(px * state.display_scale), int(py * state.display_scale))
                              for px, py in points_orig]
            point_idx = polygon_manager.get_closest_point((x, y), points_display, threshold=8)
            if point_idx != -1:
                if button_num == 1:
                    state.selected_polygon = poly_idx
                    state.selected_polygon_point = point_idx
                    state.polygon_editing_mode = True
                    print(f"[POLYGON] Selected polygon {poly_idx} point {point_idx}")
                    self.update_display()
                    return
                elif button_num == 3:
                    if len(points_orig) > 3:
                        points_orig = polygon_manager.delete_polygon_point(points_orig, point_idx)
                        state.polygons[poly_idx][0] = points_orig
                        print(f"[POLYGON] Deleted point {point_idx} from polygon {poly_idx}")
                        self.update_display()
                    return

        if button_num == 1:
            if len(state.polygon_points_preview) >= 3:
                first_pt = state.polygon_points_preview[0]
                first_disp = (int(first_pt[0] * state.display_scale),
                              int(first_pt[1] * state.display_scale))
                distance = ((x - first_disp[0]) ** 2 + (y - first_disp[1]) ** 2) ** 0.5
                if distance < 10:
                    print(f"[POLYGON] Auto-close! Distance: {distance:.1f}px")
                    self.finish_polygon_drawing()
                    return

            state.polygon_points_preview.append((x_orig, y_orig))
            print(f"[POLYGON] Added point {len(state.polygon_points_preview)}: "
                  f"({x_orig:.2f}, {y_orig:.2f}) [original scale]")
            self.update_display()

    def cancel_polygon_drawing(self):
        if len(state.polygon_points_preview) > 0:
            state.polygon_points_preview = []
            print("[POLYGON] Cancelled drawing")
            self.update_display()

    def close_main_gui(self):
        self.root.quit()
        self.root.destroy()

    def back_to_workspace(self):
        """Close this workspace and hand control back to the workspace picker."""
        if messagebox.askyesno(
            "Back to Workspaces",
            "Leave this workspace and return to the workspace list?",
            parent=self.root
        ):
            self.close_main_gui()

    def finish_polygon_drawing(self):
        if len(state.polygon_points_preview) < 3:
            messagebox.showwarning("Invalid Polygon",
                                   "Polygon must have at least 3 points.")
            return
        if not polygon_manager.is_valid_polygon(state.polygon_points_preview):
            messagebox.showwarning("Invalid Polygon",
                                   "Polygon is invalid or has duplicate points.")
            return

        state.polygons.append([state.polygon_points_preview.copy(), state.current_class])
        print(f"[POLYGON] Finished polygon with {len(state.polygon_points_preview)} points. "
              f"Class: {state.current_class}")

        state.polygon_points_preview = []
        state.selected_polygon = len(state.polygons) - 1
        self.update_display()
        self.update_info()

    # ----------------------------------------------------------
    #  Annotation mode toggle
    # ----------------------------------------------------------
    def toggle_annotation_mode(self):
        if state.annotation_mode == "bbox":
            state.annotation_mode = "polygon"
            print("[GUI] Switched to POLYGON mode")
            if hasattr(self, 'statusbar_mode'):
                self.statusbar_mode.config(text="  🔷 POLYGON MODE  ",
                                            bg=C_AMBER, fg=C_ON_ACCENT)
            messagebox.showinfo("Mode Changed",
                                "Switched to POLYGON mode.\n\n"
                                "Click to add points (min 3)\n"
                                "Double-click or Enter to finish\n"
                                "Right-click on point to delete")
        else:
            state.annotation_mode = "bbox"
            print("[GUI] Switched to BBOX mode")
            if hasattr(self, 'statusbar_mode'):
                self.statusbar_mode.config(text="  📦 BBOX MODE  ",
                                            bg=C_ACCENT, fg=C_ON_ACCENT)
            messagebox.showinfo("Mode Changed",
                                "Switched back to BBOX mode.\n"
                                "Press B to create a bounding box.")
            self.cancel_polygon_drawing()

        self.update_display()
        self.update_info()

    # ----------------------------------------------------------
    #  Class rename dialog
    # ----------------------------------------------------------
    def show_class_rename_dialog(self, annotation_type, index, current_class):
        global CLASSLIST
        CLASSLIST = class_manager.get_classes()
        if not CLASSLIST:
            messagebox.showwarning("No Classes", "No classes defined. Add one first!")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Rename Class")
        dialog.geometry("460x260")
        dialog.configure(bg=C_BASE)
        dialog.update_idletasks()
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Header
        hdr = tk.Frame(dialog, bg=C_ACCENT, height=46)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="✏️  Rename Object Class",
                 font=('Segoe UI', 12, 'bold'),
                 bg=C_ACCENT, fg=C_ON_ACCENT).pack(side=tk.LEFT, padx=16, pady=10)

        body = tk.Frame(dialog, bg=C_BASE, padx=24, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

        # Current class badge
        cur_row = tk.Frame(body, bg=C_BASE)
        cur_row.pack(fill=tk.X, pady=(0, 10))
        tk.Label(cur_row, text="Current:", font=('Segoe UI', 9),
                 bg=C_BASE, fg=C_TXT2).pack(side=tk.LEFT)
        tk.Label(cur_row, text=f"  {current_class}  ",
                 font=('Segoe UI', 10, 'bold'),
                 bg=C_AMBER, fg=C_ON_ACCENT,
                 padx=8, pady=3).pack(side=tk.LEFT, padx=8)

        tk.Frame(body, bg=C_BORDER, height=1).pack(fill=tk.X, pady=(0, 10))

        tk.Label(body, text="Select New Class:",
                 font=('Segoe UI', 10, 'bold'),
                 bg=C_BASE, fg=C_TXT1).pack(anchor='w', pady=(0, 8))

        class_var = tk.StringVar(value=current_class)

        # Style the combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Jelibox.TCombobox',
                        fieldbackground=C_CARD2,
                        background=C_CARD2,
                        foreground=C_TXT1,
                        selectbackground=C_ACCENT,
                        selectforeground=C_ON_ACCENT,
                        bordercolor=C_BORDER,
                        darkcolor=C_CARD2,
                        lightcolor=C_CARD2,
                        arrowcolor=C_ACCENT)

        dropdown = ttk.Combobox(body, textvariable=class_var, values=CLASSLIST,
                                state='readonly', font=('Segoe UI', 11), width=38,
                                style='Jelibox.TCombobox')
        dropdown.pack(fill=tk.X, ipady=4)

        def on_ok():
            new_class = class_var.get()
            if new_class and new_class != current_class:
                if annotation_type == 'bbox':
                    state.bboxes[index][4] = new_class
                    print(f"[GUI] Changed bbox class: {current_class} → {new_class}")
                elif annotation_type == 'polygon':
                    state.polygons[index][1] = new_class
                    print(f"[GUI] Changed polygon class: {current_class} → {new_class}")
                self.save_current()
                self.update_display()
                self.update_info()
                dialog.destroy()
                messagebox.showinfo("Success",
                                    f"Class updated: {current_class} → {new_class}",
                                    parent=self.root)
            else:
                if new_class == current_class:
                    messagebox.showinfo("No Change",
                                        f"Same class selected: {current_class}",
                                        parent=self.root)
                dialog.destroy()

        btn_row = tk.Frame(body, bg=C_BASE)
        btn_row.pack(fill=tk.X, pady=(16, 0))

        self._btn(btn_row, "✕  Cancel", dialog.destroy,
                  C_CARD2, C_TXT2, font_size=10).pack(side=tk.LEFT, ipady=7, ipadx=14)
        self._btn(btn_row, "✓  Apply", on_ok,
                  C_ACCENT, C_ON_ACCENT, font_size=10, bold=True).pack(
            side=tk.RIGHT, ipady=7, ipadx=16)

        dropdown.focus()

    # ----------------------------------------------------------
    #  Masking / ZeroFill
    # ----------------------------------------------------------
    def toggle_mask_mode(self):
        self.mask_mode = not self.mask_mode
        if self.mask_mode:
            self.mask_btn.config(text="⬛ ZeroFill: ON", bg=C_ACCENT, fg=C_ON_ACCENT,
                                 activebackground=C_ACCENT, activeforeground=C_ON_ACCENT)
            self.mask_polygon_points = []
            print("[MASKING] ON — Click to draw polygon, double-click to finish")
            messagebox.showinfo("ZeroFill Mode",
                                "ZeroFill masking mode activated!\n\n"
                                "Click to add polygon points (min 3)\n"
                                "Double-click or Enter to finish\n"
                                "The area inside will be blackened\n"
                                "Press ZeroFill again to cancel")
        else:
            self.mask_btn.config(text="⬛ ZeroFill: OFF", bg=C_CARD2, fg=C_TXT1,
                                 activebackground=C_BORDER, activeforeground=C_TXT1)
            self.mask_polygon_points = []
            print("[MASKING] OFF")
        self.update_display()

    def add_mask_polygon_point(self, x, y):
        if not self.mask_mode:
            return
        if len(self.mask_polygon_points) >= 3:
            first_pt = self.mask_polygon_points[0]
            first_disp = (int(first_pt[0] * state.display_scale),
                          int(first_pt[1] * state.display_scale))
            curr_disp = (int(x * state.display_scale),
                         int(y * state.display_scale))
            distance = ((curr_disp[0] - first_disp[0]) ** 2 +
                        (curr_disp[1] - first_disp[1]) ** 2) ** 0.5
            if distance < 10:
                print(f"[MASKING] Auto-close detected! Distance: {distance:.1f}px")
                self.finish_mask_polygon()
                return

        self.mask_polygon_points.append((x, y))
        print(f"[MASKING] Added point {len(self.mask_polygon_points)}: ({x:.2f}, {y:.2f})")
        self.update_display()

    def finish_mask_polygon(self):
        if len(self.mask_polygon_points) < 3:
            messagebox.showwarning("Invalid Polygon",
                                   "Polygon must have at least 3 points.")
            return
        if not polygon_manager.is_valid_polygon(self.mask_polygon_points):
            messagebox.showwarning("Invalid Polygon",
                                   "Polygon is invalid or has duplicate points.")
            return

        self.masked_regions.append(self.mask_polygon_points.copy())
        print(f"[MASKING] Finished polygon with {len(self.mask_polygon_points)} points")
        self.mask_polygon_points = []
        self.apply_mask_to_current_image()
        self.update_display()

    def apply_mask_to_current_image(self):
        if not self.masked_regions or self.current_img_pil is None:
            return
        img_name = self.images[state.current_index]
        img_path = os.path.join(input_folder, img_name)
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print(f"[MASKING] Error: Could not read {img_name}")
            return
        for polygon_points in self.masked_regions:
            pts = np.array(polygon_points, dtype=np.int32)
            cv2.fillPoly(orig_img, [pts], (0, 0, 0))
        cv2.imwrite(img_path, orig_img)
        print(f"[MASKING] Applied {len(self.masked_regions)} mask(s) to {img_name}")
        self.load_current_image()

    def cancel_mask_polygon(self):
        if len(self.mask_polygon_points) > 0:
            self.mask_polygon_points = []
            print("[MASKING] Cancelled drawing")
            self.update_display()

    # ----------------------------------------------------------
    #  Class selection
    # ----------------------------------------------------------
    def on_class_select(self, event):
        selection = self.class_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(CLASSLIST):
                state.current_class = CLASSLIST[idx]
                print(f"[GUI] Selected class: {state.current_class}")
                self.update_info()

    def select_class_by_number(self, idx):
        if 0 <= idx < len(CLASSLIST):
            self.class_listbox.selection_clear(0, tk.END)
            self.class_listbox.select_set(idx)
            self.class_listbox.see(idx)
            state.current_class = CLASSLIST[idx]
            print(f"[GUI] Quick selected class {idx+1}: {state.current_class}")
            self.update_info()

    # ----------------------------------------------------------
    #  Image navigation
    # ----------------------------------------------------------
    def next_image(self):
        self.save_current()
        state.current_index = (state.current_index + 1) % len(self.images)
        self.load_current_image()
        self.update_display()

    def prev_image(self):
        self.save_current()
        state.current_index = (state.current_index - 1) % len(self.images)
        self.load_current_image()
        self.update_display()

    def save_current(self):
        """Save current annotations - force zoom to 1x (normalized scale) first"""
        img_name = self.images[state.current_index]
        
        # Store current zoom level untuk restore setelah save
        saved_zoom = state.display_scale
        
        # Force zoom out to 1.0 sebelum save untuk normalize coordinates
        if state.display_scale != 1.0:

            # Show a blank saving overlay on canvas to avoid flashing zoom-out
            self.canvas.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill='#000000', outline='')
            try:
                self.canvas.create_text(canvas_w // 2, canvas_h // 2, text="Saving...",
                                        fill='#ffffff', font=('Segoe UI', 20, 'bold'))
            except Exception:
                # Fallback font on systems without Segoe UI
                self.canvas.create_text(canvas_w // 2, canvas_h // 2, text="Saving...",
                                        fill='#ffffff')
            self.canvas.update_idletasks()

            # Reset ke scale 1x for normalization (do NOT re-render to screen)
            state.display_scale = 1.0
            state.display_width = state.orig_width
            state.display_height = state.orig_height

            # Reload image at 1x into memory (but don't call update_display yet)
            orig_img = cv2.imread(os.path.join(input_folder, img_name))
            if orig_img is not None:
                state.frame = orig_img.copy()
                frame_rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
                self.current_img_pil = Image.fromarray(frame_rgb)

            # Center offsets adjusted for 1x (kept internal)
            self.canvas_offset_x = (canvas_w - state.orig_width) // 2
            self.canvas_offset_y = (canvas_h - state.orig_height) // 2
        
        save_and_backup_bboxes(img_name, state.orig_shape, CLASSLIST)
        
        # Restore zoom level setelah save selesai
        if saved_zoom != 1.0:
            state.display_scale = saved_zoom
            state.display_width = int(state.orig_width * saved_zoom)
            state.display_height = int(state.orig_height * saved_zoom)
            
            # Resize frame ke zoom level
            orig_img = cv2.imread(os.path.join(input_folder, img_name))
            if orig_img is not None:
                state.frame = cv2.resize(orig_img, (state.display_width, state.display_height))
                frame_rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
                self.current_img_pil = Image.fromarray(frame_rgb)
            # Recompute canvas offsets so image is centered after restoring zoom
            try:
                self.canvas.update_idletasks()
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                self.canvas_offset_x = (canvas_w - state.display_width) // 2
                self.canvas_offset_y = (canvas_h - state.display_height) // 2
            except Exception:
                # keep previous offsets if querying canvas fails
                pass

    # ----------------------------------------------------------
    #  Annotation editing
    # ----------------------------------------------------------
    def delete_selected_bbox(self):
        if state.selected_bbox is not None:
            deleted_class = state.bboxes[state.selected_bbox][4]
            del state.bboxes[state.selected_bbox]
            state.selected_bbox = None
            print(f"[GUI] Deleted bbox: {deleted_class}")
            self.update_display()
            self.update_info()
        
        elif len(getattr(state, 'selected_bboxes', set())) > 0:
            selected = sorted(getattr(state, 'selected_bboxes', set()), reverse=True)

            for idx in selected:
                if 0 <= idx < len(state.bboxes):
                    del state.bboxes[idx]

            state.selected_bboxes.clear()
            state.selected_bbox = None
            self.update_display()
            self.update_info()
                    
        elif state.selected_polygon is not None:
            deleted_class = state.polygons[state.selected_polygon][1]
            del state.polygons[state.selected_polygon]
            state.selected_polygon = None
            state.selected_polygon_point = None
            print(f"[GUI] Deleted polygon: {deleted_class}")
            self.update_display()
            self.update_info()

    def change_class_selected(self):
        if state.selected_bbox is not None and CLASSLIST:
            current_class = state.bboxes[state.selected_bbox][4]
            idx = class_manager.get_class_index(current_class)
            if idx == -1:
                idx = 0
            idx = (idx + 1) % len(CLASSLIST)
            new_class = CLASSLIST[idx]
            state.bboxes[state.selected_bbox][4] = new_class
            print(f"[GUI] Changed bbox class: {current_class} → {new_class}")
            self.update_display()
            self.update_info()
        elif state.selected_polygon is not None and CLASSLIST:
            current_class = state.polygons[state.selected_polygon][1]
            idx = class_manager.get_class_index(current_class)
            if idx == -1:
                idx = 0
            idx = (idx + 1) % len(CLASSLIST)
            new_class = CLASSLIST[idx]
            state.polygons[state.selected_polygon][1] = new_class
            print(f"[GUI] Changed polygon class: {current_class} → {new_class}")
            self.update_display()
            self.update_info()

    # ----------------------------------------------------------
    #  Training
    # ----------------------------------------------------------
    def monitor_training_process(self):
        if hasattr(state, "training_process"):
            process = state.training_process
            if process.poll() is None:
                self.root.after(2000, self.monitor_training_process)
            else:
                state.training_running = False
                exit_code = process.returncode
                if exit_code == 0:
                    messagebox.showinfo("Training Finished",
                                        "✅ YOLO Training completed successfully!",
                                        parent=self.root)
                    print("[GUI] Training completed with exit code 0 (success)")
                else:
                    messagebox.showwarning(
                        "Training Finished with Error",
                        f"⚠️ Training process exited with code: {exit_code}\n\n"
                        f"Check the terminal window for error details.",
                        parent=self.root
                    )
                    print(f"[GUI] Training exited with code {exit_code} (possible error)")

    def start_training(self):
        global CLASSLIST, model_folder, model_path
        self.save_current()
        CLASSLIST = class_manager.get_classes()

        if hasattr(state, "training_process") and state.training_process is not None:
            if state.training_process.poll() is None:
                messagebox.showwarning("Training Running",
                                       "Training already running. Please wait...",
                                       parent=self.root)
                return
            else:
                state.training_running = False
                state.training_process = None

        annotationMode = state.annotation_mode
        model_type = "seg" if annotationMode == "polygon" else "detect"
        model_type_display = ("🔷 SEGMENTATION" if annotationMode == "polygon"
                              else "📦 DETECTION")
        print(f"[GUI] Training mode: {model_type_display}")

        messagebox.showinfo("Training Model Type",
                            f"Detected model type:\n\n{model_type_display}\n\n"
                            f"{'Using Segmentation Model' if annotationMode == 'polygon' else 'Using Object Detection Model'}",
                            parent=self.root)

        config_dialog = TrainingConfigDialog(
            self.root,
            default_epoch=EPOCH,
            default_batch=BATCH,
            model_path=model_path,
            model_type=model_type,
            model_folder=model_folder
        )
        config = config_dialog.show()

        if config is None:
            return
        if config.get("action") == "removed":
            messagebox.showinfo("Model Removed",
                                "Existing model has been deleted.\n"
                                "Click Train again to choose a base model.",
                                parent=self.root)
            return

        current_epoch = config['epoch']
        current_batch = config['batch']
        current_imgsz = config['imgsz']
        selected_base = config.get('base_model')

        state.training_running = True
        train_script = os.path.join(os.path.dirname(__file__), "training.py")
        dataset_root = yolo_dataset_folder
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_folders = sorted({os.path.dirname(r['source'])
                                for r in self._find_workspace_images(image_exts)})
        if not image_folders:
            image_folders = [input_folder]
        class_args = list(CLASSLIST)

        cmd = [
            sys.executable, train_script,
            "--dataset_root",   dataset_root,
            "--images_folders", *image_folders,
            "--labels_folder",  yolo_labels_folder,
            "--model_type",     model_type,
            "--model_path",     model_path,
            "--model_folder",   model_folder,
            "--epochs",         str(current_epoch),
            "--batch",          str(current_batch),
            "--imgsz",          str(current_imgsz),
        ]
        if selected_base:
            cmd += ["--base_model", selected_base]
        cmd += ["--classlist"] + class_args

        try:
            if shutil.which("gnome-terminal"):
                terminal_cmd = ["gnome-terminal", "--wait", "--"] + cmd
            elif shutil.which("xterm"):
                terminal_cmd = ["xterm", "-hold", "-e"] + cmd
            elif shutil.which("konsole"):
                terminal_cmd = ["konsole", "-e"] + cmd
            else:
                terminal_cmd = cmd

            print(f"[GUI] Training command: {' '.join(terminal_cmd)}")
            print(f"[GUI] Training script path: {train_script}")
            print(f"[GUI] Script exists: {os.path.exists(train_script)}")

            process = subprocess.Popen(terminal_cmd)
            state.training_process = process

            messagebox.showinfo(
                "Training Started",
                f"Training started in new terminal!\n\n"
                f"🧠 Base Model  : {selected_base if selected_base else 'Existing model (continue)'}\n"
                f"📐 Input Size  : {current_imgsz} px\n"
                f"📊 Epochs      : {current_epoch}\n"
                f"📦 Batch Size  : {current_batch}\n\n"
                f"Check terminal for progress.",
                parent=self.root
            )
            self.monitor_training_process()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training:\n{str(e)}",
                                 parent=self.root)

        state.training_running = False

    # ----------------------------------------------------------
    #  Inference & misc actions
    # ----------------------------------------------------------
    def run_inference(self):
        self.save_current()
        print("[GUI] Running inference...")
        ok, message = inference_current(self.images, state.current_index)

        self.update_display()
        self.update_info()
        if not ok:
            messagebox.showwarning("Label Assistant", message, parent=self.root)
        print(f"[GUI] Inference finished: {message}")

    def toggle_theme(self):
        """Switch theme and relaunch the window so every widget picks up the new colors."""
        toggle_mode()
        try:
            # Same as navigating away - but don't mark an untouched, unannotated image as done.
            xml_path = os.path.join(vocdataset_folder, os.path.splitext(self.images[state.current_index])[0] + '.xml')
            if state.bboxes or state.polygons or os.path.exists(xml_path):
                self.save_current()
            from . import app_settings
            app_settings.set("resume", {"folder": input_folder, "image": self.images[state.current_index]})
        except Exception as exc:
            print(f"[GUI] Could not save state before theme restart: {exc}")
        self.root._jelibox_restart = True
        self.close_main_gui()

    def show_label_assistant(self):
        open_label_assistant(self.root)

    def launch_stream(self):
        """
        Launch the Streamlit live-detection demo in a separate process.
        Works on Windows, Linux, and macOS.
        """
 
        stream_script = os.path.join(os.path.dirname(__file__), "stream_app.py")

        # ── 0. Check that stream running ───────────────────────
        if self.stream_process is not None:
            # Jika proses masih ada dan belum mati (poll() me-return None)
            if self.stream_process.poll() is None:
                messagebox.showinfo(
                    "Stream — Already Running",
                    "Please check your browser tabs or the console window\n",
                    "at localhost:8501.",
                    parent=self.root
                )
                return
 
        # ── 1. Check that stream_app.py exists ───────────────────────
        if not os.path.exists(stream_script):
            messagebox.showerror(
                "Stream — File Not Found",
                f"stream_app.py not found at:\n{stream_script}\n\n"
                "Make sure stream_app.py is placed inside the package folder.",
                parent=self.root,
            )
            return
 
        # ── 2. Check that streamlit is installed ─────────────────────
        if importlib.util.find_spec("streamlit") is None:
            messagebox.showerror(
                "Stream — Missing Dependency",
                "Streamlit is not installed.\n\n"
                "Run the following command:\n"
                "    pip install streamlit",
                parent=self.root,
            )
            return
 
        # ── 3. Validate model_path ────────────────────────────────────
        if not model_path:
            messagebox.showerror(
                "Stream — No Model",
                "model_path is empty or None.\n\n"
                "Make sure a model is configured before opening the Stream.",
                parent=self.root,
            )
            return
 
        if not os.path.exists(model_path):
            if not messagebox.askyesno(
                "Stream — Model Not Found",
                f"Model file not found:\n{model_path}\n\n"
                "Stream will still open but Streamlit will show an error.\n"
                "Open anyway?",
                parent=self.root,
            ):
                return
 
        # ── 4. Build the command ──────────────────────────────────────
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            stream_script,
            "--theme.base", MODE,
            "--theme.primaryColor", C_ACCENT,
            "--theme.backgroundColor", C_BASE,
            "--theme.secondaryBackgroundColor", C_PANEL,
            "--theme.textColor", C_TXT1,
            "--",
            str(model_path),
        ]
        print(f"[STREAM] Launching: {' '.join(cmd)}")
 
        # ── 5. Spawn — platform-specific flags ───────────────────────
        try:
            if sys.platform == "win32":
                self.stream_process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                self.stream_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as exc:
            messagebox.showerror(
                "Stream — Launch Error",
                f"Failed to start the process:\n{exc}",
                parent=self.root,
            )
            return
 
        # ── 6. Peringatan Sukses ──────────────────────────────────────
        messagebox.showinfo(
            "Stream — Starting",
            "Streamlit is starting up...\n\n"
            "Your default browser will open automatically once the server is ready.\n\n"
            "A console window has been launched. Close that window to stop the stream.",
            parent=self.root,
        )

    def show_export_dialog(self):
        """Show export dialog and run model export in background thread."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Model")
        dialog.geometry("480x450")
        dialog.configure(bg=C_BASE)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        body = tk.Frame(dialog, bg=C_CARD, padx=12, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        # Export format dropdown
        tk.Label(body, text="Format:", bg=C_CARD, fg=C_TXT1).pack(anchor='w')
        formats = [
            ('TorchScript','torchscript'), ('ONNX','onnx'), ('OpenVINO','openvino'),
            ('TensorRT','engine'), ('CoreML','coreml'), ('TF SavedModel','saved_model'),
            ('TFLite','tflite'), ('TFJS','tfjs'), ('Paddle','paddle'), ('MNN','mnn'),
            ('NCNN','ncnn'), ('IMX500','imx'), ('RKNN','rknn'), ('ExecuTorch','executorch'),
            ('Axelera','axelera')
        ]
        fmt_var = tk.StringVar(value='onnx')
        fmt_names = [f[0] for f in formats]
        fmt_vals = {f[0]: f[1] for f in formats}
        fmt_combo = ttk.Combobox(body, values=fmt_names, state='readonly', font=('Segoe UI', 10))
        fmt_combo.set('ONNX')
        fmt_combo.pack(fill=tk.X, pady=(0,8))

        # Image size
        tk.Label(body, text="Image size (px):", bg=C_CARD, fg=C_TXT1).pack(anchor='w')
        imgsz_var = tk.IntVar(value=640)
        imgsz_entry = tk.Entry(body, textvariable=imgsz_var, font=('Segoe UI', 10), bg=C_CARD2, fg=C_TXT1)
        imgsz_entry.pack(fill=tk.X, pady=(0,8))

        # Options (checkboxes)
        opts_frame = tk.Frame(body, bg=C_CARD)
        opts_frame.pack(fill=tk.X, pady=(4,8))

        optimize_var = tk.BooleanVar(value=False)
        keras_var = tk.BooleanVar(value=False)
        half_var = tk.BooleanVar(value=False)
        int8_var = tk.BooleanVar(value=False)
        dynamic_var = tk.BooleanVar(value=False)
        simplify_var = tk.BooleanVar(value=False)
        end2end_var = tk.BooleanVar(value=False)

        # Helper to create a Checkbutton that visibly changes when active
        def make_cb(text, var):
            # Use text-based checkmark to control color reliably across platforms.
            display_text = f"   {text}"
            cb = tk.Checkbutton(
                opts_frame, text=display_text, variable=var,
                bg=C_CARD, fg=C_TXT1, anchor='w', indicatoron=True,
                selectcolor=C_CARD2, activebackground=C_CARD2, activeforeground=C_TXT1,
                font=('Segoe UI', 9)
            )
            cb.pack(anchor='w', fill=tk.X, padx=2, pady=2)


            return cb

        cb1 = make_cb('Optimize', optimize_var)
        cb2 = make_cb('Keras', keras_var)
        cb3 = make_cb('Half', half_var)
        cb4 = make_cb('INT8', int8_var)
        cb5 = make_cb('Dynamic', dynamic_var)
        cb6 = make_cb('Simplify', simplify_var)
        cb7 = make_cb('End2End', end2end_var)

        # Info link
        def open_docs(evt=None):
            webbrowser.open('https://docs.ultralytics.com/modes/export/#arguments')
        link = tk.Label(body, text='Ultralytics export arguments', fg=C_ACCENT, bg=C_CARD, cursor='hand2')
        link.pack(anchor='w', pady=(8,0))
        link.bind('<Button-1>', open_docs)

        status_lbl = tk.Label(body, text='', bg=C_CARD, fg=C_TXT1)
        status_lbl.pack(fill=tk.X, pady=(8,0))

        def do_export():
            fmt_name = fmt_combo.get()
            fmt = fmt_vals.get(fmt_name, 'onnx')
            try:
                imgsz = int(imgsz_var.get())
            except Exception:
                imgsz = 640

            opts = dict(
                fmt=fmt,
                imgsz=imgsz,
                optimize=optimize_var.get(),
                keras=keras_var.get(),
                half=half_var.get(),
                int8=int8_var.get(),
                dynamic=dynamic_var.get(),
                simplify=simplify_var.get(),
                end2end=end2end_var.get()
            )

            export_btn.config(state='disabled')
            status_lbl.config(text='Exporting...')

            def worker():
                success, msg, out_path = export_module.export_model(
                    model_path,
                    fmt=opts['fmt'],
                    imgsz=opts['imgsz'],
                    optimize=opts['optimize'],
                    keras=opts['keras'],
                    half=opts['half'],
                    int8=opts['int8'],
                    dynamic=opts['dynamic'],
                    simplify=opts['simplify'],
                    end2end=opts['end2end'],
                    save_dir=export_model_folder
                )

                def on_done():
                    export_btn.config(state='normal')
                    if success:
                        status_lbl.config(text=f'Export finished')
                        messagebox.showinfo('Export Completed', f'Export successful')
                    else:
                        status_lbl.config(text='Export failed')
                        messagebox.showerror('Export Failed', msg)

                self.root.after(10, on_done)

            threading.Thread(target=worker, daemon=True).start()

        export_btn = tk.Button(body, text='Export', command=do_export, bg=C_ACCENT, fg=C_ON_ACCENT)
        export_btn.pack(side=tk.RIGHT, pady=(12,0), ipadx=12, ipady=6)

        tk.Button(body, text='Close', command=dialog.destroy, bg=C_CARD2, fg=C_TXT1).pack(side=tk.RIGHT, pady=(12,0), padx=(8,0), ipadx=8, ipady=6)

    def repeat_annotations(self):
        global CLASSLIST, colorsPalette
        CLASSLIST = class_manager.get_classes()
        repeat_last_annotations(self.images, state.current_index, CLASSLIST)
        self.update_display()
        self.update_info()
        print("[GUI] Repeated annotations from previous image")

    def delete_image(self):
        img_name = self.images[state.current_index]
        if messagebox.askyesno(
            "Confirm Delete",
            f"Delete image '{img_name}' and all annotations?\n\nThis cannot be undone!",
            parent=self.root
        ):
            result = delete_current_image(self.images, state.current_index)
            if result[0] is None:
                messagebox.showinfo("Dataset Empty",
                                    "All images deleted. Exiting application...",
                                    parent=self.root)
                self.root.destroy()
                return
            state.current_index, self.images = result
            self.load_current_image()
            self.update_display()
            self.refresh_image_list()
            print(f"[GUI] Deleted image: {img_name}")