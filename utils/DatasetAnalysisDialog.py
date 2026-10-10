"""
Analyze Dataset window: charts (matplotlib inside Tk) of one workspace, from the numbers in dataset_analysis.py.

Tabs: Overview, Object size, Images, Resize table. The cards and the warning strip above the tabs always follow the
chosen view: the original sizes, or the sizes after a calculated resize (nothing is resized on disk).
"""
import gc
import tkinter as tk
from tkinter import ttk, messagebox

from . import app_settings
from . import dataset_analysis as da
from .progress_popup import show_progress_popup
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT, C_ACCENT_TINT, C_ON_ACCENT, C_AMBER,
                    C_RED, C_TXT1, C_TXT2, C_TXT3, MODE)

TOOLBAR_BG = "#E8E4DA"
LIMIT_RANGE = (1, 1024)
PALETTE = ("#4A5BF0", "#1F9D74", "#E08A1E", "#C9353A", "#8E5BD9", "#2B9BC9", "#B8A12A", "#D4568F")


def _px(pair):
    return "-" if pair is None else f"{pair[0]:.0f} x {pair[1]:.0f} px"


class DatasetAnalysisWindow:
    def __init__(self, parent, workspace, data=None):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except ImportError:
            messagebox.showerror("Analyze Dataset", "matplotlib is needed for the charts.\n"
                                 "Run the installer again, or:  python -m pip install matplotlib", parent=parent)
            return
        self._Figure, self._Canvas, self._Toolbar = Figure, FigureCanvasTkAgg, NavigationToolbar2Tk
        self.workspace = workspace
        self.data = data if data is not None else da.analyze_workspace(workspace)
        self.simulated = False
        self.limit = self._saved_limit()
        self.figures = {}
        self.top = tk.Toplevel(parent)
        self.top.title(f"Analyze Dataset - {workspace}")
        self.top.configure(bg=C_BASE)
        self.top.geometry("1180x780")
        self.top.minsize(900, 600)
        self.top.bind("<Destroy>", self._destroyed)
        self.size_var = tk.StringVar(value="640")
        self.mode_var = tk.StringVar(value=da.LETTERBOX)
        self._build()
        self.refresh()

    def _destroyed(self, event):
        # a Tk variable collected on another thread aborts Python ("Tcl_AsyncDelete"): collect on the Tk thread
        if event.widget is self.top:
            self.figures.clear()
            try:
                self.top.master.after_idle(gc.collect)
            except tk.TclError:
                pass

    @staticmethod
    def _saved_limit():
        """The small-object limit the user chose last time (px); 32 until they change it."""
        try:
            value = int(app_settings.get("small_limit", da.SMALL_LIMIT))
        except (TypeError, ValueError):
            return da.SMALL_LIMIT
        return value if LIMIT_RANGE[0] <= value <= LIMIT_RANGE[1] else da.SMALL_LIMIT

    # ----------------------------------------------------------
    def _build(self):
        header = tk.Frame(self.top, bg=C_PANEL)
        header.pack(fill=tk.X)
        tk.Label(header, text=f"Analyze Dataset  -  {self.workspace}", bg=C_PANEL, fg=C_TXT1,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=14, pady=10)
        self.sub = tk.Label(header, text="", bg=C_PANEL, fg=C_TXT3, font=("Segoe UI", 9))
        self.sub.pack(side=tk.LEFT, padx=4)
        tk.Frame(self.top, bg=C_BORDER, height=1).pack(fill=tk.X)

        bar = tk.Frame(self.top, bg=C_BASE)
        bar.pack(fill=tk.X, padx=14, pady=(10, 4))
        # the small-object limit sits on the right and is packed first so it stays visible in a narrow window
        self.limit_var = tk.StringVar(value=str(self.limit))
        tk.Label(bar, text="px", bg=C_BASE, fg=C_TXT2, font=("Segoe UI", 9)).pack(side=tk.RIGHT)
        self.limit_box = tk.Spinbox(bar, from_=LIMIT_RANGE[0], to=LIMIT_RANGE[1], width=5,
                                    textvariable=self.limit_var, command=self._limit_changed)
        self.limit_box.pack(side=tk.RIGHT, padx=(6, 3))
        tk.Label(bar, text="Small object limit", bg=C_BASE, fg=C_TXT1,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT, padx=(12, 0))
        self.limit_box.bind("<Return>", lambda e: self._limit_changed())
        self.limit_box.bind("<FocusOut>", lambda e: self._limit_changed())
        tk.Label(bar, text="Resize simulation", bg=C_BASE, fg=C_TXT1, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(bar, text="input size", bg=C_BASE, fg=C_TXT2, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 4))
        combo = ttk.Combobox(bar, textvariable=self.size_var, width=7, values=[str(s) for s in da.COMPARE_SIZES])
        combo.pack(side=tk.LEFT)
        tk.Label(bar, text="px", bg=C_BASE, fg=C_TXT2, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(3, 10))
        for text, value in (("Letterbox (keep ratio)", da.LETTERBOX), ("Stretch to square", da.STRETCH)):
            tk.Radiobutton(bar, text=text, value=value, variable=self.mode_var, bg=C_BASE, fg=C_TXT1,
                           selectcolor=C_CARD, activebackground=C_BASE, activeforeground=C_TXT1,
                           font=("Segoe UI", 9), command=self._mode_changed).pack(side=tk.LEFT, padx=4)
        self.sim_btn = tk.Button(bar, text="Simulate", command=self.simulate, bg=C_ACCENT, fg=C_ON_ACCENT,
                                 font=("Segoe UI", 9, "bold"), relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                 activebackground=C_ACCENT, activeforeground=C_ON_ACCENT)
        self.sim_btn.pack(side=tk.LEFT, padx=(12, 6), ipadx=12, ipady=3)
        self.reset_btn = tk.Button(bar, text="Original sizes", command=self.reset, bg=C_CARD2, fg=C_TXT1,
                                   font=("Segoe UI", 9), relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                   activebackground=C_CARD2, activeforeground=C_TXT1)
        self.reset_btn.pack(side=tk.LEFT, ipadx=10, ipady=3)
        combo.bind("<Return>", lambda e: self.simulate())
        combo.bind("<<ComboboxSelected>>", lambda e: self.simulate())

        self.cards = tk.Frame(self.top, bg=C_BASE)
        self.cards.pack(fill=tk.X, padx=14, pady=4)

        # warning strip: lives in the window (no dialog), hidden when nothing is small
        self.alert = tk.Frame(self.top, bg=C_AMBER)
        self.remove_btn = tk.Button(self.alert, text="Remove these objects", command=self.remove_small_objects,
                                    bg="#FFFFFF", fg="#7A4A00", font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                                    cursor="hand2", borderwidth=0, activebackground="#F3E3C6",
                                    activeforeground="#7A4A00")
        self.remove_btn.pack(side=tk.RIGHT, padx=(6, 12), pady=5, ipadx=10, ipady=2)
        self.alert_text = tk.Label(self.alert, bg=C_AMBER, fg="#FFFFFF", font=("Segoe UI", 9, "bold"),
                                   anchor="w", justify=tk.LEFT, wraplength=950)
        self.alert_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12, pady=6)

        style = ttk.Style(self.top)
        style.configure("Analysis.TNotebook", background=C_BASE, borderwidth=0)
        self.tabs = ttk.Notebook(self.top, style="Analysis.TNotebook")
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 12))
        self.pages = {}
        for key, title in (("overview", "Overview"), ("size", "Object size"), ("images", "Images"),
                           ("table", "Resize table")):
            page = tk.Frame(self.tabs, bg=C_BASE)
            self.tabs.add(page, text=f"  {title}  ")
            self.pages[key] = page
        self.table = None

    # ----------------------------------------------------------
    def _target(self):
        try:
            size = int(float(self.size_var.get()))
        except ValueError:
            raise ValueError("Enter the input size as a number of pixels, e.g. 640.")
        if not 16 <= size <= 8192:
            raise ValueError("The input size must be between 16 and 8192 px.")
        return size

    def _limit_changed(self):
        """Apply the number typed in the Small object limit box (every chart, the warning and Remove follow it)."""
        text = self.limit_var.get().strip()
        try:
            value = int(text)
            if not LIMIT_RANGE[0] <= value <= LIMIT_RANGE[1]:
                raise ValueError
        except ValueError:
            self.limit_var.set(str(self.limit))
            messagebox.showwarning("Small object limit", f"Enter a whole number of pixels between {LIMIT_RANGE[0]} "
                                   f"and {LIMIT_RANGE[1]}.", parent=self.top)
            return
        self.limit_var.set(str(value))
        if value != self.limit:
            self.limit = value
            app_settings.set("small_limit", value)
            self.refresh()

    def simulate(self):
        try:
            self.size = self._target()
        except ValueError as e:
            messagebox.showwarning("Resize simulation", str(e), parent=self.top)
            return
        self.simulated = True
        self.refresh()

    def reset(self):
        self.simulated = False
        self.refresh()

    def _mode_changed(self):
        if self.simulated:
            self.refresh()

    def current_view(self):
        if self.simulated:
            return da.view(self.data, self.size, self.mode_var.get())
        return da.view(self.data)

    # ----------------------------------------------------------
    def refresh(self):
        v = self.current_view()
        self.view = v
        self.sub.config(text=f"{self.data.n_images} images in {len(self.data.instances)} dataset(s), "
                             f"read from the VOC XML files")
        self._cards(v)
        self._alert(v)
        self._overview()
        self._size_charts(v)
        self._image_charts()
        self._table()
        self.sim_btn.config(text="Simulate" if not self.simulated else f"Simulating {self.size} px")

    def _card(self, title, value, note=""):
        card = tk.Frame(self.cards, bg=C_PANEL, highlightbackground=C_BORDER, highlightthickness=1)
        card.pack(side=tk.LEFT, padx=(0, 8), fill=tk.Y)
        tk.Label(card, text=title.upper(), bg=C_PANEL, fg=C_TXT3, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=10, pady=(7, 0))
        tk.Label(card, text=value, bg=C_PANEL, fg=C_TXT1, font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=10)
        tk.Label(card, text=note or " ", bg=C_PANEL, fg=C_TXT2, font=("Segoe UI", 8)).pack(anchor="w", padx=10, pady=(0, 7))

    def _cards(self, v):
        for w in self.cards.winfo_children():
            w.destroy()
        d = self.data
        self._card("Images", str(d.n_images), f"{d.n_labelled} labelled, {d.n_unlabelled} without objects")
        self._card("Objects", str(d.n_objects), f"{len(d.classes)} classes")
        original = da.smallest(da.view(d))
        if self.simulated:
            now = da.smallest(v)
            self._card("Smallest object", _px(now), f"was {_px(original)}  (at {self.size} px)")
            lost = d.n_objects - len(v["w"])
            if lost:
                self._card("Not simulated", str(lost), "image size unknown")
        else:
            self._card("Smallest object", _px(original), "original image sizes")
        st = da.spread(v["w"])
        if st:
            self._card("Median object", f"{st['median']:.0f} x {da.spread(v['h'])['median']:.0f} px", "width x height")

    def _alert(self, v):
        info = da.small_summary(v, self.limit)
        if info is None:
            self.alert.pack_forget()
            return
        where = f"at {self.size} px" if self.simulated else "in the original images"
        worst = ", ".join(f"{name} ({n})" for name, n in info["by_class"][:4])
        self.alert_text.config(
            text=f"⚠  {info['count']} object(s) ({info['percent']:.1f}%) have a side under {self.limit} px {where}.  "
                 f"Smallest: {_px(info['smallest'])}.  Most affected: {worst}.")
        self.alert.pack(fill=tk.X, padx=14, pady=(2, 2), before=self.tabs)

    def remove_small_objects(self):
        """Delete the flagged objects from the dataset (after a warning): the XML and YOLO files are changed."""
        size, mode = (self.size, self.mode_var.get()) if self.simulated else (None, da.LETTERBOX)
        info = da.small_summary(self.view, self.limit)
        if info is None:
            return
        files = da.small_sources(self.data, size, mode, self.limit)
        where = (f"measured at {size} px, {'letterbox' if mode == da.LETTERBOX else 'stretch'}" if self.simulated
                 else "measured in the original images")
        message = (
            f'This will MODIFY the dataset of workspace "{self.workspace}": {info["count"]} object(s) with a side '
            f"under {self.limit} px ({where}) will be permanently deleted from {len(files)} annotation file(s) "
            f"(the Pascal VOC XML files and their YOLO labels). The images themselves are not touched. "
            f"This cannot be undone.\n\n"
            f"Make sure you have backed up the current dataset before continuing (Export Dataset can save a "
            f"copy).\n\nDelete these objects now?")
        if not messagebox.askyesno("Remove small objects", message, icon="warning", default="no", parent=self.top):
            return
        from . import workspace_config
        popup, update = show_progress_popup(self.top, "Removing small objects...", C_RED)
        failure, result = None, None
        try:
            result = da.remove_small(self.data, da.yolo_dir_for(self.workspace),
                                     workspace_config.get_classes(self.workspace) or [], size, mode, self.limit,
                                     update)
            update(0, 1, "Reading the annotations again...")
            self.data = da.analyze_workspace(self.workspace, update)
            update(1, 1, "Updating the charts...")
            self.refresh()
        except Exception as exc:
            failure = exc
        finally:
            popup.destroy()
        if failure is not None:
            messagebox.showerror("Remove small objects", f"Removing the objects failed:\n{failure}", parent=self.top)
            return
        text = f"Removed {result['objects']} object(s) from {result['files']} annotation file(s)."
        if result["emptied"]:
            text += f"\n{result['emptied']} image(s) now have no objects."
        messagebox.showinfo("Remove small objects", text, parent=self.top)

    # ----------------------------------------------------------
    #  matplotlib plumbing
    # ----------------------------------------------------------
    def _figure(self, key, rows=1, cols=1, host=None):
        """A fresh figure in the page `key` (or in `host`; the previous content is dropped), styled with the app theme."""
        page = host or self.pages[key]
        if host is None:
            for w in page.winfo_children():
                w.destroy()
        fig = self._Figure(figsize=(1, 1), dpi=100, facecolor=C_BASE)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.9, hspace=0.55, wspace=0.28)
        canvas = self._Canvas(fig, master=page)
        toolbar = self._Toolbar(canvas, page, pack_toolbar=False)
        toolbar.config(background=TOOLBAR_BG)         # its icons are dark: a light strip keeps them visible in both themes
        for child in toolbar.winfo_children():
            try:
                child.config(background=TOOLBAR_BG)
            except tk.TclError:
                pass
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        axes = [fig.add_subplot(rows, cols, i + 1) for i in range(rows * cols)]
        for ax in axes:
            ax.set_facecolor(C_PANEL)
            for spine in ax.spines.values():
                spine.set_color(C_BORDER)
            ax.tick_params(colors=C_TXT2, labelsize=8)
            ax.xaxis.label.set_color(C_TXT2)
            ax.yaxis.label.set_color(C_TXT2)
            ax.grid(True, color=C_BORDER, linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
        self.figures[key] = fig
        return fig, canvas, axes

    def _title(self, ax, text):
        ax.set_title(text, color=C_TXT1, fontsize=10, fontweight="bold", loc="left")

    def _hover(self, canvas, ax, patches, labels):
        """Tooltip with labels[i] while the pointer is over patches[i] (bars, histogram columns)."""
        note = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", color=C_TXT1, fontsize=8,
                           bbox=dict(boxstyle="round", fc=C_CARD, ec=C_BORDER), zorder=10)
        note.set_visible(False)

        def move(event):
            if event.inaxes is not ax:
                if note.get_visible():
                    note.set_visible(False)
                    canvas.draw_idle()
                return
            for patch, label in zip(patches, labels):
                if patch.contains(event)[0]:
                    note.xy = (event.xdata, event.ydata)
                    note.set_text(label)
                    note.set_visible(True)
                    canvas.draw_idle()
                    return
            if note.get_visible():
                note.set_visible(False)
                canvas.draw_idle()
        canvas.mpl_connect("motion_notify_event", move)

    def _hover_points(self, canvas, ax, scatter, labels):
        note = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", color=C_TXT1, fontsize=8,
                           bbox=dict(boxstyle="round", fc=C_CARD, ec=C_BORDER), zorder=10)
        note.set_visible(False)

        def move(event):
            if event.inaxes is ax:
                hit, info = scatter.contains(event)
                if hit and len(info["ind"]):
                    i = int(info["ind"][0])
                    note.xy = scatter.get_offsets()[i]
                    note.set_text(labels[i])
                    note.set_visible(True)
                    canvas.draw_idle()
                    return
            if note.get_visible():
                note.set_visible(False)
                canvas.draw_idle()
        canvas.mpl_connect("motion_notify_event", move)

    # ----------------------------------------------------------
    #  Tabs
    # ----------------------------------------------------------
    def _overview(self):
        d = self.data
        fig, canvas, (a1, a2) = self._figure("overview", 1, 2)
        fig.subplots_adjust(left=0.13)
        names = [n for n, _ in d.class_counts()]
        counts = [c for _, c in d.class_counts()]
        self._title(a1, "Objects per class")
        if names:
            bars = a1.barh(range(len(names)), counts, color=[PALETTE[i % len(PALETTE)] for i in range(len(names))])
            a1.set_yticks(range(len(names)), names)
            a1.invert_yaxis()
            a1.set_xlabel("objects")
            total = max(sum(counts), 1)
            for b, c in zip(bars, counts):
                a1.text(c, b.get_y() + b.get_height() / 2, f" {c}", va="center", color=C_TXT2, fontsize=8)
            self._hover(canvas, a1, bars, [f"{n}: {c} ({100 * c / total:.1f}%)" for n, c in zip(names, counts)])
        else:
            a1.text(0.5, 0.5, "No annotated objects yet", ha="center", va="center", color=C_TXT3, transform=a1.transAxes)
        self._title(a2, "Images per dataset")
        inst = d.instances
        if inst:
            x = range(len(inst))
            lab = [i["labelled"] for i in inst]
            unl = [i["images"] - i["labelled"] for i in inst]
            b1 = a2.bar(x, lab, color=C_ACCENT, label="labelled")
            b2 = a2.bar(x, unl, bottom=lab, color=C_CARD2, edgecolor=C_BORDER, label="no objects")
            step = max(1, len(inst) // 12)
            a2.set_xticks(list(x)[::step], [i["name"].rsplit("-", 1)[-1] for i in inst][::step])
            a2.set_xlabel("dataset index")
            a2.set_ylabel("images")
            a2.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TXT1, fontsize=8)
            self._hover(canvas, a2, list(b1) + list(b2),
                        [f"{i['name']}: {i['labelled']} labelled" for i in inst] +
                        [f"{i['name']}: {i['images'] - i['labelled']} without objects" for i in inst])
        canvas.draw()

    def _size_charts(self, v):
        import numpy as np
        fig, canvas, (a1, a2, a3, a4) = self._figure("size", 2, 2)
        label = f"at {self.size} px" if self.simulated else "original"
        w, h = v["w"], v["h"]
        if not len(w):
            for ax in (a1, a2, a3, a4):
                ax.text(0.5, 0.5, "No objects", ha="center", va="center", color=C_TXT3, transform=ax.transAxes)
            canvas.draw()
            return
        hi = float(np.percentile(np.concatenate([w, h]), 99.5)) or 1.0
        bins = np.linspace(0, hi, 41)
        for ax, vals, name in ((a1, w, "Width"), (a2, h, "Height")):
            self._title(ax, f"{name} of objects ({label})")
            _, edges, patches = ax.hist(np.clip(vals, 0, hi), bins=bins, color=C_ACCENT, edgecolor=C_PANEL)
            ax.axvline(self.limit, color=C_AMBER, linestyle="--", linewidth=1.2, label=f"{self.limit} px")
            ax.set_xlabel("px (the last column also holds everything larger)")
            ax.set_ylabel("objects")
            ax.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TXT1, fontsize=8)
            self._hover(canvas, ax, patches, [f"{edges[i]:.0f}-{edges[i + 1]:.0f} px: {int(p.get_height())} objects"
                                              for i, p in enumerate(patches)])
        self._title(a3, f"Width x height ({label})")
        n = len(w)
        pick = np.arange(n) if n <= 20000 else np.random.default_rng(0).choice(n, 20000, replace=False)
        names = sorted(set(v["cls"].tolist()))
        colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(names)}
        sc = a3.scatter(w[pick], h[pick], s=6, alpha=0.55, c=[colors[c] for c in v["cls"][pick]], linewidths=0)
        a3.axvline(self.limit, color=C_AMBER, linestyle="--", linewidth=1)
        a3.axhline(self.limit, color=C_AMBER, linestyle="--", linewidth=1)
        a3.set_xlabel("width px")
        a3.set_ylabel("height px")
        self._hover_points(canvas, a3, sc, [f"{v['cls'][i]}: {w[i]:.0f} x {h[i]:.0f} px" for i in pick])
        self._title(a4, f"Shortest side per class ({label})")
        short = np.minimum(w, h)
        groups = [short[v["cls"] == c] for c in names]
        box = a4.boxplot(groups, patch_artist=True, showfliers=False,
                         medianprops=dict(color=C_TXT1), whiskerprops=dict(color=C_TXT3), capprops=dict(color=C_TXT3))
        for i, patch in enumerate(box["boxes"]):
            patch.set(facecolor=colors[names[i]], alpha=0.8, edgecolor=C_BORDER)
        a4.set_xticks(range(1, len(names) + 1), names)           # (boxplot's own label argument changed names in matplotlib 3.9)
        a4.axhline(self.limit, color=C_AMBER, linestyle="--", linewidth=1)
        a4.set_ylabel("px")
        a4.tick_params(axis="x", rotation=30)
        canvas.draw()

    def _image_charts(self):
        d = self.data
        fig, canvas, (a1, a2) = self._figure("images", 1, 2)
        fig.subplots_adjust(left=0.13)
        self._title(a1, "Image resolutions")
        top = d.image_sizes.most_common(10)
        if top:
            labels = [f"{w}x{h}" for (w, h), _ in top]
            counts = [c for _, c in top]
            bars = a1.barh(range(len(top)), counts, color=C_ACCENT)
            a1.set_yticks(range(len(top)), labels)
            a1.invert_yaxis()
            a1.set_xlabel("images")
            rest = sum(d.image_sizes.values()) - sum(counts)
            if rest:
                a1.set_title(f"Image resolutions (top 10, {rest} more in other sizes)", color=C_TXT1, fontsize=10,
                             fontweight="bold", loc="left")
            self._hover(canvas, a1, bars, [f"{l}: {c} image(s)" for l, c in zip(labels, counts)])
        else:
            a1.text(0.5, 0.5, "No image sizes in the XML files", ha="center", va="center", color=C_TXT3,
                    transform=a1.transAxes)
        self._title(a2, "Objects per image")
        if d.per_image:
            top_n = max(d.per_image)
            edges = list(range(1, min(top_n, 40) + 2))
            _, _, patches = a2.hist([min(c, 40) for c in d.per_image], bins=edges, color=C_ACCENT, edgecolor=C_PANEL,
                                    align="left", rwidth=0.9)
            a2.set_xlabel("objects in the image (40 = 40 or more)")
            a2.set_ylabel("images")
            self._hover(canvas, a2, patches, [f"{i + 1} object(s): {int(p.get_height())} image(s)"
                                              for i, p in enumerate(patches)])
        canvas.draw()

    def _table(self):
        page = self.pages["table"]
        for w in page.winfo_children():
            w.destroy()
        rows = da.compare(self.data, mode=self.mode_var.get(), limit=self.limit)
        mode_text = "letterbox" if self.mode_var.get() == da.LETTERBOX else "stretch"
        tk.Label(page, text=f"Where does the smallest object end up for each model input size? ({mode_text}, calculated)",
                 bg=C_BASE, fg=C_TXT1, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 6))
        cols = ("size", "smallest", "median", "small", "percent")
        tree = ttk.Treeview(page, columns=cols, show="headings", height=len(rows))
        for col, text, width in (("size", "Input size", 100), ("smallest", "Smallest object", 170),
                                 ("median", "Median shortest side", 170), ("small", f"Objects < {self.limit} px", 150),
                                 ("percent", "Share", 90)):
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor="center")
        for r in rows:
            tree.insert("", tk.END, values=(f"{r['size']} px", _px(r["smallest"]),
                                            "-" if r["median_side"] is None else f"{r['median_side']:.1f} px",
                                            r["small"], f"{r['small_percent']:.1f}%"))
        tree.pack(fill=tk.X)
        self.table = tree
        chart = tk.Frame(page, bg=C_BASE)
        chart.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        fig, canvas, (ax,) = self._figure("table_chart", 1, 1, host=chart)
        fig.subplots_adjust(bottom=0.28, top=0.88)
        self._title(ax, "Smallest object's shortest side vs input size")
        sides = [min(r["smallest"]) if r["smallest"] else 0 for r in rows]
        xs = [r["size"] for r in rows]
        line = ax.plot(xs, sides, marker="o", color=C_ACCENT)[0]
        ax.axhline(self.limit, color=C_AMBER, linestyle="--", linewidth=1.2, label=f"{self.limit} px")
        ax.set_xlabel("model input size (px)")
        ax.set_ylabel("px")
        ax.set_xticks(xs)
        ax.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TXT1, fontsize=8)
        for x, y in zip(xs, sides):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
                        color=C_TXT2, fontsize=8)
        canvas.draw()


def open_dataset_analysis(parent, workspace, data=None):
    return DatasetAnalysisWindow(parent, workspace, data)
