"""
Terminal output redirector for Tkinter text widget
"""
import queue
import tkinter as tk


class TkTerminalRedirector:
    """Redirect stdout/stderr to Tkinter text widget via queue"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()

    def write(self, msg):
        """Queue message to be written"""
        self.queue.put(msg)

    def flush(self):
        """No-op flush"""
        pass

    def update(self):
        """Process queued messages and update text widget"""
        while not self.queue.empty():
            msg = self.queue.get()
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)
