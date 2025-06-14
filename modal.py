import tkinter as tk
from tkinter import messagebox

class IPModal:
    def __init__(self, parent):
        self.parent = parent
        self.ips = []
        self.entries = []

        self.modal = tk.Toplevel(parent)
        self.modal.title("Nhập địa chỉ IP Camera")
        self.modal.geometry("420x300")
        self.modal.resizable(False, False)
        self.modal.transient(parent)
        self.modal.grab_set()

        self.container = tk.Frame(self.modal)
        self.container.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        self.entry_frame = tk.Frame(self.container)
        self.entry_frame.pack(fill=tk.BOTH, expand=True)

        # Thêm 2 ô mặc định
        for _ in range(2):
            self.add_entry()

        # Nút điều khiển
        control_frame = tk.Frame(self.container)
        control_frame.pack(pady=10)

        self.add_btn = tk.Button(control_frame, text="+ Thêm IP", command=self.add_entry)
        self.add_btn.grid(row=0, column=0, padx=5)

        self.submit_btn = tk.Button(control_frame, text="Xác nhận", command=self.submit)
        self.submit_btn.grid(row=0, column=1, padx=5)

        self.modal.protocol("WM_DELETE_WINDOW", self.on_closing)

    def add_entry(self):
        if len(self.entries) >= 4:
            messagebox.showinfo("Thông báo", "Tối đa 4 địa chỉ IP.")
            return

        row = len(self.entries)

        row_frame = tk.Frame(self.entry_frame)
        row_frame.grid(row=row, column=0, pady=5, sticky="w")

        label = tk.Label(row_frame, text=f"IP Camera {row + 1}:", width=12, anchor="w")
        label.pack(side=tk.LEFT)

        entry = tk.Entry(row_frame, width=25)
        entry.pack(side=tk.LEFT, padx=5)

        delete_btn = tk.Button(row_frame, text="X", command=lambda: self.remove_entry(row_frame))
        delete_btn.pack(side=tk.LEFT)

        self.entries.append((row_frame, entry))

    def remove_entry(self, frame):
        if len(self.entries) <= 2:
            messagebox.showwarning("Cảnh báo", "Phải nhập ít nhất 2 địa chỉ IP.")
            return

        for i, (f, e) in enumerate(self.entries):
            if f == frame:
                f.destroy()
                self.entries.pop(i)
                break

        self.refresh_labels()

    def refresh_labels(self):
        for idx, (frame, _) in enumerate(self.entries):
            label = frame.winfo_children()[0]
            label.config(text=f"IP Camera {idx + 1}:")

    def submit(self):
        self.ips = [entry.get().strip() for _, entry in self.entries if entry.get().strip()]
        if len(self.ips) < 2:
            messagebox.showerror("Lỗi", "Vui lòng nhập ít nhất 2 địa chỉ IP hợp lệ.")
        else:
            self.modal.destroy()

    def on_closing(self):
        self.ips = []
        self.modal.destroy()

    def get_ips(self):
        self.modal.wait_window()
        return self.ips
