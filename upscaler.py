# -*- coding: utf-8 -*-
"""AI Image Upscaler (PC) - professional dark UI, single + batch mode, GPU/CPU auto."""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from upscale_engine import load_session, upscale, Cancelled

# ---------------- theme ----------------
BG_MAIN = '#0d1220'
BG_CARD = '#161e30'
BG_INPUT = '#0b1120'
CARD_BORDER = '#2a3a58'
ACCENT = '#22d3ee'
ACCENT_DEEP = '#0e7490'
TEXT = '#ffffff'
MUTED = '#a9b7cc'
SUCCESS = '#34d399'
FONT = 'Segoe UI'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    'fast': {'label': 'দ্রুত (সাধারণ ব্যবহার)',
             'file': os.path.join(BASE_DIR, 'models', 'realesr-general-x4v3.onnx'),
             'tile': 512},
    'best': {'label': 'সেরা কোয়ালিটি (সর্বোচ্চ ডিটেইল)',
             'file': os.path.join(BASE_DIR, 'models', 'RealESRGAN_x4plus.onnx'),
             'tile': 256},
}
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')


# ---------------- gradient helpers ----------------
def _hex(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lighten(h, amt=0.25):
    r, g, b = _hex(h)
    r = int(r + (255 - r) * amt)
    g = int(g + (255 - g) * amt)
    b = int(b + (255 - b) * amt)
    return f'#{r:02x}{g:02x}{b:02x}'


def vgrad(w, h, top, bottom):
    t = np.array(_hex(top), dtype=np.float32)
    b = np.array(_hex(bottom), dtype=np.float32)
    rows = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    arr = (t * (1 - rows) + b * rows).astype(np.uint8)
    arr = np.repeat(arr, w, axis=1)
    return Image.fromarray(arr, 'RGB')


def rounded_grad_img(w, h, top, bottom, radius=12):
    base = vgrad(w, h, top, bottom).convert('RGBA')
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    base.putalpha(mask)
    return ImageTk.PhotoImage(base)


class GradButton(tk.Button):
    """Rounded gradient button with hover + disabled states (text rendered natively)."""

    def __init__(self, parent, text, command, w=300, h=46,
                 top='#3ae2fc', bottom='#0e7490', fg='#ffffff',
                 font=(FONT, 12, 'bold'), radius=12):
        self._img_n = rounded_grad_img(w, h, top, bottom, radius)
        self._img_h = rounded_grad_img(w, h, _lighten(top), _lighten(bottom), radius)
        self._img_d = rounded_grad_img(w, h, '#3a475c', '#232c40', radius)
        super().__init__(parent, image=self._img_n, text=text, compound='center',
                         command=command, relief='flat', bd=0, highlightthickness=0,
                         cursor='hand2', fg=fg, activeforeground=fg, font=font,
                         disabledforeground='#7d8aa0', bg=BG_CARD, activebackground=BG_CARD)
        self.bind('<Enter>', lambda e: self._hover(True))
        self.bind('<Leave>', lambda e: self._hover(False))

    def _hover(self, on):
        if str(self['state']) != 'disabled':
            self.config(image=self._img_h if on else self._img_n)

    def set_enabled(self, on):
        self.config(state='normal' if on else 'disabled',
                    image=self._img_n if on else self._img_d)


# ---------------- misc ----------------
def save_image(img, path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        img.convert('RGB').save(path, quality=95)
    elif ext == '.webp':
        img.save(path, quality=95, method=6)
    else:
        img.save(path)


def run_batch(files, out_dir, sess, in_name, tile, scale, cb, cancel_event):
    """Headless batch loop. cb(kind, *args). kinds: file_done, error, progress, tile, finished, cancelled."""
    total = len(files)
    ok = 0
    for i, src in enumerate(files):
        if cancel_event.is_set():
            cb('cancelled', ok, total)
            return
        name = os.path.basename(src)
        cb('progress', i, total, name)
        try:
            img = Image.open(src)
            img.load()
            result = upscale(img, sess, in_name, scale=scale,
                             tile_size=tile, overlap=16,
                             progress_cb=lambda d, t: cb('tile', i, total, name, d, t),
                             cancel_event=cancel_event)
            base, ext = os.path.splitext(name)
            if ext.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
                ext = '.png'
            dst = os.path.join(out_dir, f'{base}_AI_{scale}x{ext}')
            save_image(result, dst)
            ok += 1
            cb('file_done', i, total, name, dst)
        except Exception as e:  # noqa: BLE001 - per-file errors are logged, batch continues
            cb('error', i, total, name, str(e))
    cb('finished', ok, total)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('AI Image Upscaler')
        self.geometry('660x800')
        self.minsize(660, 760)
        self.configure(bg=BG_MAIN)

        self.src_path = None
        self.src_img = None
        self.result_img = None
        self.sessions = {}
        self.job_q = queue.Queue()
        self.working = False
        self.cancel_event = threading.Event()
        self.s_cancel = threading.Event()
        self.batch_in = ''
        self.batch_out = ''

        self._style()
        self._banner()
        self._tabs()
        self._footer()
        self.after(150, self.poll_queue)

    # ---------- styling ----------
    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use('clam')
        except tk.TclError:
            pass
        s.configure('TNotebook', background=BG_MAIN, borderwidth=0)
        s.configure('TNotebook.Tab', background=BG_CARD, foreground='#c3cede',
                    padding=(20, 10), font=(FONT, 11, 'bold'), borderwidth=0)
        s.map('TNotebook.Tab',
              background=[('selected', ACCENT_DEEP)],
              foreground=[('selected', '#ffffff')])
        s.configure('TFrame', background=BG_MAIN)
        s.configure('TCombobox', fieldbackground=BG_INPUT, background=BG_CARD,
                    foreground='#ffffff', arrowcolor=ACCENT, font=(FONT, 11))
        s.configure('Horizontal.TProgressbar', troughcolor=BG_INPUT,
                    background=ACCENT, borderwidth=0, thickness=16)
        s.configure('Vertical.TScrollbar', background='#2a3a58', troughcolor=BG_MAIN,
                    borderwidth=0, arrowcolor=MUTED, relief='flat')
        # readable dropdown list on all systems
        self.option_add('*TCombobox*Listbox.background', BG_INPUT)
        self.option_add('*TCombobox*Listbox.foreground', '#ffffff')
        self.option_add('*TCombobox*Listbox.selectBackground', ACCENT_DEEP)
        self.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        self.option_add('*TCombobox*Listbox.font', (FONT, 11))

    def _banner(self):
        cv = tk.Canvas(self, width=2000, height=96, highlightthickness=0, bd=0, bg=BG_MAIN)
        cv.pack(fill='x')
        self._banner_img = ImageTk.PhotoImage(vgrad(2000, 96, '#123043', '#0d1220'))
        cv.create_image(0, 0, image=self._banner_img, anchor='nw')
        cv.create_text(330, 36, text='AI Image Upscaler', fill='#ffffff', font=(FONT, 24, 'bold'))
        cv.create_text(330, 68, text='আসল AI ডিটেইলে ছবি বড় করুন — ১০০% অফলাইন ও প্রাইভেট',
                       fill=MUTED, font=(FONT, 11))
        cv.create_rectangle(0, 94, 2000, 96, fill=ACCENT, outline='')

    def _card(self, parent):
        f = tk.Frame(parent, bg=BG_CARD, highlightbackground=CARD_BORDER,
                     highlightthickness=1, padx=16, pady=12)
        f.pack(fill='x', padx=16, pady=7)
        return f

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=BG_CARD, fg=ACCENT,
                 font=(FONT, 10, 'bold')).pack(anchor='w', pady=(0, 8))

    def _scrollable(self, parent):
        """Wraps tab content in a canvas with a right-side scrollbar."""
        canvas = tk.Canvas(parent, bg=BG_MAIN, highlightthickness=0, bd=0)
        vbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=BG_MAIN)
        iid = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _sync(_e=None):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(iid, width=canvas.winfo_width())

        inner.bind('<Configure>', lambda e: _sync())
        canvas.bind('<Configure>', lambda e: _sync())
        canvas.bind('<MouseWheel>',
                    lambda e: canvas.yview_scroll(-int(e.delta / 120), 'units'))
        return inner

    def _tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=10, pady=(6, 2))
        self.tab_single = ttk.Frame(nb)
        self.tab_batch = ttk.Frame(nb)
        nb.add(self.tab_single, text='  একক ছবি  ')
        nb.add(self.tab_batch, text='  বাল্ক (ফোল্ডার)  ')
        self._build_single(self._scrollable(self.tab_single))
        self._build_batch(self._scrollable(self.tab_batch))

    def _footer(self):
        tk.Label(self, text='v2.1   •   100% অফলাইন   •   আপনার ছবি কোথাও যায় না',
                 bg=BG_MAIN, fg='#5b6b85', font=(FONT, 9)).pack(pady=6)

    # ---------- single image tab ----------
    def _build_single(self, tab):
        c = self._card(tab)
        self._section(c, 'ছবি')
        row = tk.Frame(c, bg=BG_CARD)
        row.pack(fill='x')
        GradButton(row, '📁  ছবি বাছুন', self.pick_image, w=190, h=42,
                   top='#4a5a75', bottom='#2b3648', font=(FONT, 11, 'bold')).pack(side='left')
        self.s_info = tk.Label(row, text='কোনো ছবি বাছা হয়নি', bg=BG_CARD, fg=MUTED,
                               font=(FONT, 10))
        self.s_info.pack(side='left', padx=14)
        self.s_prev = tk.Label(c, bg=BG_CARD)
        self.s_prev.pack(pady=(8, 0))

        c2 = self._card(tab)
        self._section(c2, 'সেটিংস')
        tk.Label(c2, text='মডেল', bg=BG_CARD, fg=MUTED, font=(FONT, 10)).pack(anchor='w')
        self.s_model = ttk.Combobox(c2, width=34, state='readonly', font=(FONT, 11))
        self.s_model['values'] = [m['label'] for m in MODELS.values()]
        self.s_model.current(1)
        self.s_model.pack(pady=(4, 2), anchor='w')
        self.s_engine = tk.Label(c2, text='ইঞ্জিন: —', bg=BG_CARD, fg=MUTED, font=(FONT, 9))
        self.s_engine.pack(anchor='w', pady=(0, 8))
        tk.Label(c2, text='স্কেল', bg=BG_CARD, fg=MUTED, font=(FONT, 10)).pack(anchor='w', pady=(0, 2))
        self.s_scale = tk.IntVar(value=4)
        rf = tk.Frame(c2, bg=BG_CARD)
        rf.pack(anchor='w')
        for v in (2, 3, 4):
            tk.Radiobutton(rf, text=f'{v}x', variable=self.s_scale, value=v,
                           bg=BG_CARD, fg=TEXT, selectcolor=ACCENT_DEEP,
                           activebackground=BG_CARD, activeforeground=TEXT,
                           font=(FONT, 12, 'bold'), padx=12).pack(side='left')

        cta = tk.Frame(tab, bg=BG_MAIN)
        cta.pack(pady=(6, 4))
        self.s_go = GradButton(cta, '✨  আপস্কেল করুন', self.start_single, w=400, h=54,
                               font=(FONT, 14, 'bold'))
        self.s_go.pack(side='left')
        self.s_stop = GradButton(cta, '⏹  বন্ধ করুন', self.stop_single, w=160, h=54,
                                 top='#4a5a75', bottom='#2b3648', font=(FONT, 12, 'bold'))
        self.s_stop.pack(side='left', padx=(12, 0))
        self.s_stop.set_enabled(False)

        c3 = self._card(tab)
        self.s_prog = ttk.Progressbar(c3, style='Horizontal.TProgressbar', mode='determinate')
        self.s_prog.pack(fill='x', pady=(2, 6))
        self.s_status = tk.Label(c3, text='', bg=BG_CARD, fg=SUCCESS, font=(FONT, 10))
        self.s_status.pack()
        self.s_save = GradButton(c3, '💾  সেভ করুন', self.save_single, w=220, h=42,
                                 top='#4a5a75', bottom='#2b3648', font=(FONT, 11, 'bold'))
        self.s_save.pack(pady=(8, 0))
        self.s_save.set_enabled(False)

    def pick_image(self):
        p = filedialog.askopenfilename(title='ছবি বাছুন',
            filetypes=[('Image', '*.png *.jpg *.jpeg *.webp *.bmp *.tiff'), ('All', '*.*')])
        if not p:
            return
        try:
            img = Image.open(p)
            img.load()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror('এরর', f'ছবি খোলা যায়নি:\n{e}')
            return
        self.src_path, self.src_img, self.result_img = p, img, None
        self.s_save.set_enabled(False)
        w, h = img.size
        self.s_info.config(text=f'{os.path.basename(p)}   •   {w}×{h}', fg=TEXT)
        th = img.copy()
        th.thumbnail((200, 200))
        self._thumb = ImageTk.PhotoImage(th)
        self.s_prev.config(image=self._thumb)
        self.s_status.config(text='')

    def _model_key(self, combo):
        return list(MODELS.keys())[combo.current()]

    def _get_session(self, key, force_cpu=False):
        ckey = (key, 'cpu') if force_cpu else (key, 'gpu')
        if ckey not in self.sessions:
            sess, in_name, engine = load_session(MODELS[key]['file'],
                                                prefer_gpu=not force_cpu)
            self.sessions[ckey] = (sess, in_name, engine)
            self.job_q.put(('engine', key, engine))
        return self.sessions[ckey]

    def start_single(self):
        if self.working or self.src_img is None:
            if self.src_img is None:
                messagebox.showwarning('সতর্কতা', 'আগে একটি ছবি বাছুন।')
            return
        key = self._model_key(self.s_model)
        if not os.path.exists(MODELS[key]['file']):
            messagebox.showerror('এরর', 'মডেল ফাইল পাওয়া যায়নি।\nmodels ফোল্ডারটি ঠিক আছে কি না দেখুন।')
            return
        scale = self.s_scale.get()
        w, h = self.src_img.size
        if w * scale > 12000 or h * scale > 12000:
            if not messagebox.askyesno('বড় আউটপুট',
                    f'আউটপুট হবে {w*scale}×{h*scale} — এতে অনেক সময় লাগতে পারে।\nচালিয়ে যাবেন?'):
                return
        self.working = True
        self.s_cancel.clear()
        self.s_go.set_enabled(False)
        self.s_stop.set_enabled(True)
        self.s_save.set_enabled(False)
        self.s_prog['value'] = 0
        threading.Thread(target=self._single_worker, args=(key, scale), daemon=True).start()

    def stop_single(self):
        self.s_cancel.set()
        self.s_status.config(text='বন্ধ করা হচ্ছে...')

    def _single_worker(self, key, scale):
        q = self.job_q
        try:
            q.put(('s_status', 'মডেল লোড হচ্ছে... (প্রথমবার একটু সময় নেবে)'))
            img = self.src_img.copy()
            tile = MODELS[key]['tile']

            def run(force_cpu):
                sess, in_name, engine = self._get_session(key, force_cpu=force_cpu)
                q.put(('s_status', 'AI আপস্কেল চলছে...'))
                result = upscale(img, sess, in_name, scale=scale,
                                 tile_size=tile, overlap=16,
                                 progress_cb=lambda d, t: q.put(('s_prog', d / t * 100, f'টাইল {d}/{t}')),
                                 cancel_event=self.s_cancel)
                return result, engine

            engine = 'CPU'
            try:
                result, engine = run(False)
            except Cancelled:
                raise
            except Exception:
                # GPU failed mid-run (bad driver) -> rebuild on CPU, retry once
                if engine == 'GPU':
                    q.put(('s_status', 'GPU-তে সমস্যা হয়েছে — CPU-তে আবার চেষ্টা করা হচ্ছে...'))
                    result, engine = run(True)
                else:
                    raise
            q.put(('s_done', result, scale))
        except Cancelled:
            q.put(('s_cancelled',))
        except Exception as e:  # noqa: BLE001
            q.put(('s_error', str(e)))

    def save_single(self):
        if self.result_img is None:
            return
        base = os.path.splitext(os.path.basename(self.src_path))[0]
        scale = self.s_scale.get()
        p = filedialog.asksaveasfilename(title='সেভ করুন',
            initialfile=f'{base}_AI_{scale}x.png', defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('WebP', '*.webp')])
        if not p:
            return
        try:
            save_image(self.result_img, p)
            self.s_status.config(text=f'💾 সেভ হয়েছে: {p}')
            messagebox.showinfo('সম্পন্ন', 'ছবি সেভ হয়েছে!')
        except Exception as e:  # noqa: BLE001
            messagebox.showerror('এরর', f'সেভ করা যায়নি:\n{e}')

    # ---------- batch tab ----------
    def _build_batch(self, tab):
        c = self._card(tab)
        self._section(c, 'ফোল্ডার')
        row = tk.Frame(c, bg=BG_CARD)
        row.pack(fill='x')
        GradButton(row, '📂  ইনপুট ফোল্ডার', self.pick_in, w=200, h=42,
                   top='#4a5a75', bottom='#2b3648', font=(FONT, 11, 'bold')).pack(side='left')
        self.b_in_lbl = tk.Label(row, text='বাছা হয়নি', bg=BG_CARD, fg=MUTED,
                                 font=(FONT, 9), wraplength=330, justify='left')
        self.b_in_lbl.pack(side='left', padx=12)
        row2 = tk.Frame(c, bg=BG_CARD)
        row2.pack(fill='x', pady=(8, 0))
        GradButton(row2, '📂  আউটপুট ফোল্ডার', self.pick_out, w=200, h=42,
                   top='#4a5a75', bottom='#2b3648', font=(FONT, 11, 'bold')).pack(side='left')
        self.b_out_lbl = tk.Label(row2, text='বাছা হয়নি (ডিফল্ট: ইনপুট/ upscaled)',
                                  bg=BG_CARD, fg=MUTED, font=(FONT, 9),
                                  wraplength=330, justify='left')
        self.b_out_lbl.pack(side='left', padx=12)

        c2 = self._card(tab)
        self._section(c2, 'সেটিংস')
        self.b_model = ttk.Combobox(c2, width=34, state='readonly', font=(FONT, 11))
        self.b_model['values'] = [m['label'] for m in MODELS.values()]
        self.b_model.current(0)
        self.b_model.pack(anchor='w', pady=(0, 2))
        self.b_engine = tk.Label(c2, text='ইঞ্জিন: —', bg=BG_CARD, fg=MUTED, font=(FONT, 9))
        self.b_engine.pack(anchor='w', pady=(0, 8))
        self.b_scale = tk.IntVar(value=4)
        rf = tk.Frame(c2, bg=BG_CARD)
        rf.pack(anchor='w')
        for v in (2, 3, 4):
            tk.Radiobutton(rf, text=f'{v}x', variable=self.b_scale, value=v,
                           bg=BG_CARD, fg=TEXT, selectcolor=ACCENT_DEEP,
                           activebackground=BG_CARD, activeforeground=TEXT,
                           font=(FONT, 12, 'bold'), padx=12).pack(side='left')

        btnrow = tk.Frame(tab, bg=BG_MAIN)
        btnrow.pack(fill='x', padx=16, pady=(4, 2))
        self.b_go = GradButton(btnrow, '▶  সব আপস্কেল করুন', self.start_batch, w=270, h=50,
                               font=(FONT, 12, 'bold'))
        self.b_go.pack(side='left')
        self.b_stop = GradButton(btnrow, '⏹  বন্ধ করুন', self.stop_batch, w=200, h=50,
                                 top='#4a5a75', bottom='#2b3648', font=(FONT, 12, 'bold'))
        self.b_stop.pack(side='left', padx=12)
        self.b_stop.set_enabled(False)

        c3 = self._card(tab)
        self.b_prog = ttk.Progressbar(c3, style='Horizontal.TProgressbar', mode='determinate')
        self.b_prog.pack(fill='x', pady=(2, 6))
        self.b_status = tk.Label(c3, text='', bg=BG_CARD, fg=SUCCESS, font=(FONT, 10))
        self.b_status.pack()
        self.b_log = tk.Listbox(c3, height=5, bg=BG_INPUT, fg=TEXT, font=(FONT, 9),
                                relief='flat', highlightthickness=0,
                                selectbackground=ACCENT_DEEP)
        self.b_log.pack(fill='x', pady=(8, 0))

    def pick_in(self):
        d = filedialog.askdirectory(title='ছবির ফোল্ডার বাছুন')
        if d:
            self.batch_in = d
            n = len([f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)])
            self.b_in_lbl.config(text=f'{d}\n{n}টি ছবি পাওয়া গেছে', fg=TEXT)

    def pick_out(self):
        d = filedialog.askdirectory(title='সেভ করার ফোল্ডার বাছুন')
        if d:
            self.batch_out = d
            self.b_out_lbl.config(text=d, fg=TEXT)

    def start_batch(self):
        if self.working:
            return
        if not self.batch_in or not os.path.isdir(self.batch_in):
            messagebox.showwarning('সতর্কতা', 'ইনপুট ফোল্ডার বাছুন।')
            return
        if not self.batch_out:
            self.batch_out = os.path.join(self.batch_in, 'upscaled')
        os.makedirs(self.batch_out, exist_ok=True)
        self.b_out_lbl.config(text=self.batch_out, fg=TEXT)
        files = sorted(os.path.join(self.batch_in, f) for f in os.listdir(self.batch_in)
                       if f.lower().endswith(IMG_EXTS)
                       and os.path.isfile(os.path.join(self.batch_in, f)))
        if not files:
            messagebox.showwarning('সতর্কতা', 'এই ফোল্ডারে কোনো ছবি পাওয়া যায়নি।')
            return
        key = self._model_key(self.b_model)
        if not os.path.exists(MODELS[key]['file']):
            messagebox.showerror('এরর', 'মডেল ফাইল পাওয়া যায়নি।')
            return
        self.working = True
        self.cancel_event.clear()
        self.b_go.set_enabled(False)
        self.b_stop.set_enabled(True)
        self.b_prog['value'] = 0
        self.b_log.delete(0, 'end')
        threading.Thread(target=self._batch_worker,
                         args=(files, key, self.b_scale.get()), daemon=True).start()

    def stop_batch(self):
        self.cancel_event.set()
        self.b_status.config(text='বন্ধ করা হচ্ছে...')

    def _batch_worker(self, files, key, scale):
        q = self.job_q
        try:
            q.put(('b_status', 'মডেল লোড হচ্ছে...'))
            tile = MODELS[key]['tile']

            def cb(kind, *a):
                q.put(('b_' + kind,) + a)

            def run(force_cpu):
                sess, in_name, engine = self._get_session(key, force_cpu=force_cpu)
                run_batch(files, self.batch_out, sess, in_name, tile, scale, cb,
                          self.cancel_event)
                return engine

            engine = 'CPU'
            try:
                engine = run(False)
            except Cancelled:
                raise
            except Exception:
                # GPU failed mid-run (bad driver) -> rebuild on CPU, retry once
                if engine == 'GPU':
                    q.put(('b_status', 'GPU-তে সমস্যা হয়েছে — CPU-তে আবার চেষ্টা করা হচ্ছে...'))
                    run(True)
                else:
                    raise
            q.put(('b_done',))
        except Cancelled:
            q.put(('b_cancelled',))
        except Exception as e:  # noqa: BLE001
            q.put(('b_error', str(e)))

    # ---------- queue pump ----------
    def poll_queue(self):
        try:
            while True:
                msg = self.job_q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.after(150, self.poll_queue)

    def _engine_text(self, engine):
        return '🚀 GPU ইঞ্জিন সক্রিয় (দ্রুত)' if engine == 'GPU' else '🖥️ CPU ইঞ্জিন'

    def _handle(self, msg):
        k = msg[0]
        if k == 'engine':
            _, key, engine = msg
            txt = f'ইঞ্জিন: {self._engine_text(engine)}'
            if key == self._model_key(self.s_model):
                self.s_engine.config(text=txt, fg=SUCCESS)
            if key == self._model_key(self.b_model):
                self.b_engine.config(text=txt, fg=SUCCESS)
        elif k == 's_status':
            self.s_status.config(text=msg[1])
        elif k == 's_prog':
            self.s_prog['value'] = msg[1]
            self.s_status.config(text=f'AI আপস্কেল চলছে... {msg[2]}')
        elif k == 's_done':
            _, result, scale = msg
            self.result_img = result
            self.working = False
            self.s_go.set_enabled(True)
            self.s_stop.set_enabled(False)
            self.s_save.set_enabled(True)
            self.s_prog['value'] = 100
            w, h = result.size
            self.s_status.config(text=f'✅ সম্পন্ন! আউটপুট: {w}×{h} ({scale}x AI)')
        elif k == 's_cancelled':
            self.working = False
            self.s_go.set_enabled(True)
            self.s_stop.set_enabled(False)
            self.s_status.config(text='⏹ বন্ধ করা হয়েছে')
        elif k == 's_error':
            self.working = False
            self.s_go.set_enabled(True)
            self.s_stop.set_enabled(False)
            messagebox.showerror('এরর', f'আপস্কেল ব্যর্থ:\n{msg[1]}')
        elif k == 'b_status':
            self.b_status.config(text=msg[1])
        elif k == 'b_progress':
            _, i, total, name = msg
            self.b_prog['value'] = i / total * 100
            self.b_status.config(text=f'{i+1}/{total} — {name}')
        elif k == 'b_tile':
            _, i, total, name, d, t = msg
            self.b_prog['value'] = (i + d / t) / total * 100
            self.b_status.config(text=f'{i+1}/{total} — {name} (টাইল {d}/{t})')
        elif k == 'b_file_done':
            _, i, total, name, dst = msg
            self.b_log.insert('end', f'✅ {name}')
            self.b_log.see('end')
        elif k == 'b_error':
            if len(msg) == 2:
                self.working = False
                self.b_go.set_enabled(True)
                self.b_stop.set_enabled(False)
                messagebox.showerror('এরর', f'বাল্ক ব্যর্থ:\n{msg[1]}')
            else:
                _, i, total, name, err = msg
                self.b_log.insert('end', f'❌ {name}: {err[:60]}')
                self.b_log.see('end')
        elif k == 'b_cancelled':
            _, ok, total = msg
            self.working = False
            self.b_go.set_enabled(True)
            self.b_stop.set_enabled(False)
            self.b_status.config(text=f'⏹ বন্ধ করা হয়েছে — {ok}/{total}টি সম্পন্ন')
        elif k == 'b_finished':
            _, ok, total = msg
            self.working = False
            self.b_go.set_enabled(True)
            self.b_stop.set_enabled(False)
            self.b_prog['value'] = 100
            self.b_status.config(text=f'✅ সম্পন্ন! {ok}/{total}টি ছবি → {self.batch_out}')
            messagebox.showinfo('সম্পন্ন', f'{ok}/{total}টি ছবি আপস্কেল হয়েছে!\nফোল্ডার: {self.batch_out}')


if __name__ == '__main__':
    App().mainloop()
