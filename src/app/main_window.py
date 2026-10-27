from tkinter import filedialog
import customtkinter as ctk
from PIL import Image
import numpy as np
from src.app import image_io
from src.app.filters import FILTERS, FILTER_NAMES, MASK_SIZES, FilterParams

# Dominik Sobieraj
# Filtrowanie splotowe

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Filtrowanie splotowe")
        self.geometry("1600x1200")
        self.minsize(1600,1200)

        self.original_image = None
        self.filtered_image = None
        self._ctk_images: dict[str, ctk.CTkImage] = {} # do trzymania referencji obrazów

        self.mask_size_var = ctk.StringVar(value = str(MASK_SIZES[0]))
        self.filter_var = ctk.StringVar(value = FILTER_NAMES[0])
        self.sigma_var = ctk.DoubleVar(value = 1)
        self.fast_convolution_var = ctk.BooleanVar(value = True)
        self.alpha_var = ctk.DoubleVar(value=1.0)

        self._build_top_panel()
        self._build_image_panel()
        self.on_filter_change(self.filter_var.get())


    def _build_top_panel(self):
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=10, pady=10)
        top.grid_columnconfigure(10, weight=1)

        ctk.CTkButton(top, text="Wczytaj obraz", command=self.load_image).grid(
            row=0, column=0, padx=10, pady=10)

        ctk.CTkLabel(top, text="Typ filtra:").grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkOptionMenu(
            top, variable=self.filter_var, values=FILTER_NAMES,
            command=self.on_filter_change,
        ).grid(row=0, column=2, padx=10, pady=10)

        self.mask_label = ctk.CTkLabel(top, text="Wielkość maski:")
        self.mask_label.grid(row=0, column=3, padx=10, pady=10)
        self.mask_menu = ctk.CTkOptionMenu(
            top, variable=self.mask_size_var, values=[str(x) for x in MASK_SIZES])
        self.mask_menu.grid(row=0, column=4, padx=5, pady=10)

        self.sigma_label = ctk.CTkLabel(top, text="Sigma: 1.00")
        self.sigma_label.grid(row=0, column=5, padx=10, pady=10)
        self.sigma_slider = ctk.CTkSlider(
            top, from_=0.1, to=5.0, number_of_steps=49,  # type: ignore
            variable=self.sigma_var, command=self.update_sigma_label, width=120)
        self.sigma_slider.grid(row=0, column=6, padx=5, pady=10)

        self.fast_checkbox = ctk.CTkCheckBox(
            top, variable=self.fast_convolution_var, text="Szybki splot")
        self.fast_checkbox.grid(row=0, column=7, padx=5, pady=10)

        self.alpha_label = ctk.CTkLabel(top, text="Alfa: 1.00")
        self.alpha_label.grid(row=0, column=8, padx=10, pady=10)
        self.alpha_slider = ctk.CTkSlider(
            top, from_=0.5, to=5.0, number_of_steps=45,  # type: ignore
            variable=self.alpha_var, command=self.update_alpha_label, width=120)
        self.alpha_slider.grid(row=0, column=9, padx=5, pady=10)

        ctk.CTkButton(top, text="Wyczyść obraz", command=self.clear).grid(
            row=0, column=11, padx=10, pady=10)
        ctk.CTkButton(
            top, text="Filtruj", command=self.apply_filter,
            fg_color="green", hover_color="darkgreen",
        ).grid(row=0, column=12, padx=10, pady=10, sticky="e")

        self.mask_widgets = (self.mask_label, self.mask_menu)
        self.sigma_widgets = (self.sigma_label, self.sigma_slider)
        self.fast_widgets = (self.fast_checkbox,)
        self.alpha_widgets = (self.alpha_label, self.alpha_slider)


    def _build_image_panel(self):
        self.panel_frame = ctk.CTkFrame(self)
        self.panel_frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)
        self.panel_frame.rowconfigure(1, weight=1)
        self._render_panels([("Oryginał", None), ("Po filtrowaniu", None)])


    def _render_panels(self, panels):
        for w in self.panel_frame.winfo_children():
            w.destroy()
        self._ctk_images.clear()

        n = len(panels)
        for c in range(6):  # reset wag po poprzednim renderze
            self.panel_frame.columnconfigure(c, weight=1 if c < n else 0)

        max_w = (self.winfo_width() - 60) // n  # obraz skalowany do szerokości kolumny
        for col, (title, image) in enumerate(panels):
            ctk.CTkLabel(self.panel_frame, text=title, font=("Arial", 16, "bold")).grid(
                row=0, column=col, pady=(10, 0))
            label = ctk.CTkLabel(self.panel_frame, text="Brak obrazu")
            label.grid(row=1, column=col, padx=10, pady=10, sticky="nsew")
            if image is not None:
                scale = min(1.0, max_w / image.width)
                size = (int(image.width * scale), int(image.height * scale))
                ctk_img = ctk.CTkImage(light_image=image.copy(), size=size)
                self._ctk_images[col] = ctk_img  # referencja, inaczej GC usunie obraz
                label.configure(image=ctk_img, text="")


    def update_sigma_label(self, value):
        self.sigma_label.configure(text=f"Sigma: {value:.2f}")


    def update_alpha_label(self, value):
        self.alpha_label.configure(text=f"Alfa: {value:.2f}")


    @staticmethod
    def set_visible(widgets, visible: bool):
        for w in widgets:
            if visible:
                w.grid()
            else:
                w.grid_remove()


    def on_filter_change(self, choice):
        spec = FILTERS[choice]
        self.set_visible(self.mask_widgets, spec.uses_mask)
        self.set_visible(self.sigma_widgets, spec.uses_sigma)
        self.set_visible(self.fast_widgets, spec.uses_fast)
        self.set_visible(self.alpha_widgets, spec.uses_alpha)


    def load_image(self):
        path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png")],
        )
        if not path:
            return
        self.original_image = image_io.load_rgb(path)
        self.clear()


    def clear(self):
        self.filtered_image = None
        self._render_panels([("Oryginał", self.original_image), ("Po filtrowaniu", None)])

    def apply_filter(self):
        if self.original_image is None:
            print("Najpierw wczytaj obraz")
            return

        spec = FILTERS[self.filter_var.get()]
        params = FilterParams(
            mask_size=int(self.mask_size_var.get()),
            sigma=self.sigma_var.get(),
            fast=self.fast_convolution_var.get(),
            alpha=self.alpha_var.get(),
        )
        out = spec.func(np.array(self.original_image), params)

        if isinstance(out, np.ndarray):  # zwykły filtr: jeden obraz
            out = [("Po filtrowaniu", out)]

        panels = [("Oryginał", self.original_image)]
        panels += [(title, Image.fromarray(arr)) for title, arr in out]
        self.filtered_image = panels[-1][1]  # ostatni panel = wynik końcowy
        self._render_panels(panels)
