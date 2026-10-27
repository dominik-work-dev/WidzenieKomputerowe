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

        self._build_top_panel()
        self._build_image_panel()
        self.on_filter_change(self.filter_var.get())


    def _build_top_panel(self):
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=10, pady=10)
        top.grid_columnconfigure(8, weight=1)

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

        ctk.CTkButton(top, text="Wyczyść obraz", command=self.clear).grid(
            row=0, column=9, padx=10, pady=10)
        ctk.CTkButton(
            top, text="Filtruj", command=self.apply_filter,
            fg_color="green", hover_color="darkgreen",
        ).grid(row=0, column=10, padx=10, pady=10, sticky="e")

        self.mask_widgets = (self.mask_label, self.mask_menu)
        self.sigma_widgets = (self.sigma_label, self.sigma_slider)
        self.fast_widgets = (self.fast_checkbox,)


    def _build_image_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)
        frame.columnconfigure((0, 1), weight=1)
        frame.rowconfigure(1, weight=1)

        for col, title in enumerate(("Oryginał", "Po filtrowaniu")):
            ctk.CTkLabel(frame, text=title, font=("Arial", 16, "bold")).grid(row=0, column=col, pady=(10, 0))

        self.label_orig = ctk.CTkLabel(frame, text="Brak obrazu")
        self.label_orig.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.label_filt = ctk.CTkLabel(frame, text="Brak obrazu")
        self.label_filt.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")


    def update_sigma_label(self, value):
        self.sigma_label.configure(text=f"Sigma: {value:.2f}")


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


    def load_image(self):
        path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png")],
        )
        if not path:
            return
        self.original_image = image_io.load_rgb(path)
        self._show(self.original_image, self.label_orig, "orig")
        self.clear()


    def clear(self):
        self.filtered_image = None
        self._ctk_images.pop("filt", None)
        self.label_filt.configure(image=None, text="Brak obrazu")


    def _show(self, image: Image.Image, label: ctk.CTkLabel, key: str):
        ctk_img = ctk.CTkImage(light_image=image.copy(), size=image.size)
        self._ctk_images[key] = ctk_img  # inaczej GC może usunąć obraz
        label.configure(image=ctk_img, text="")

    def apply_filter(self):
        if self.original_image is None:
            print("Najpierw wczytaj obraz")
            return

        spec = FILTERS[self.filter_var.get()]
        params = FilterParams(
            mask_size=int(self.mask_size_var.get()),
            sigma=self.sigma_var.get(),
            fast=self.fast_convolution_var.get(),
        )
        print(f"Filtrowanie: {spec.name}")

        img_array = np.array(self.original_image)  # (512, 512, 3)
        result_array = spec.func(img_array, params)

        self.filtered_image = Image.fromarray(np.uint8(result_array))
        self._show(self.filtered_image, self.label_filt, "filt")
