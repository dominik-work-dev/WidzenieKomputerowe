from tkinter import filedialog
import customtkinter as ctk
from PIL import Image
import numpy as np
from src.app import image_io

# Dominik Sobieraj
# Filtrowanie splotowe
# GENAI Podpowiedziało zamysł i idee zoptymalizowanego algorytmu splotu, gdzie nie przechodzimy po pixelach, ale po masce mnożąc jej wartości przez cały obraz ciągle dodając do finalnej wersji.

MASK_SIZES = [3, 5, 7, 9]
FILTERS = ["Uśredniający", "Gaussa", "Gaussa1d", "Wyostrzający", "Sobela", "Laplace"]

class FiltrowanieSplotowe(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Filtrowanie splotowe")
        self.geometry("1600x1200")
        self.minsize(1600,1200)

        self.original_image = None
        self.filtered_image = None
        self._ctk_images: dict[str, ctk.CTkImage] = {} # do trzymania referencji obrazów

        self.mask_size_var = ctk.StringVar(value = str(MASK_SIZES[0]))
        self.filter_var = ctk.StringVar(value = FILTERS[0])
        self.sigma_var = ctk.DoubleVar(value = 1)
        self.fast_convolution_var = ctk.BooleanVar(value = True)

        self._build_top_panel()
        self._build_image_panel()
        self.on_filter_change(self.filter_var.get())

        self.filter_functions = {
            "Uśredniający": self.filter_average,
            "Gaussa": self.filter_gaussian,
            "Gaussa1d": self.filter_gaussian1d,
            "Wyostrzający": self.filter_sharpen,
            "Sobela": self.filter_sobel,
            "Laplace": self.filter_laplace,
        }


    def _build_top_panel(self):
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=10, pady=10)

        ctk.CTkButton(top, text="Wczytaj obraz", command=self.load_image).pack(side="left", padx=10, pady=10)
        ctk.CTkLabel(top, text="Wielkość maski:").pack(side="left", padx=10, pady=10)
        self.mask_menu = ctk.CTkOptionMenu(top, variable=self.mask_size_var, values=[str(x) for x in MASK_SIZES])
        self.mask_menu.pack(side="left", padx=5, pady=10)

        ctk.CTkLabel(top, text="Typ filtra:").pack(side="left", padx=10, pady=10)
        ctk.CTkOptionMenu(
            top,
            variable=self.filter_var,
            values=FILTERS,
            command=self.on_filter_change
        ).pack(side="left", padx=5, pady=10)

        self.sigma_label = ctk.CTkLabel(top, text="Sigma: 1.00")
        self.sigma_label.pack(side="left", padx=10, pady=10)
        self.sigma_slider = ctk.CTkSlider(
            top, from_=0.1, to=5.0, number_of_steps=49,  # type: ignore
            variable=self.sigma_var, command=self.update_sigma_label, width=120)
        self.sigma_slider.pack(side="left", padx=5, pady=10)

        self.fastConvolution_checkbox = ctk.CTkCheckBox(top, variable=self.fast_convolution_var, text="Szybki splot")
        self.fastConvolution_checkbox.pack(side="left", padx=5, pady=10)

        ctk.CTkButton(top, text="Wyczyść obraz", command=self.clear).pack(
            side="left", padx=10, pady=10)
        ctk.CTkButton(
            top, text="Filtruj", command=self.apply_filter,
            fg_color="green", hover_color="darkgreen",
        ).pack(side="right", padx=10, pady=10)

    def _build_image_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)
        frame.columnconfigure((0, 1), weight=1)
        frame.rowconfigure(1, weight=1)

        for col, title in enumerate(("Oryginał", "Po filtrowaniu")):
            ctk.CTkLabel(frame, text=title, font=("Arial", 16, "bold")).grid(
                row=0, column=col, pady=(10, 0))

        self.label_orig = ctk.CTkLabel(frame, text="Brak obrazu")
        self.label_orig.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.label_filt = ctk.CTkLabel(frame, text="Brak obrazu")
        self.label_filt.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")


    def update_sigma_label(self, value):
        self.sigma_label.configure(text=f"Sigma: {value:.2f}")

    def on_filter_change(self, choice):
        # Aktywuj suwak tylko dla filtra Gaussa
        if choice == "Gaussa" or choice =="Gaussa1d":
            self.sigma_slider.configure(state="normal")
        else:
            self.sigma_slider.configure(state="disabled")

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

        mask_size = int(self.mask_size_var.get())
        filter_type = self.filter_var.get()

        print(f"Filtrowanie: {filter_type} z maską {mask_size}x{mask_size}")
        img_array = np.array(self.original_image)   # img w numpy ma format (512, 512, 3)

        #Genai wygenerowało implementacje wywoływania funkcji w zależności od zmiennej oraz przekonwertowanie obrazu na format ctk
        selected_function = self.filter_functions.get(filter_type)
        if selected_function:
            result_array = selected_function(
                img_array,
                mask_size=mask_size,
                sigma=self.sigma_var.get()
            )
        else:
            result_array = img_array.copy()

        self.filtered_image = Image.fromarray(np.uint8(result_array))
        self._show(self.filtered_image, self.label_filt, "filt")

    # Filtry
    def filter_average(self, img_array, mask_size, **kwargs):
        kernel = np.ones((mask_size, mask_size)) / (mask_size ** 2) # stworzenie maski (n x n) i podzielenie jej przez broadcasting, żeby każdy element miał taką samą wartość
        return self.apply_convolution(img_array, kernel, mask_size, fast_convolution=self.fast_convolution_var.get())

    def filter_gaussian(self, img_array, mask_size, sigma, **kwargs):
        half = mask_size // 2
        kernel = []
        total_sum = 0.0
        for y in range(-half, half + 1):
            row = []
            for x in range(-half, half + 1):
                val = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
                row.append(val)
                total_sum += val
            kernel.append(row)

        for y in range(mask_size):
            for x in range(mask_size):
                kernel[y][x] /= total_sum

        kernel_np = np.array(kernel)
        return self.apply_convolution(img_array, kernel_np, mask_size, fast_convolution=self.fast_convolution_var.get())

    def filter_gaussian1d(self,img_array,sigma, **kwargs):
        half = int(np.ceil(3.0 * sigma))
        x = np.arange(-half, half + 1)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel /= np.sum(kernel)
        mask_size = kernel.shape[0]
        kernel_np = np.array(kernel)
        return self.apply_convolution1d(img_array, kernel_np, mask_size)

    def filter_sharpen(self, img_array, mask_size, **kwargs):
        kernel = -np.ones((mask_size, mask_size))
        center = mask_size // 2
        kernel[center, center] = mask_size ** 2 # dla 3x3, środek bedzie 9, reszta -8
        return self.apply_convolution(img_array, kernel, mask_size, fast_convolution=self.fast_convolution_var.get())

    def filter_sobel(self, img_array, mask_size, **kwargs):
        if mask_size != 3:
            print("Uwaga: Wymuszam maskę 3x3")

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = self.apply_convolution(img_array, kx, 3, raw_output=True)
        grad_y = self.apply_convolution(img_array, ky, 3, raw_output=True)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    def filter_laplace(self, img_array, mask_size, **kwargs):
        if mask_size != 3:
            print("Uwaga: Wymuszam maskę 3x3")

        kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        return self.apply_convolution(
            img_array,
            kernel,
            3,
            fast_convolution=self.fast_convolution_var.get(),
        )

    # Splot
    @staticmethod
    def apply_convolution(image_array, kernel, mask_size, raw_output=False, fast_convolution=True):
        pad = mask_size // 2

        # Dodanie marginesów
        padded_image = np.pad(image_array, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        output = np.zeros_like(image_array, dtype=np.float32)

        if fast_convolution:# Przechodzimy po masce nie po obrazku
            for i in range(mask_size):
                for j in range(mask_size):
                    # pixel kernela * każdy pixel obrazka od pixela i,j
                    output += kernel[i, j] * padded_image[i:i + image_array.shape[0], j:j + image_array.shape[1], :]

        else: # Przechodzenie po każdym pixelu
            height, width = image_array.shape[:2]
            k_height, k_width = kernel.shape[:2]
            output = np.zeros_like(image_array, dtype=np.float64)
            for y in range(1,height - 1):
                for x in range(1,width - 1):
                    pixel_sum = 0.0
                    for ky in range(k_height):
                        for kx in range(k_width):
                            weight = kernel[ky, kx]
                            neighbor_pixel = padded_image[y + ky, x + kx]
                            pixel_sum += weight * neighbor_pixel
                    output[y, x] = pixel_sum

        if raw_output:
            return output

        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_convolution1d(image_array, kernel, mask_size):
        pad = mask_size // 2
        padded_v = np.pad(image_array, ((pad, pad), (0,0), (0, 0)), mode='edge')
        tmp_output = np.zeros_like(image_array, dtype=np.float32)

        k_v = kernel[:, np.newaxis, np.newaxis]
        k_h = kernel[np.newaxis, :, np.newaxis]

        for i in range(mask_size):
            tmp_output += k_v[i] * padded_v[i:i + image_array.shape[0],:,:]

        padded_h = np.pad(tmp_output, ((0, 0), (pad, pad), (0, 0)), mode='edge')
        final_output = np.zeros_like(image_array, dtype=np.float32)
        for j in range(mask_size):
            final_output += k_h[0, j] * padded_h[:, j:j + image_array.shape[1],:]

        return np.clip(final_output, 0, 255).astype(np.uint8)
