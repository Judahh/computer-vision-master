# b) Escreva ou utilize rotinas para inserção de ruído gaussiano e do tipo sal-e-pimenta e, após,
# implemente um filtro de ruídos Gaussiano e Mediano (não é filtro média) e aplique sobre as
# imagens, mostrando o resultado da aplicação dos dois filtros sobre as duas imagens.
# O código ou a função utilizada deve permitir a especificação da largura do filtro, para percepção
# dos resultados. Mostre as imagens e discuta os resultados.
import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize


# -------------------------------
# Função para adicionar ruído Gaussiano
# -------------------------------
def add_gaussian_noise(image, mean=0, var=20):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# -------------------------------
# Função para adicionar ruído Sal-e-Pimenta
# -------------------------------
def add_salt_pepper_noise(image, prob=0.02):
    noisy = np.copy(image)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy[rnd < prob / 2] = 0           # pixels pretos (pimenta)
    noisy[rnd > 1 - prob / 2] = 255     # pixels brancos (sal)
    return noisy


def median_filter(I, n=3):
    """
    Aplica o filtro mediano em uma imagem em escala de cinza
    :param I: imagem de entrada (numpy array 2D)
    :param n: tamanho da janela (ímpar)
    :return: imagem filtrada
    """
    # Verificar se n é ímpar
    if n % 2 == 0:
        raise ValueError("O tamanho da máscara deve ser ímpar!")

    # Dimensões da imagem
    rows, cols = I.shape
    # Saída inicializada
    Im = np.zeros_like(I)

    # Quanto andar para cada lado
    pad = n // 2

    # Preenche as bordas da imagem (padding)
    padded = np.pad(I, pad, mode="edge")

    for i in range(rows):
        for j in range(cols):
            # Extrair vizinhança n x n
            window = padded[i : i + n, j : j + n]
            # Calcular a mediana dos elementos da janela
            m = np.median(window)
            # Atribuir ao pixel de saída
            Im[i, j] = m

    return Im

def gaussian_kernel_1d(size: int, sigma: float):
    """
    Cria um kernel gaussiano 1D
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-(ax**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # normaliza
    return kernel

# SEPAR_FILTER
def separable_gaussian_filter(I, size=5, sigma=1.0):
    """
    Aplica convolução gaussiana usando separabilidade (2x convolução 1D).
    :param I: imagem em escala de cinza (numpy array 2D)
    :param size: tamanho da máscara gaussiana (ímpar)
    :param sigma: desvio padrão da gaussiana
    :return: imagem filtrada
    """
    g = gaussian_kernel_1d(size, sigma)
    pad = size // 2

    # Padding (replicação das bordas)
    I_padded = np.pad(I, ((0, 0), (pad, pad)), mode="edge")
    rows, cols = I.shape

    # 1ª etapa: convolução horizontal
    I_r = np.zeros_like(I, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            region = I_padded[i, j : j + size]
            I_r[i, j] = np.sum(region * g)

    # Padding vertical
    I_r_padded = np.pad(I_r, ((pad, pad), (0, 0)), mode="edge")

    # 2ª etapa: convolução vertical
    I_out = np.zeros_like(I_r, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            region = I_r_padded[i : i + size, j]
            I_out[i, j] = np.sum(region * g)

    return np.clip(I_out, 0, 255).astype(np.uint8)

# -------------------------------
# Função para aplicar filtros
# -------------------------------
def apply_filters_cv2(image, ksize=5):
    # filtro Gaussiano
    gauss_filtered = cv2.GaussianBlur(image, (ksize, ksize), 0)
    # filtro Mediano
    median_filtered = cv2.medianBlur(image, ksize)
    # ambos
    both_filtered = cv2.GaussianBlur(median_filtered, (ksize, ksize), 0)
    return gauss_filtered, median_filtered, both_filtered

def apply_filters(image, ksize=5):
    gauss_filtered = np.zeros_like(image)
    median_filtered = np.zeros_like(image)
    both_filtered = np.zeros_like(image)

    for c in range(3):  # para cada canal
        gauss_filtered[:, :, c] = separable_gaussian_filter(
            image[:, :, c], ksize, sigma=1.0
        )
        median_filtered[:, :, c] = median_filter(image[:, :, c], ksize)
        both_filtered[:, :, c] = separable_gaussian_filter(
            median_filtered[:, :, c], ksize, sigma=1.0
        )

    return gauss_filtered, median_filtered, both_filtered

def image_to_array(file, width=None, height=None):
    if file.lower().endswith(".npz"):
        # Carregar npz
        data = np.load(file)
        # se tiver a chave "resultado", usa ela. Se não, pega a primeira existente
        arr = data["resultado"] if "resultado" in data else data[list(data.keys())[0]]

        # garantir dtype float64
        np_img = np.array(arr, dtype=np.float64)

        if width is not None and height is not None:
            # Se precisar forçar resize (padrão igual ao das imagens PIL)
            if (np_img.shape[1], np_img.shape[0]) != (width, height):
                np_img = resize(
                    np_img, (height, width, np_img.shape[2]), preserve_range=True
                ).astype(np.float64)

    else:
        # Imagem regular
        img = Image.open(file).convert("RGB")
        if width is not None and height is not None:
            img = img.resize((width, height))
        np_img = np.array(img, dtype=np.float64)

    return np_img


def save_image(output, name, img_type):
    name = f"{name}.{img_type}"
    if img_type.lower() == "npz":
        np.savez_compressed(name, resultado=output)
    else:
        output = np.clip(output, 0, 255).astype(np.uint8)
        # Salva a imagem média
        img_final = Image.fromarray(output, mode="RGB")
        img_final.save(name)
    print(f"Imagem salva como {name}")

# Carregar imagem (exemplo com imagem colorida)
image = cv2.imread("average.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converter pra RGB

# add noise
noisy_gaussian = add_gaussian_noise(image)
noisy_sp = add_salt_pepper_noise(image)

# save images with noise
save_image(noisy_gaussian, "noisy_gaussian", "png")
save_image(noisy_sp, "noisy_sp", "png")

# Aplicar filtros
gauss_filt_g, median_filt_g, both_filt_g, = apply_filters(noisy_gaussian)
gauss_filt_sp, median_filt_sp, both_filt_sp = apply_filters(noisy_sp)
gauss_filt_cv_g, median_filt_cv_g, both_filt_cv_g, = apply_filters_cv2(noisy_gaussian)
gauss_filt_cv_sp, median_filt_cv_sp, both_filt_cv_sp = apply_filters_cv2(noisy_sp)

# save images with filter
save_image(gauss_filt_g, "gaussian_ruido_gaussiano", "png")
save_image(median_filt_g, "mediana_ruido_gaussiano", "png")
save_image(both_filt_g, "gaussian+mediana_ruido_gaussiano", "png")

save_image(gauss_filt_sp, "gaussian_ruido_sal_pimenta", "png")
save_image(median_filt_sp, "mediana_ruido_sal_pimenta", "png")
save_image(both_filt_sp, "gaussian+mediana_ruido_sal_pimenta", "png")

# versões OpenCV
save_image(gauss_filt_cv_g, "cv2_gaussian_ruido_gaussiano", "png")
save_image(median_filt_cv_g, "cv2_mediana_ruido_gaussiano", "png")
save_image(both_filt_cv_g, "cv2_gaussian+mediana_ruido_gaussiano", "png")

save_image(gauss_filt_cv_sp, "cv2_gaussian_ruido_sal_pimenta", "png")
save_image(median_filt_cv_sp, "cv2_mediana_ruido_sal_pimenta", "png")
save_image(both_filt_cv_sp, "cv2_gaussian+mediana_ruido_sal_pimenta", "png")

# Aplicar filtros
gauss_filt_g, median_filt_g, both_filt_g, = apply_filters(noisy_gaussian, 3)
gauss_filt_sp, median_filt_sp, both_filt_sp = apply_filters(noisy_sp, 3)
gauss_filt_cv_g, median_filt_cv_g, both_filt_cv_g, = apply_filters_cv2(noisy_gaussian, 3)
gauss_filt_cv_sp, median_filt_cv_sp, both_filt_cv_sp = apply_filters_cv2(noisy_sp, 3)

# save images with filter
save_image(gauss_filt_g, "gaussian_ruido_gaussiano_3", "png")
save_image(median_filt_g, "mediana_ruido_gaussiano_3", "png")
save_image(both_filt_g, "gaussian+mediana_ruido_gaussiano_3", "png")

save_image(gauss_filt_sp, "gaussian_ruido_sal_pimenta_3", "png")
save_image(median_filt_sp, "mediana_ruido_sal_pimenta_3", "png")
save_image(both_filt_sp, "gaussian+mediana_ruido_sal_pimenta_3", "png")

# versões OpenCV
save_image(gauss_filt_cv_g, "cv2_gaussian_ruido_gaussiano_3", "png")
save_image(median_filt_cv_g, "cv2_mediana_ruido_gaussiano_3", "png")
save_image(both_filt_cv_g, "cv2_gaussian+mediana_ruido_gaussiano_3", "png")

save_image(gauss_filt_cv_sp, "cv2_gaussian_ruido_sal_pimenta_3", "png")
save_image(median_filt_cv_sp, "cv2_mediana_ruido_sal_pimenta_3", "png")
save_image(both_filt_cv_sp, "cv2_gaussian+mediana_ruido_sal_pimenta_3", "png")
