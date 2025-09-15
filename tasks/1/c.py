# c) Use uma rotina qualquer para o algoritmo CANNY e produza diversas imagens indicando
# os efeitos de escala (uma escala menor se refere a uma imagem vista de perto ou zoom em
# uma determinada área) e os limiares de contraste nos contornos detectados. Implemente os
# detectores de Robert e Sobel nas mesmas imagens, para uma única escala, e compare os três
# algoritmos. Mostre as imagens originais. Sugestão: Use a função icanny fornecida pela
# Machine Vison ToolBox, que pode ser baixada do site www.petercorke.com. Podem ser
# usadas quaisquer funções de outras bibliotecas que correspondam ao algoritmo Canny.
# Comente os resultados.
import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize

def linear_filter(I: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Implementação manual de filtro linear 2D por convolução.
    I: imagem (numpy array 2D)
    A: máscara do filtro (numpy array m x m, pode ser par ou ímpar)
    """
    m = A.shape[0]  # tamanho do kernel (supomos quadrado)
    N, M = I.shape
    IA = np.zeros_like(I, dtype=np.float64)

    if m % 2 == 1:
        # Caso ímpar: centro está bem definido
        offset = m // 2
        i_start, i_end = offset, N - offset
        j_start, j_end = offset, M - offset
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                region = I[i - offset:i + offset + 1, j - offset:j + offset + 1]
                IA[i, j] = np.sum(A * region)

    else:
        # Caso par: usar offset "meio a meio"
        offset = m // 2
        i_start, i_end = offset - 1, N - offset
        j_start, j_end = offset - 1, M - offset
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                region = I[i - offset + 1:i + offset + 1,
                           j - offset + 1:j + offset + 1]
                IA[i, j] = np.sum(A * region)

    # Normalizar para [0,255] se imagem
    IA = np.clip(IA, 0, 255)
    return IA.astype(np.uint8)

def roberts_step_1(img: np.ndarray):
    # filtro feijão com arroz
    img_s = cv2.GaussianBlur(img, (5, 5), 1)
    return img_s


def roberts_step_2(img_s: np.ndarray):
    # mascaras
    kx = np.array([
        [1, -1],
        [-1, 1]
    ], dtype=np.float32)
    ky = np.array([
        [-1, 1],
        [1, -1]
    ], dtype=np.float32)
    gx = linear_filter(kx, img_s)
    gy = linear_filter(ky, img_s)
    return gx, gy


def sobel_step_2(img_s: np.ndarray):
    # mascaras
    kx = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)
    ky = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    gx = linear_filter(kx, img_s)
    gy = linear_filter(ky, img_s)
    return gx, gy


def roberts_step_3(gx, gy):
    # Magnitude
    g = np.sqrt(gx**2 + gy**2)

    # Normalizar para 0-255
    g = (g / g.max()) * 255
    g = g.astype(np.uint8)
    return g


def roberts_step_4(g, tau):
    edges = np.zeros_like(g, dtype=np.uint8)
    edges[g > tau] = 255
    return edges


def roberts_edge_det(img: np.ndarray, tau: float) -> np.ndarray:
    img_s = roberts_step_1(img)

    gx, gy = roberts_step_2(img_s)

    g = roberts_step_3(gx, gy)

    return roberts_step_4(g, tau)


def sobel_edge_det(img: np.ndarray, tau: float) -> np.ndarray:
    img_s = roberts_step_1(img)

    gx, gy = sobel_step_2(img_s)

    g = roberts_step_3(gx, gy)

    return roberts_step_4(g, tau)

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
        if len(output.shape) == 2:
            img_final = Image.fromarray(output, mode="L")
        else:
            img_final = Image.fromarray(output, mode="RGB")
        img_final.save(name)
    print(f"Imagem salva como {name}")

# 1. Carregar imagem (em RGB e depois converter para grayscale)
arr = image_to_array("average.png")
img_gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)

# 2. Aplicar detectores
edges_roberts_50 = roberts_edge_det(img_gray, tau=50)
edges_sobel_50 = sobel_edge_det(img_gray, tau=50)

edges_roberts_100 = roberts_edge_det(img_gray, tau=100)
edges_sobel_100 = sobel_edge_det(img_gray, tau=100)

# Canny com dois thresholds
edges_canny_1 = cv2.Canny(img_gray, 50, 100)
edges_canny_2 = cv2.Canny(img_gray, 100, 200)

# 3. Salvar resultados
save_image(img_gray, "gray", "png")
save_image(edges_roberts_50, "edges_roberts_50", "png")
save_image(edges_sobel_50, "edges_sobel_50", "png")
save_image(edges_roberts_100, "edges_roberts_100", "png")
save_image(edges_sobel_100, "edges_sobel_100", "png")
save_image(edges_canny_1, "edges_canny_50_100", "png")
save_image(edges_canny_2, "edges_canny_100_200", "png")
