# d) Implemente o algoritmo CORNERS (mostrado nos slides) e use uma interface que mostre
# os cantos superpostos às imagens originais. Construa uma imagem sintética com um quadrado
# branco sobre um fundo preto e teste o algoritmo. Compare os resultados com o algoritmo de
# Harris, nas mesmas imagens. Em seguida use a imagem disponível no site do curso
# (building2-1.png). Rode a função icorner no Matlab, fornecida pela Machine Vison ToolBox
# que pode ser baixada do site www.petercorke.com (algoritmo de detector de cantos de Harris)
# ou qualquer outra função que realize operação semelhante, sobre a mesma imagem (o limiar
# tem de ter escolha adequada para fazer comparação com os outros algoritmos). Faça o mesmo
# para os métodos SURF (pode ser o ORB que é de livre instalação. O SURF pode ter restrições
# na versão free) e SIFT na plataforma que você estiver utilizando. Discuta os resultados.
import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize

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

def gradients(img, N):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Derivada em x
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Derivada em y

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Suavização
    ksize = 2 * N + 1
    Sx2 = cv2.GaussianBlur(Ix2, (ksize, ksize), 1)
    Sy2 = cv2.GaussianBlur(Iy2, (ksize, ksize), 1)
    Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), 1)
    return Sx2, Sy2, Sxy

def corners_detector(img, N=3, tau=1e-4):
    # 1. Converter para escala de cinza (caso seja RGB)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = np.float32(gray) / 255.0
    h, w = gray.shape

    # 2. Gradientes
    Sx2, Sy2, Sxy = gradients(gray, N)

    # Resposta λ2 em cada pixel
    lambda2 = np.zeros((h, w))
    pontos = []

    # 2. Para cada pixel p
    for y in range(h):
        for x in range(w):
            # a) Construir matriz C
            C = np.array([[Sx2[y, x], Sxy[y, x]], [Sxy[y, x], Sy2[y, x]]])

            # b) Calcular menor autovalor
            eigvals = np.linalg.eigvalsh(C)
            l2 = np.min(eigvals)
            lambda2[y, x] = l2

            # c) Testar com τ
            if l2 > tau:
                pontos.append((y, x, l2))

    # 3. Ordenar lista em ordem decrescente de λ2
    pontos.sort(key=lambda p: p[2], reverse=True)

    # 4. Supressão de não máximos: evitar vizinhanças sobrepostas
    final_points = []
    marcado = np.zeros((h, w), dtype=bool)

    for (y, x, val) in pontos:
        if not marcado[y, x]:
            final_points.append((y, x, val))
            # marcar vizinhança como ocupada
            y1, y2 = max(0, y - N), min(h, y + N + 1)
            x1, x2 = max(0, x - N), min(w, x + N + 1)
            marcado[y1:y2, x1:x2] = True

    return final_points, lambda2


# ============================================================
# Imagem Sintética
# ============================================================
img_synthetic = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img_synthetic, (50, 50), (150, 150), 255, -1)

corners, response = corners_detector(img_synthetic, N=3, tau=0.05)

# Visualização
img_color = cv2.cvtColor(img_synthetic, cv2.COLOR_GRAY2BGR)
for y, x, val in corners:
    cv2.circle(img_color, (x, y), 3, (0, 0, 255), -1)

# Salvar resultados
save_image(img_color, "synthetic_corners", "png")
save_image(response, "synthetic_lambda2", "npz")

# --- Harris ---
harris = cv2.cornerHarris(np.float32(img_synthetic) / 255.0, 2, 3, 0.04)
harris_img = cv2.cvtColor(img_synthetic, cv2.COLOR_GRAY2BGR)
harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]
save_image(harris_img, "synthetic_harris", "png")
save_image(harris, "synthetic_response", "npz")

# --- ORB ---
orb = cv2.ORB_create()
kp_orb = orb.detect(img_synthetic, None)
orb_img = cv2.drawKeypoints(img_synthetic, kp_orb, None, color=(0, 255, 0))
save_image(orb_img, "synthetic_orb", "png")

# --- SIFT ---
sift = cv2.SIFT_create()
kp_sift, _ = sift.detectAndCompute(img_synthetic, None)
sift_img = cv2.drawKeypoints(img_synthetic, kp_sift, None, color=(255, 0, 0))
save_image(sift_img, "synthetic_sift", "png")


# ============================================================
# building2-1.png
# ============================================================
img = cv2.imread("building2-1.png", cv2.IMREAD_GRAYSCALE)


corners, response = corners_detector(img, N=3, tau=0.05)

# Visualização
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for y, x, val in corners:
    cv2.circle(img_color, (x, y), 3, (0, 0, 255), -1)

# Salvar resultados
save_image(img_color, "building_corners", "png")
save_image(response, "building_lambda2", "npz")

# --- Harris ---
harris = cv2.cornerHarris(np.float32(img) / 255.0, 2, 3, 0.04)
harris_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]
save_image(harris_img, "building_harris", "png")
save_image(harris, "building_harris_response", "npz")

# --- ORB ---
orb = cv2.ORB_create()
kp_orb = orb.detect(img, None)
orb_img = cv2.drawKeypoints(img, kp_orb, None, color=(0, 255, 0))
save_image(orb_img, "building_orb", "png")

# --- SIFT ---
sift = cv2.SIFT_create()
kp_sift, _ = sift.detectAndCompute(img, None)
sift_img = cv2.drawKeypoints(img, kp_sift, None, color=(255, 0, 0))
save_image(sift_img, "building_sift", "png")