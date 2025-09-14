# a) Estime o ruído de seu sistema de aquisição de imagens (pode ser uma webcamera ou
# qualquer outra) usando os passos propostos no algoritmo EST_NOISE e
# AUTO_COVARIANCE, expostos nos slides em sala de aula. Você precisará capturar
# diversas imagens da mesma cena estática, em um ambiente sem variações de iluminação
# (lâmpadas fluorescentes afetam a iluminação das imagens, bem como imagens externas de
# nuvens e árvores, sendo indicada cena interna e com fixação perfeita da câmera), plotando
# depois o gráfico da intensidade da imagem ao longo de uma linha e a variação produzida pelo
# ruído. Esta variação deverá ser pequena, da ordem de menos de 5% em geral, dependendo da
# qualidade da câmera. Discuta os resultados.

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def get_images(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".npz"))
    ]

    if not files:
        raise ValueError("Nenhuma imagem encontrada na pasta.")

    first_file = files[0]

    if first_file.lower().endswith(".npz"):
        # Carrega o arquivo npz
        data = np.load(first_file)
        # Se soubermos que a chave é "resultado"
        if "resultado" in data:
            arr = data["resultado"]
        else:
            # caso contrário, pega a primeira chave disponível
            key = list(data.keys())[0]
            arr = data[key]

        height, width = arr.shape[:2]

    else:
        # Carregar como imagem comum
        img_ref = Image.open(first_file).convert("RGB")
        width, height = img_ref.size

    return {
        "files": files,
        "width": width,
        "height": height
    }

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


def average_images(folder, output="average", img_type="png"):
    images = get_images(folder)
    files = images["files"]
    width = images["width"]
    height = images["height"]

    # Acumula as imagens em numpy arrays
    sum = np.zeros((height, width, 3), dtype=np.float64)

    index = 0
    for file in files:
        np_img = image_to_array(file, width, height)
        # print(f'Imagem-{index}')
        # print(np_img)
        sum += np_img
        index += 1

    # print('Soma')
    # print(sum)

    avg = sum / len(files)

    # print('Media')
    # print(avg)

    save_image(avg, output, img_type)


def variation_images(folder, avg="average", output="variation/", img_type="png"):
    images = get_images(folder)
    files = images["files"]
    width = images["width"]
    height = images["height"]
    avg_name = f"{avg}.{img_type}"
    # garantir tamanho
    avg_img = image_to_array(avg_name, width, height)

    index = 0
    for file in files:
        np_img = image_to_array(file, width, height)
        # print(f'Imagem-{index}')
        # print(np_img)
        sub = np_img - avg_img

        name = f'{output}{index}'
        save_image(sub, name, img_type)
        index += 1


def neighbors_images(folder, output="neighbors/"):
    images = get_images(folder)
    files = images["files"]
    width = images["width"]
    height = images["height"]
    # garantir tamanho

    index = 0
    for file in files:
        # garantir tamanho
        img = image_to_array(file, width, height)
        # Soma total de todos os pixels (por canal)
        soma_total = np.sum(img, axis=(0, 1))  # vetor [R_total, G_total, B_total]

        # Para cada pixel, o novo valor é soma_total - pixel_atual
        resultado = soma_total - img

        name = f'{output}{index}'
        save_image(resultado, name, 'npz')
        index += 1


def deviation_images(folder, avg="average.png", output="deviation", image_type="png"):
    images = get_images(folder)
    files = images["files"]
    width = images["width"]
    height = images["height"]

    # Acumula as imagens em numpy arrays
    sum = np.zeros((height, width, 3), dtype=np.float64)
    # garantir tamanho
    avg_img = image_to_array(avg, width, height)

    index = 0
    for file in files:
        # garantir tamanho
        np_img = image_to_array(file, width, height)
        # print(f'Imagem-{index}')
        # print(np_img)
        sub = avg_img - np_img
        pow = sub ** 2
        sum += pow
        index += 1

    # print('Soma')
    # print(sum)

    avg = sum / len(files)

    # print('Media')
    # print(avg)

    root = avg ** 0.5

    # print('Raiz')
    # print(root)

    save_image(root, output, image_type)


def covariance_image(variation_folder="variation", neighbors_folder="neighbors", neighbors_average="neighbors_average.npz", output="covariance/", img_type="png"):
    variation_images = get_images(variation_folder)
    variation_files = variation_images["files"]
    width = variation_images["width"]
    height = variation_images["height"]
    neighbors_images = get_images(neighbors_folder)
    neighbors_files = neighbors_images["files"]
    n_np_img_avg = image_to_array(neighbors_average, width, height)
    c = 1 / (width * height)

    s = len(variation_files)
    if s != len(variation_files):
        raise Exception("Size Error")

    index = 0
    for index in range(0, s):
        variation_file = variation_files[index]
        neighbors_file = neighbors_files[index]

        v_np_img = image_to_array(variation_file, width, height)
        n_np_img = image_to_array(neighbors_file, width, height)

        result = n_np_img - n_np_img_avg
        result = result * v_np_img
        result = result * c
        name = f"{output}{index}"
        save_image(result, name, img_type)
    



def image_average(input="deviation.png"):
    img = image_to_array(input)
    avg = np.mean(img)
    print(f"Media da imagem {input}: {avg}")
    return avg


def get_arrays(folder, add=["average.png"]):
    images = get_images(folder)
    files: list = images["files"]
    width = images["width"]
    height = images["height"]
    for e in add:
        files.append(e)
    a: list = []
    for file in files:
        print(file)
        np_image = image_to_array(file, width, height)
        a.append(np_image)
    return a


def plot_gray(imgs, output="graf.png"):
    """
    Plota em escala de cinza todas as imagens e a média.
    NÃO inclui o desvio.
    """
    idx_media = len(imgs) - 1  # média é última

    plt.figure(figsize=(12, 6))

    for i, img in enumerate(imgs):
        gray = img.mean(axis=2).flatten()

        if i == idx_media:  # destaque para média
            lw, alpha, estilo, nome, cor = 0.5, 1, "-", "media", "red"
        else:
            lw, alpha, estilo, nome, cor = 0.5, 1, "--", f"img{i}", "black"

        plt.plot(
            gray,
            color=cor,
            linestyle=estilo,
            lw=lw,
            alpha=alpha,
            label=nome if i == idx_media else None,
        )

    plt.title("Curvas em escala de cinza (imagens + média)")
    plt.xlabel("Índice do pixel (flattened)")
    plt.ylabel("Intensidade (0-255)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico (sem desvio) salvo em '{output}'")


def plot_desvio(img, output="graf_desvio.png"):
    """
    Plota apenas a imagem de desvio em grayscale.
    """
    gray = img.mean(axis=2).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(
        gray,
        color="black",
        linestyle="-",
        lw=0.5,
        label="desvio"
    )

    plt.title("Curva em escala de cinza (desvio)")
    plt.xlabel("Índice do pixel (flattened)")
    plt.ylabel("Intensidade (0-255)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico do desvio salvo em '{output}'")

def pixel_p(value):
    return (value / 255) * 100

average_images("pictures")
deviation_images("pictures")

dev = image_average()
dev_p = pixel_p(dev)


variation_images("pictures")
neighbors_images("pictures")
average_images("neighbors", "neighbors_average", "npz")

covariance_image()



var = image_average("variation_average.png")
var_p = pixel_p(var)

print(f"% do devio médio: {dev_p}")
print(f"% da variação média: {var_p}")

arrays = get_arrays("pictures")

plot_gray(arrays)
plot_desvio(image_to_array("deviation.png"))
plot_desvio(image_to_array("variation_average.png"), "graf_variation.png")