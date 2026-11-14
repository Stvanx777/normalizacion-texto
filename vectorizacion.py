import spacy
from gensim.models import Word2Vec

# 1. Cargar el modelo de español
nlp = spacy.load("es_core_news_sm")

# Desactivar componentes que no necesitamos
nlp.disable_pipes("ner")

# 2. Leer el libro original
with open("libro.txt", "r", encoding="utf-8-sig") as archivo:
    texto = archivo.read()

print(f"Libro cargado: {len(texto)} caracteres")

# 3. Dividir el texto en párrafos
parrafos = [
    parrafo.strip()
    for parrafo in texto.split("\n")
    if parrafo.strip()
]

print(f"Párrafos encontrados: {len(parrafos)}")

# 4. Procesar el texto y conservar las oraciones
oraciones = []

for doc in nlp.pipe(parrafos, batch_size=16):
    for oracion in doc.sents:
        palabras = [
            token.lemma_.lower()
            for token in oracion
            if token.is_alpha
            and not token.is_stop
            and len(token.lemma_) > 2
        ]

        if len(palabras) >= 2:
            oraciones.append(palabras)

print(f"Oraciones procesadas: {len(oraciones)}")
print("Ejemplo de oración normalizada:")
print(oraciones[0] if oraciones else "No se encontraron oraciones")

# 5. Entrenar el modelo Word2Vec
print("\n--- ENTRENAMIENTO WORD2VEC ---")

modelo = Word2Vec(
    sentences=oraciones,
    vector_size=50,
    window=5,
    min_count=3,
    workers=1,
    sg=1,
    epochs=15,
    seed=42
)

print("Modelo Word2Vec entrenado correctamente")
print(f"Palabras en el vocabulario: {len(modelo.wv)}")

# 6. Consultar palabras semánticamente relacionadas
palabras_consulta = ["caballero", "amor", "aventura", "sancho"]

for palabra in palabras_consulta:
    if palabra in modelo.wv:
        print(f"\nPalabras relacionadas con '{palabra}':")

        for similar, puntuacion in modelo.wv.most_similar(
            palabra, topn=5
        ):
            print(f"  {similar}: {puntuacion:.4f}")
    else:
        print(f"\nLa palabra '{palabra}' no está en el vocabulario")

# 7. Guardar el modelo entrenado
modelo.save("word2vec_quijote.model")

print("\nModelo guardado: word2vec_quijote.model")

# 8. Visualización del espacio semántico en 2D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("\n--- VISUALIZACIÓN WORD2VEC 2D ---")

# Seleccionar palabras representativas de la novela
palabras_interes = [
    "quijote", "sancho", "caballero",
    "escudero", "aventura", "amor",
    "dulcinea", "rocín", "molino",
    "venta", "camino", "armar",
    "espada", "doncella", "castillo",
    "gigante", "sierra", "morena"
]

# Conservar únicamente palabras presentes en el modelo
palabras_validas = [
    palabra for palabra in palabras_interes
    if palabra in modelo.wv
]

if len(palabras_validas) < 3:
    raise ValueError(
        "No hay suficientes palabras para generar la gráfica"
    )

# Obtener los vectores de las palabras
vectores = [
    modelo.wv[palabra]
    for palabra in palabras_validas
]

# Reducir las 50 dimensiones a 2
pca = PCA(n_components=2)
coordenadas = pca.fit_transform(vectores)

# Crear gráfica
plt.figure(figsize=(12, 8))

plt.scatter(
    coordenadas[:, 0],
    coordenadas[:, 1],
    color="darkblue",
    s=70,
    alpha=0.8
)

# Etiquetar cada palabra
for i, palabra in enumerate(palabras_validas):
    plt.annotate(
        palabra,
        (coordenadas[i, 0], coordenadas[i, 1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10
    )

plt.title(
    "Espacio semántico Word2Vec - Don Quijote de la Mancha"
)
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.grid(alpha=0.3)
plt.tight_layout()

# Guardar imagen
plt.savefig(
    "espacio_semantico_word2vec_2d.png",
    dpi=300
)

plt.close()

print("Gráfica generada correctamente")
print("Archivo: espacio_semantico_word2vec_2d.png")

# 9. Visualización del espacio semántico en 3D
from mpl_toolkits.mplot3d import Axes3D

print("\n--- VISUALIZACIÓN WORD2VEC 3D ---")

# Reducir los vectores de 50 a 3 dimensiones
pca_3d = PCA(n_components=3)
coordenadas_3d = pca_3d.fit_transform(vectores)

# Crear figura tridimensional
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# Representar las palabras
ax.scatter(
    coordenadas_3d[:, 0],
    coordenadas_3d[:, 1],
    coordenadas_3d[:, 2],
    color="darkred",
    s=70,
    alpha=0.8
)

# Agregar etiquetas
for i, palabra in enumerate(palabras_validas):
    ax.text(
        coordenadas_3d[i, 0],
        coordenadas_3d[i, 1],
        coordenadas_3d[i, 2],
        palabra,
        fontsize=9
    )

ax.set_title(
    "Espacio semántico Word2Vec 3D - Don Quijote de la Mancha"
)

ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
ax.set_zlabel("Componente principal 3")

# Ajustar el ángulo de visualización
ax.view_init(elev=25, azim=45)

plt.tight_layout()

# Guardar la segunda imagen
plt.savefig(
    "espacio_semantico_word2vec_3d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Gráfica 3D generada correctamente")
print("Archivo: espacio_semantico_word2vec_3d.png")