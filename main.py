
import spacy
import pandas as pd
from nltk.stem import SnowballStemmer

# 1. Cargar el modelo de español
nlp = spacy.load("es_core_news_sm")

# 2. Leer el libro
with open("libro.txt", "r", encoding="utf-8-sig") as archivo:
    texto = archivo.read()

print(f"Texto cargado: {len(texto)} caracteres.")

# 3. Tokenización
# Procesamos por bloques para admitir libros extensos.
tokens = []

for inicio in range(0, len(texto), 100000):
    bloque = texto[inicio:inicio + 100000]
    doc = nlp(bloque)
    tokens.extend(doc)

print("\n--- TOKENIZACIÓN ---")
print(f"Total de tokens: {len(tokens)}")
print("Primeros 20 tokens:")
print([token.text for token in tokens[:20]])

# 4. Eliminación de ruido
tokens_relevantes = [
    token for token in tokens
    if not token.is_stop
    and not token.is_punct
    and not token.is_space
]

print("\n--- ELIMINACIÓN DE STOP WORDS ---")
print(f"Tokens originales: {len(tokens)}")
print(f"Tokens conservados: {len(tokens_relevantes)}")
print("Ejemplos:")
print([token.text for token in tokens_relevantes[:10]])

# 5. Lematización y normalización
tokens_normalizados = [
    token.lemma_.lower()
    for token in tokens_relevantes
]

print("\n--- LEMATIZACIÓN ---")
print("Primeros 20 lemas:")
print(tokens_normalizados[:20])

# 6. Comparación entre stemming y lematización
stemmer = SnowballStemmer("spanish")

comparacion = []

for token in tokens_relevantes[:30]:
    comparacion.append({
        "Original": token.text,
        "Stemming": stemmer.stem(token.text),
        "Lematización": token.lemma_.lower()
    })

df = pd.DataFrame(comparacion)

print("\n--- STEMMING VS LEMATIZACIÓN ---")
print(df.to_string(index=False))

# 7. Guardar el resultado de la limpieza
with open("texto_normalizado.txt", "w", encoding="utf-8") as archivo:
    archivo.write(" ".join(tokens_normalizados))

print("\nProcesamiento terminado correctamente.")
print("Archivo generado: texto_normalizado.txt")
