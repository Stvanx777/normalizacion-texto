# Normalización y Lematización de Texto

## Descripción

Proyecto académico de Procesamiento de Lenguaje Natural (PLN) desarrollado en Python.

Se utiliza el libro _Don Quijote de la Mancha_, de Miguel de Cervantes, para aplicar diferentes técnicas de limpieza y procesamiento de texto.

## Técnicas implementadas

- Tokenización mediante spaCy.
- Eliminación de stop words.
- Eliminación de signos de puntuación y espacios.
- Normalización mediante conversión a minúsculas.
- Lematización.
- Comparación entre stemming y lematización utilizando NLTK.

## Tecnologías

- Python
- spaCy
- NLTK
- pandas

## Instalación

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Ejecución

```bash
python main.py
```

## Resultados

El programa muestra estadísticas de tokenización, palabras conservadas después del filtrado, ejemplos de lematización y una comparación entre stemming y lematización.

También genera el archivo `texto_normalizado.txt` con el resultado del procesamiento.

## Propósito académico

Comprender cómo las técnicas de normalización y lematización permiten reducir las variaciones morfológicas del lenguaje natural para facilitar su procesamiento computacional.
