import csv
import os
import time

from deep_translator import GoogleTranslator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CSV = os.path.join(BASE_DIR, "emails.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "emails_pt.csv")
TEMP_CSV = OUTPUT_CSV + ".tmp"


def traduzir_texto(texto):
    if texto is None or not texto.strip() or texto.lower() == "nan":
        return texto
    try:
        return GoogleTranslator(source="en", target="pt").translate(texto[:4500]) or texto
    except Exception as error:
        print(f"Não foi possível traduzir o trecho: {error}")
        return texto


def traduzir_arquivo():
    if not os.path.exists(SOURCE_CSV):
        raise FileNotFoundError(f"Arquivo de origem não encontrado: {SOURCE_CSV}")

    with open(SOURCE_CSV, mode="r", encoding="utf-8", newline="") as entrada:
        leitor = csv.DictReader(entrada)
        campos = leitor.fieldnames
        if not campos:
            raise ValueError("O CSV está vazio ou não tem cabeçalho.")

        with open(TEMP_CSV, mode="w", encoding="utf-8-sig", newline="") as saida:
            escritor = csv.DictWriter(saida, fieldnames=campos)
            escritor.writeheader()
            for numero, linha in enumerate(leitor, start=1):
                for coluna, valor in linha.items():
                    if valor and coluna.lower() not in ("spam", "label", "class"):
                        linha[coluna] = traduzir_texto(valor)
                        time.sleep(0.3)
                escritor.writerow(linha)
                if numero % 10 == 0:
                    print(f"{numero} linhas processadas.")

    os.replace(TEMP_CSV, OUTPUT_CSV)
    print(f"Arquivo traduzido salvo em: {OUTPUT_CSV}")


if __name__ == "__main__":
    traduzir_arquivo()
