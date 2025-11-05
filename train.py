import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emails.csv")

# Transformar o dataset em português ( Usar a lib do google para tradução em massa )

if not os.path.exists(CSV_PATH):

    raise SystemExit(f"Arquivo CSV não encontrado em {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="latin1", engine="python", on_bad_lines="skip")

# Vamos conseguir todas as linhas
# Adicionar NLP para identificar se é ou não