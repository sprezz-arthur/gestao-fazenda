import logging
import string

from fazenda import models


def criar_vaca(vaca_excel_row: str) -> None:
    split_char = ";"
    if ";" in vaca_excel_row:
        split_char = ";"
    else:
        split_char = ","

    numero_prefixo, numero_nome = vaca_excel_row.split(split_char)

    num1, *prefixo = numero_prefixo.split()
    num1 = num1.zfill(3)
    prefixo = " ".join(prefixo)

    num2, *nome = numero_nome.split()
    num2 = num2.zfill(3)
    nome = " ".join(nome)

    assert num1 == num2, f"num1: {num1} != num2: {num2}"
    models.Vaca.objects.create(numero=num1, nome=string.capwords(nome), prefixo=prefixo)


def importar_vacas(file):
    vacas = file.splitlines()

    for vaca_row in vacas[1:]:
        try:
            criar_vaca(vaca_row)
        except Exception as e:
            logging.warning(e)


if __name__ == "__main__":
    importar_vacas("rebanho.csv")
