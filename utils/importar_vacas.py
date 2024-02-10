import string
from fazenda import models
from fazenda import choices

def importar_vacas(file):
    vacas = file.splitlines()

    for numero, nome in [vaca.split(",") for vaca in vacas[1:]]:
        p = None
        for prefixo in choices.PREFIXO_CHOICES:
            if nome and nome.split()[0] == prefixo[0]:
                p = prefixo[0]
                nome = nome[len(p) + 1 :]
        models.Vaca.objects.create(numero=numero, nome=string.capwords(nome), prefixo=p)

if __name__ == "__main__":
    importar_vacas("rebanho.csv")
