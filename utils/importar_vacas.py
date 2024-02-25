import string
from fazenda import models
from fazenda import choices

def importar_vacas(file):
    file = file.decode('latin-1')
    vacas = file.splitlines()

    for numero, prefixo, nome in [vaca.split(";") for vaca in vacas[1:]]:
        models.Vaca.objects.create(numero=numero, nome=string.capwords(nome), prefixo=prefixo)

if __name__ == "__main__":
    importar_vacas("rebanho.csv")
