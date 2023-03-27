import string
from fazenda import models
from fazenda import choices

with open("rebanho.csv", "r") as rebanho:
    vacas = rebanho.read().splitlines()

for numero, nome in [vaca.split(",") for vaca in vacas[1:]]:
    p = None
    for prefixo in choices.PREFIXO_CHOICES:
        if nome.split()[0] == prefixo[0]:
            p = prefixo[0]
            nome = nome[len(p) + 1 :]
    models.Vaca.objects.create(numero=numero, nome=string.capwords(nome), prefixo=p)
