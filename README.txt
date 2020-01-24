El codi que podreu trobar amb el nom de "ScriptFinal.R" el podeu executar de principi a final amb un dels data sets balancejats que adjuntem. Realment, nosaltres hem executat aquest mateix codi amb tots per tal de veure quin era el que hem servia millor en la predicció per cada model. El que hem posat en comentaris, són els models que millors ens han predit per cadascun per si es volen replicar les sortides.
Hem inclòs dues funcions extres:
   - Mesures: ens permet obtenir l'error total, el true positive rate, el true negative rate i la f1 score (al final no l'hem utilitzat). Els seus paràmetres d'entrada són: la classe target amb els valors reals i les classes predites pel model.
   - Eficiència: calcula la proporció de trucades que resulten en "yes" respecte al total de trucades que faig. Els paràmetres d'entrada són el TPR i TNR (en el script trobareu que no la fem servir enlloc, ja que realment la feiem servir a l'hora de triar models i ens guardavem els resultats per separat).

Notar que cada fitxer .RData conté tant el conjunt de dades "bank" com els indexs de train ("learn").
