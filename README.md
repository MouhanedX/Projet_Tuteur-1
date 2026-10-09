# GTSRB-CARLA Traffic-Sign Recognition

Projet tuteuré de reconnaissance de panneaux routiers avec un CNN PyTorch
entraîné sur le **German Traffic Sign Recognition Benchmark (GTSRB)** et
évalué sur des images capturées dans **CARLA 0.9.16**.

Le dépôt contient uniquement le code d'exécution, les poids du modèle et cette
documentation. Le dataset GTSRB, CARLA et les images générées sont volontairement
exclus du dépôt.

## Objectif

Le projet suit ce flux :

```text
Dataset GTSRB
    -> entraînement du CNN 43 classes
    -> gtsrb_cnn_best.pt
    -> capture de scènes CARLA avec véhicule, piétons et panneaux
    -> raw_frames/
    -> insertion d'images du test set GTSRB sur les panneaux CARLA
    -> treated_frames/
    -> classification manuelle avec le CNN
```

Le traitement des panneaux est réalisé en post-traitement OpenCV. Les panneaux
CARLA sont des supports géométriques : une image réelle du test set GTSRB est
projetée sur leur face par transformation perspective. Cela évite de modifier
les assets Unreal Engine de CARLA.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `train_gtsrb.py` | Entraîne le CNN sur les 43 classes et évalue le test set officiel |
| `capture_carla_frames.py` | Lance CARLA, crée les acteurs, conduit le véhicule et capture les frames |
| `process_carla_frames.py` | Ajoute les crops GTSRB sur les panneaux visibles |
| `classify_image.py` | Classifie manuellement une image avec les poids entraînés |
| `gtsrb_cnn_best.pt` | Meilleur snapshot de poids disponible |

Les éléments suivants ne sont pas versionnés :

- `dataset_GTSRB/` ;
- `raw_frames/` ;
- `treated_frames/` ;
- `capture_manifest.jsonl` ;
- `test_preds.npy` ;
- caches Python et sorties temporaires.

## Prérequis

### Logiciels

- Windows 10/11 ;
- Python 3.11 recommandé ;
- CARLA 0.9.16 installé séparément, par exemple :
  `C:\Users\<user>\Desktop\CARLA_0.9.16` ;
- une carte CARLA contenant une route et la navigation piétonne si des
  piétons doivent être utilisés.

### Python

Installer les dépendances dans l'environnement Python utilisé :

```powershell
python -m pip install numpy opencv-python torch carla
```

Le package CARLA peut aussi être installé depuis le fichier `.whl` fourni par
l'installation CARLA si `pip install carla` n'est pas disponible.

### Dataset

Placer le dataset GTSRB dans le dossier du projet sous :

```text
dataset_GTSRB\
    Train.csv
    Test.csv
    train\0\*.png
    train\1\*.png
    ...
    train\42\*.png
    Test\*.png
```

Le script d'entraînement attend les 43 répertoires `train/0` à `train/42` et
le fichier `Test.csv` avec les colonnes `Path`, `Roi.X1`, `Roi.Y1`, `Roi.X2`,
`Roi.Y2` et `ClassId`.

## 1. Entraîner le modèle

Depuis PowerShell :

```powershell
cd "C:\Users\<user>\Desktop\Projet_Tuteur-1"
python .\train_gtsrb.py
```

`train_gtsrb.py` :

1. charge les images d'entraînement des 43 classes ;
2. redimensionne chaque image en `48 x 48` ;
3. crée une validation stratifiée de 5 % ;
4. applique des augmentations aléatoires pendant l'entraînement :
   rotation jusqu'à ±12°, échelle ±10 %, translations ±8 %, contraste et
   luminosité ;
5. entraîne pendant 30 epochs avec Adam ;
6. utilise un scheduler avec baisse du learning rate aux epochs 20 et 26 ;
7. sauvegarde uniquement le meilleur score de validation dans
   `gtsrb_cnn_best.pt` ;
8. calcule l'accuracy du test set officiel et écrit `test_preds.npy` localement.

Paramètres principaux :

```text
classes       : 43
entrée        : 48 x 48 x 3
batch         : 128
epochs        : 30
validation    : 5 %
learning rate : 1e-3
seed          : 42
```

Une nouvelle exécution supprime l'ancien checkpoint et recommence un
entraînement frais.

## 2. Lancer CARLA et capturer les frames

Dans une première fenêtre PowerShell, lancer CARLA :

```powershell
& "C:\Users\<user>\Desktop\CARLA_0.9.16\CarlaUE4.exe"
```

Attendre que CARLA soit complètement démarré. Dans une deuxième fenêtre :

```powershell
cd "C:\Users\<user>\Desktop\Projet_Tuteur-1"
python .\capture_carla_frames.py
```

La connexion utilise :

```text
host    : 127.0.0.1
port    : 2000
timeout : 120 secondes
```

### Acteurs CARLA

Le script crée :

- un `vehicle.tesla.model3` avec le rôle `hero` ;
- une caméra `sensor.camera.rgb` attachée au véhicule ;
- jusqu'à 12 piétons avec `controller.ai.walker` lorsque des positions de
  navigation sont disponibles ;
- des panneaux `static.prop.streetsign`.

Le Spectator CARLA suit le véhicule depuis une position arrière et surélevée
afin de rendre la scène visible dans la fenêtre CARLA.

### Route et conduite

- carte courante CARLA ;
- point de départ : premier spawn point de la carte ;
- route calculée par waypoints espacés de 5 m ;
- environ 320 waypoints ;
- conduite par contrôleur pure-pursuit simplifié ;
- vitesse cible proche de 20 km/h ;
- mode synchrone ;
- `fixed_delta_seconds = 0.05` ;
- capture enregistrée tous les 3 ticks (`CAPTURE_EVERY_TICKS = 3`).

La simulation avance donc à 20 ticks/s, mais l'écriture des images est limitée à
environ 6,7 captures/s pour éviter un volume excessif.

### Scénarios de panneaux

Les 43 classes ne sont pas placées simultanément. Elles sont réparties en trois
scénarios :

```text
Scénario 1 : classes 0..13  (14 panneaux)
Scénario 2 : classes 14..27 (14 panneaux)
Scénario 3 : classes 28..42 (15 panneaux)
```

Pour chaque scénario, le véhicule parcourt la route sous trois météos :

```text
ClearNoon
CloudySunset
SoftRainNoon
```

Les panneaux sont placés sur le bord de la route, hors jonction lorsque
possible, avec un contrôle du sol par raycast. Les images sont gardées lorsque
le panneau est :

```text
distance : 8 à 45 m
largeur projetée minimale : 14 pixels
entièrement dans l'image
devant la caméra
```

### Sorties de capture

Le script crée localement :

```text
raw_frames\
capture_manifest.jsonl
```

Une entrée du manifeste contient le nom de la frame, le scénario, la météo et,
pour chaque panneau visible, sa classe, sa distance et les quatre coins projetés
dans l'image :

```json
{
  "file": "scenario1_ClearNoon_000000.png",
  "scenario": 1,
  "weather": "ClearNoon",
  "signs": [
    {
      "class": 0,
      "distance_m": 40.77,
      "quad": [[...], [...], [...], [...]]
    }
  ]
}
```

Une frame sans panneau valide n'est pas enregistrée.

## 3. Traiter les frames CARLA

Après la capture :

```powershell
cd "C:\Users\<user>\Desktop\Projet_Tuteur-1"
python .\process_carla_frames.py
```

Le script :

1. lit chaque ligne de `capture_manifest.jsonl` ;
2. charge la frame correspondante depuis `raw_frames/` ;
3. charge des crops ROI du **test set GTSRB** ;
4. vérifie que les 43 classes sont disponibles ;
5. sélectionne une image GTSRB de la classe du panneau ;
6. ajuste la luminosité et le gamma à la région CARLA ;
7. applique une homographie sur les quatre coins du panneau ;
8. adoucit les contours avec un flou gaussien ;
9. écrit la frame complète dans `treated_frames/`.

Les frames brutes ne sont jamais modifiées.

```text
raw_frames/frame.png
        -> traitement
treated_frames/frame.png
```

## 4. Classifier une image

Pour une frame traitée issue du manifeste :

```powershell
python .\classify_image.py ".\treated_frames\scenario1_ClearNoon_000000.png"
```

Le script charge automatiquement :

```text
gtsrb_cnn_best.pt
```

Il reconstruit la même architecture `SignCNN`, passe en mode évaluation et
affiche la classe prédite ainsi que la confiance :

```text
sign 0: class=14 confidence=96.42%
```

Si l'image correspond à une entrée de `capture_manifest.jsonl`, le script
utilise les quadrilatères du manifeste pour classifier chaque panneau visible.
Pour une image indépendante ou un crop, il classe directement l'image entière.

Pour choisir explicitement les poids :

```powershell
python .\classify_image.py ".\image.png" `
  --weights ".\gtsrb_cnn_best.pt"
```

Les classes retournées sont les identifiants GTSRB `0` à `42`.

## Ordre complet

```powershell
# Entraînement, sans CARLA
cd "C:\Users\<user>\Desktop\Projet_Tuteur-1"
python .\train_gtsrb.py

# Nouvelle fenêtre : serveur CARLA
& "C:\Users\<user>\Desktop\CARLA_0.9.16\CarlaUE4.exe"

# Après le démarrage de CARLA
cd "C:\Users\<user>\Desktop\Projet_Tuteur-1"
python .\capture_carla_frames.py

# Après la capture
python .\process_carla_frames.py
python .\classify_image.py ".\treated_frames\scenario1_ClearNoon_000000.png"
```

## Nettoyage local des sorties

Ces commandes ne suppriment pas les scripts ni les poids :

```powershell
Remove-Item ".\raw_frames" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ".\treated_frames" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ".\capture_manifest.jsonl" -Force -ErrorAction SilentlyContinue
Remove-Item ".\test_preds.npy" -Force -ErrorAction SilentlyContinue
```

## Limites connues

- CARLA doit être lancé séparément.
- La caméra du véhicule et le Spectator sont deux vues différentes.
- Les panneaux sont ajoutés visuellement après capture et ne sont pas des
  textures Unreal natives.
- Les piétons dépendent des positions de navigation disponibles sur la carte.
- L'accuracy GTSRB propre ne garantit pas la même performance sous pluie, faible
  résolution, perspective ou occultation.
- Les poids `.pt` sont binaires et peuvent être volumineux ; Git LFS est
  recommandé si la plateforme impose une limite de taille.
