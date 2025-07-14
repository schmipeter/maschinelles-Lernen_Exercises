## Ordnerübersicht
│ catData.mat / dogData.mat               ← Rohdaten  
│ catData_w.mat / dogData_w.mat           ← Wavelet-Features  
│ make_wavelet_mat.py                     ← erzeugt *_w.mat aus den Rohdaten  
│ make_cm_png.py                          ← schreibt confusion_matrix.png  
│ live\                                   ← dvclive-Ergebnisse  
│ ├─ metrics.json                         ← Train-/Test-Scores  
│ ├─ artifacts\model.skops                ← gespeichertes Modell  
│ └─ plots\sklearn\confusion_matrix.json  ← Rohdaten der Matrix  
│ cm.html                                 ← HTML-Ansicht der Matrix (per dvc plots)  
│ src\MECH-M-DUAL-2-MLB-DATA\…            ← Code + ETL + Training  
│ requirements.txt                        ← benötigte Pakete  
│ .dvc\ + .dvc\config(.local)             ← DVC + WebDAV-Remote  

## Install & Run (PowerShell)
git clone <repo-url>
cd <repo-ordner>

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt     # inkl. dvc[all]

dvc pull                                      # WebDAV-Login nötig
# kein Login?  python make_wavelet_mat.py     # erzeugt Wavelet-Files lokal

$env:PYTHONPATH = "$PWD\src\MECH-M-DUAL-2-MLB-DATA"
python src\MECH-M-DUAL-2-MLB-DATA\train.py

## Ergebnisse ansehen

Scores     ->   live\metrics.json                             
Modell     ->   live\artifacts\model.skops                    
Konfusionsmatrix (PNG)     ->    python make_cm_png.py  # PNG im Projektroot doppelklicken
HTML-Matrix    cm.html        ->     dvc plots show live/plots/sklearn/confusion_matrix.json -o cm.html<br>start cm.html

## DVC-Remote (Sakai WebDAV)
dvc remote modify --local myremote user <user>
dvc remote modify --local myremote password <Passwort>
dvc remote modify myremote jobs 4
dvc push        # hochladen
dvc pull        # herunterladen

**Hinweis**  
Ohne WebDAV-Zugang genügt `python make_wavelet_mat.py`; die Wavelet-Dateien werden lokal erzeugt, `dvc pull` ist dann nicht erforderlich.