# Ollama pour Windows : installation + application de chat

## Fichiers

| Fichier | Rôle |
|---|---|
| `Installer-Ollama.bat` | Double-cliquer pour lancer l'installation (contourne la politique d'exécution PowerShell) |
| `Install-Ollama.ps1` | Télécharge `OllamaSetup.exe`, vérifie sa signature, installe en silencieux, démarre le serveur |
| `chat_ollama.py` | Application de conversation (Tkinter, sans dépendance) avec gestionnaire de modèles |

## 1. Installer Ollama

Double-cliquez sur `Installer-Ollama.bat`, ou dans PowerShell :

```powershell
.\Install-Ollama.ps1                                   # installation standard
.\Install-Ollama.ps1 -Model llama3.2                   # + télécharge un modèle
.\Install-Ollama.ps1 -InstallDir "D:\Ollama" -ModelsDir "D:\OllamaModels"
.\Install-Ollama.ps1 -Force                            # réinstaller / mettre à jour
```

Pas besoin de droits administrateur. Ouvrez un nouveau terminal ensuite pour utiliser `ollama`.

## 2. Lancer l'application de chat

Prérequis : Python 3.8+ (python.org, Tkinter inclus) et Ollama démarré.

```powershell
python chat_ollama.py
```

- Choix du modèle, réponses en direct, bouton Arrêter
- Instructions système et température réglables
- Export de la conversation (Markdown, texte, JSON)
- **Modèles…** : installer un modèle avec barre de progression, mettre à jour, supprimer

Pour un PC sans carte graphique puissante, commencez par un petit modèle : `llama3.2:1b` ou `gemma3:1b`.
