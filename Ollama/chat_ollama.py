#!/usr/bin/env python3
"""
Chat Ollama — application de conversation avec un modèle local Ollama.

Aucune dépendance externe : uniquement la bibliothèque standard Python (3.8+).
Prérequis : Ollama installé et lancé (http://localhost:11434), avec au moins un modèle
(ex. : `ollama pull llama3.2`).

Lancement :  python chat_ollama.py
"""

import json
import queue
import threading
import tkinter as tk
import urllib.error
import urllib.request
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

OLLAMA_URL = "http://localhost:11434"
DEFAULT_SYSTEM = "Tu es un assistant utile. Réponds en français, de façon claire et concise."


# --------------------------------------------------------------------------
# Client Ollama (API HTTP)
# --------------------------------------------------------------------------
class OllamaClient:
    def __init__(self, base_url=OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def _request(self, path, payload=None, timeout=10, method=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method or ("POST" if data else "GET"),
        )
        return urllib.request.urlopen(req, timeout=timeout)

    def list_models_detailed(self):
        with self._request("/api/tags") as resp:
            data = json.load(resp)
        return sorted(data.get("models", []), key=lambda m: m["name"])

    def list_models(self):
        return [m["name"] for m in self.list_models_detailed()]

    def pull_stream(self, model, stop_event=None):
        """Générateur : renvoie les étapes du téléchargement (dicts status/total/completed)."""
        with self._request("/api/pull", {"model": model, "stream": True}, timeout=3600) as resp:
            for line in resp:
                if stop_event is not None and stop_event.is_set():
                    return
                line = line.strip()
                if not line:
                    continue
                step = json.loads(line)
                if "error" in step:
                    raise RuntimeError(step["error"])
                yield step

    def delete_model(self, model):
        with self._request("/api/delete", {"model": model}, method="DELETE"):
            pass

    def chat_stream(self, model, messages, options=None, stop_event=None):
        """Générateur : renvoie les morceaux de texte de la réponse au fil de l'eau."""
        payload = {"model": model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options
        with self._request("/api/chat", payload, timeout=600) as resp:
            for line in resp:
                if stop_event is not None and stop_event.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break


# --------------------------------------------------------------------------
# Interface graphique
# --------------------------------------------------------------------------
class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chat Ollama")
        self.geometry("900x680")
        self.minsize(600, 450)

        self.client = OllamaClient()
        self.history = []              # messages user/assistant
        self.events = queue.Queue()    # communication thread -> interface
        self.stop_event = threading.Event()
        self.busy = False

        self._build_ui()
        self.refresh_models()
        self.after(50, self._process_events)

    # ---- Construction de l'interface ----------------------------------
    def _build_ui(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Barre du haut
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Modèle :").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, width=30, state="readonly")
        self.model_combo.pack(side="left", padx=(4, 4))
        ttk.Button(top, text="⟳", width=3, command=self.refresh_models).pack(side="left")

        ttk.Label(top, text="Température :").pack(side="left", padx=(14, 4))
        self.temp_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(top, from_=0.0, to=2.0, increment=0.1, width=5,
                    textvariable=self.temp_var).pack(side="left")

        ttk.Button(top, text="Exporter", command=self.export_chat).pack(side="right")
        ttk.Button(top, text="Modèles…", command=self.open_model_manager).pack(side="right", padx=(6, 0))
        ttk.Button(top, text="Nouvelle conversation", command=self.new_chat).pack(side="right", padx=6)

        # Prompt système
        sys_frame = ttk.Frame(self, padding=(10, 0))
        sys_frame.pack(fill="x")
        ttk.Label(sys_frame, text="Instructions système :").pack(side="left")
        self.system_var = tk.StringVar(value=DEFAULT_SYSTEM)
        ttk.Entry(sys_frame, textvariable=self.system_var).pack(side="left", fill="x", expand=True, padx=4)

        # Zone de conversation
        chat_frame = ttk.Frame(self, padding=10)
        chat_frame.pack(fill="both", expand=True)
        self.chat = tk.Text(chat_frame, wrap="word", state="disabled", font=("Segoe UI", 11),
                            padx=10, pady=10, borderwidth=0, background="#fafafa")
        scroll = ttk.Scrollbar(chat_frame, command=self.chat.yview)
        self.chat.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)

        self.chat.tag_configure("user_name", foreground="#1a5fb4", font=("Segoe UI", 11, "bold"))
        self.chat.tag_configure("bot_name", foreground="#26a269", font=("Segoe UI", 11, "bold"))
        self.chat.tag_configure("user", lmargin1=10, lmargin2=10)
        self.chat.tag_configure("bot", lmargin1=10, lmargin2=10)
        self.chat.tag_configure("info", foreground="#888888", font=("Segoe UI", 9, "italic"))
        self.chat.tag_configure("error", foreground="#c01c28")

        # Zone de saisie
        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        self.input = tk.Text(bottom, height=3, wrap="word", font=("Segoe UI", 11))
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>", self._on_enter)
        self.input.focus_set()

        btns = ttk.Frame(bottom)
        btns.pack(side="left", padx=(8, 0), fill="y")
        self.send_btn = ttk.Button(btns, text="Envoyer", command=self.send)
        self.send_btn.pack(fill="x")
        self.stop_btn = ttk.Button(btns, text="Arrêter", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=(4, 0))

        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 2),
                  relief="sunken").pack(fill="x", side="bottom")

        ttk.Label(self, text="Entrée : envoyer · Maj+Entrée : nouvelle ligne",
                  foreground="#888888", padding=(10, 0)).pack(anchor="w", side="bottom")

    # ---- Affichage ------------------------------------------------------
    def _append(self, text, tag=None):
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    # ---- Modèles --------------------------------------------------------
    def refresh_models(self):
        try:
            models = self.client.list_models()
        except (urllib.error.URLError, OSError):
            self.status_var.set("Ollama injoignable sur " + OLLAMA_URL)
            self._append("⚠ Impossible de joindre Ollama. Vérifiez qu'il est lancé "
                         "(icône dans la barre des tâches ou `ollama serve`), puis cliquez sur ⟳.\n\n",
                         "error")
            return
        self.model_combo["values"] = models
        if not models:
            self.status_var.set("Aucun modèle installé")
            self._append("Aucun modèle trouvé. Cliquez sur « Modèles… » pour en installer un "
                         "(ex. : llama3.2).\n\n", "info")
            return
        if self.model_var.get() not in models:
            self.model_var.set(models[0])
        self.status_var.set(f"{len(models)} modèle(s) disponible(s)")

    # ---- Envoi / réception ---------------------------------------------
    def _on_enter(self, event):
        if event.state & 0x0001:   # Maj enfoncée : laisser passer le saut de ligne
            return None
        self.send()
        return "break"

    def send(self):
        if self.busy:
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        model = self.model_var.get()
        if not model:
            messagebox.showwarning("Chat Ollama", "Choisissez d'abord un modèle.")
            return

        self.input.delete("1.0", "end")
        self.history.append({"role": "user", "content": text})
        self._append("Vous\n", "user_name")
        self._append(text + "\n\n", "user")
        self._append(f"{model}\n", "bot_name")

        messages = []
        system = self.system_var.get().strip()
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self.history)

        try:
            temperature = float(self.temp_var.get())
        except (tk.TclError, ValueError):
            temperature = 0.7

        self._set_busy(True)
        self.stop_event.clear()
        threading.Thread(target=self._worker, args=(model, messages, temperature),
                         daemon=True).start()

    def _worker(self, model, messages, temperature):
        reply = []
        try:
            for piece in self.client.chat_stream(model, messages, {"temperature": temperature},
                                                 self.stop_event):
                reply.append(piece)
                self.events.put(("chunk", piece))
            self.events.put(("done", "".join(reply)))
        except urllib.error.URLError as e:
            self.events.put(("error", f"Connexion à Ollama impossible : {e.reason}", "".join(reply)))
        except Exception as e:  # noqa: BLE001
            self.events.put(("error", str(e), "".join(reply)))

    def _process_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "chunk":
                    self._append(ev[1], "bot")
                elif kind == "done":
                    self._finish(ev[1], stopped=self.stop_event.is_set())
                elif kind == "error":
                    self._append(f"\n⚠ {ev[1]}", "error")
                    self._finish(ev[2])
        except queue.Empty:
            pass
        self.after(30, self._process_events)

    def _finish(self, reply, stopped=False):
        if stopped:
            self._append("\n[réponse interrompue]", "info")
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        else:
            self.history.pop()  # pas de réponse : on retire la question de l'historique
        self._append("\n\n")
        self._set_busy(False)

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Arrêt en cours…")

    def _set_busy(self, busy):
        self.busy = busy
        self.send_btn.configure(state="disabled" if busy else "normal")
        self.stop_btn.configure(state="normal" if busy else "disabled")
        self.model_combo.configure(state="disabled" if busy else "readonly")
        self.status_var.set("Le modèle répond…" if busy else "Prêt")
        if not busy:
            self.input.focus_set()

    def open_model_manager(self):
        if getattr(self, "_manager", None) and self._manager.winfo_exists():
            self._manager.lift()
            return
        self._manager = ModelManager(self, self.client, on_change=self.refresh_models)

    # ---- Conversation ---------------------------------------------------
    def new_chat(self):
        if self.busy:
            self.stop()
        self.history.clear()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.status_var.set("Nouvelle conversation")

    def export_chat(self):
        if not self.history:
            messagebox.showinfo("Chat Ollama", "La conversation est vide.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            initialfile=f"conversation_{datetime.now():%Y-%m-%d_%H-%M}.md",
            filetypes=[("Markdown", "*.md"), ("Texte", "*.txt"), ("JSON", "*.json")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                json.dump({"model": self.model_var.get(), "messages": self.history},
                          f, ensure_ascii=False, indent=2)
            else:
                f.write(f"# Conversation avec {self.model_var.get()}\n\n")
                for m in self.history:
                    who = "Vous" if m["role"] == "user" else "Assistant"
                    f.write(f"**{who}** :\n\n{m['content']}\n\n---\n\n")
        self.status_var.set(f"Exporté : {path}")


# --------------------------------------------------------------------------
# Gestionnaire de modèles (installer / supprimer)
# --------------------------------------------------------------------------
SUGGESTED_MODELS = [
    "llama3.2", "llama3.2:1b", "llama3.1", "mistral", "gemma3", "gemma3:1b",
    "qwen3", "qwen2.5-coder", "deepseek-r1", "phi4", "phi4-mini",
]


def human_size(n):
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or unit == "To":
            return f"{n:.1f} {unit}" if unit != "o" else f"{n} o"
        n /= 1024


class ModelManager(tk.Toplevel):
    def __init__(self, master, client, on_change=None):
        super().__init__(master)
        self.title("Gestion des modèles")
        self.geometry("620x480")
        self.minsize(500, 380)
        self.transient(master)

        self.client = client
        self.on_change = on_change
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.pulling = False

        self._build_ui()
        self.refresh()
        self.after(50, self._process_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Installation
        box = ttk.LabelFrame(self, text="Installer un modèle", padding=10)
        box.pack(fill="x", padx=10, pady=(10, 5))

        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="Nom :").pack(side="left")
        self.name_var = tk.StringVar()
        self.name_combo = ttk.Combobox(row, textvariable=self.name_var, values=SUGGESTED_MODELS)
        self.name_combo.pack(side="left", fill="x", expand=True, padx=6)
        self.name_combo.bind("<Return>", lambda e: self.install())
        self.install_btn = ttk.Button(row, text="Installer", command=self.install)
        self.install_btn.pack(side="left")
        self.cancel_btn = ttk.Button(row, text="Annuler", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(4, 0))

        link = ttk.Label(box, text="Catalogue complet : ollama.com/library  (ex. : mistral, qwen3:8b, gemma3:4b)",
                         foreground="#1a5fb4", cursor="hand2")
        link.pack(anchor="w", pady=(6, 0))
        link.bind("<Button-1>", lambda e: __import__("webbrowser").open("https://ollama.com/library"))

        self.progress = ttk.Progressbar(box, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(8, 2))
        self.pstatus = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.pstatus, foreground="#555555").pack(anchor="w")

        # Liste des modèles installés
        lst = ttk.LabelFrame(self, text="Modèles installés", padding=10)
        lst.pack(fill="both", expand=True, padx=10, pady=5)
        cols = ("name", "size", "modified")
        self.tree = ttk.Treeview(lst, columns=cols, show="headings", selectmode="browse")
        for col, title, width in (("name", "Nom", 260), ("size", "Taille", 100), ("modified", "Modifié", 150)):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w" if col == "name" else "center")
        sb = ttk.Scrollbar(lst, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Actualiser", command=self.refresh).pack(side="left")
        self.delete_btn = ttk.Button(bottom, text="Supprimer", command=self.delete)
        self.delete_btn.pack(side="left", padx=6)
        self.update_btn = ttk.Button(bottom, text="Mettre à jour", command=self.update_selected)
        self.update_btn.pack(side="left")
        ttk.Button(bottom, text="Fermer", command=self._on_close).pack(side="right")

    # ---- Liste ----------------------------------------------------------
    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        try:
            models = self.client.list_models_detailed()
        except (urllib.error.URLError, OSError):
            self.pstatus.set("⚠ Ollama injoignable.")
            return
        for m in models:
            modified = (m.get("modified_at") or "")[:16].replace("T", " ")
            self.tree.insert("", "end", iid=m["name"],
                             values=(m["name"], human_size(m.get("size", 0)), modified))

    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Modèles", "Sélectionnez un modèle dans la liste.", parent=self)
            return None
        return sel[0]

    def delete(self):
        name = self._selected()
        if not name:
            return
        if not messagebox.askyesno("Supprimer", f"Supprimer le modèle « {name} » ?", parent=self):
            return
        try:
            self.client.delete_model(name)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Erreur", f"Suppression impossible : {e}", parent=self)
            return
        self.pstatus.set(f"« {name} » supprimé.")
        self.refresh()
        if self.on_change:
            self.on_change()

    def update_selected(self):
        name = self._selected()
        if name:
            self.name_var.set(name)
            self.install()

    # ---- Installation ---------------------------------------------------
    def install(self):
        if self.pulling:
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showinfo("Modèles", "Indiquez le nom du modèle à installer.", parent=self)
            return
        self.pulling = True
        self.stop_event.clear()
        self.install_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.pstatus.set(f"Téléchargement de « {name} »…")
        threading.Thread(target=self._worker, args=(name,), daemon=True).start()

    def _worker(self, name):
        try:
            for step in self.client.pull_stream(name, self.stop_event):
                self.events.put(("step", step))
            if self.stop_event.is_set():
                self.events.put(("cancelled", name))
            else:
                self.events.put(("done", name))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(msg).get("error", msg)
            except ValueError:
                pass
            self.events.put(("error", msg))
        except urllib.error.URLError as e:
            self.events.put(("error", f"Ollama injoignable : {e.reason}"))
        except Exception as e:  # noqa: BLE001
            self.events.put(("error", str(e)))

    def cancel(self):
        self.stop_event.set()
        self.pstatus.set("Annulation…")

    def _process_events(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "step":
                    self._show_step(data)
                else:
                    self._end_pull(kind, data)
        except queue.Empty:
            pass
        self.after(100, self._process_events)

    def _show_step(self, step):
        status = step.get("status", "")
        total, done = step.get("total"), step.get("completed")
        if total:
            if str(self.progress.cget("mode")) != "determinate":
                self.progress.stop()
                self.progress.configure(mode="determinate")
            pct = 100 * (done or 0) / total
            self.progress["value"] = pct
            label = "Téléchargement" if status.startswith("pulling") else status
            self.pstatus.set(f"{label} : {human_size(done or 0)} / {human_size(total)} ({pct:.0f} %)")
        else:
            translations = {
                "pulling manifest": "Récupération du manifeste…",
                "verifying sha256 digest": "Vérification de l'intégrité…",
                "writing manifest": "Écriture du manifeste…",
                "removing any unused layers": "Nettoyage…",
                "success": "Terminé",
            }
            self.pstatus.set(translations.get(status, status))

    def _end_pull(self, kind, data):
        self.pulling = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.install_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if kind == "done":
            self.progress["value"] = 100
            self.pstatus.set(f"✔ « {data} » installé.")
            self.refresh()
            if self.on_change:
                self.on_change()
        elif kind == "cancelled":
            self.progress["value"] = 0
            self.pstatus.set("Téléchargement annulé (il reprendra là où il s'est arrêté).")
        else:
            self.progress["value"] = 0
            if "not found" in data.lower() or "file does not exist" in data.lower():
                data = "modèle introuvable. Vérifiez le nom sur ollama.com/library."
            self.pstatus.set(f"⚠ Erreur : {data}")

    def _on_close(self):
        if self.pulling and not messagebox.askyesno(
                "Modèles", "Un téléchargement est en cours. L'annuler et fermer ?", parent=self):
            return
        self.stop_event.set()
        self.destroy()


if __name__ == "__main__":
    ChatApp().mainloop()
