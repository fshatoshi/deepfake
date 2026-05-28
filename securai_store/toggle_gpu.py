#!/usr/bin/env python3
"""
toggle_gpu.py — T4 SecurAI
────────────────────────────────────────────────────────────────────────────────
Bascule le device d'inférence local (CPU ↔ CUDA/T4) et redémarre le serveur
Flask. La sélection est persistée dans `.env` (SECURAI_DEVICE).

Usage
─────
    python toggle_gpu.py              # bascule automatiquement CPU↔GPU
    python toggle_gpu.py --gpu        # force GPU (CUDA)
    python toggle_gpu.py --cpu        # force CPU
    python toggle_gpu.py --status     # affiche le device actuel + dispo CUDA
    python toggle_gpu.py --no-restart # change le .env SANS redémarrer le serveur

Environnement
─────────────
    SECURAI_DEVICE=cpu|cuda   (lu/écrit dans .env à côté de ce fichier)
    SECURAI_PORT=5000          (port Flask, optionnel)
    SECURAI_APP=app.py         (chemin relatif de l'app, optionnel)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent.resolve()
_ENV_FILE  = _HERE / ".env"
_PID_FILE  = _HERE / ".flask.pid"
_APP_FILE  = _HERE / os.environ.get("SECURAI_APP", "app.py")
_PORT      = int(os.environ.get("SECURAI_PORT", 5000))

VALID_DEVICES = ("cpu", "cuda")


# ── Helpers .env ──────────────────────────────────────────────────────────────

def _read_env() -> dict[str, str]:
    """Lit le fichier .env et retourne un dict clé=valeur."""
    env: dict[str, str] = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _write_env(env: dict[str, str]) -> None:
    """Réécrit le fichier .env en préservant les commentaires existants."""
    lines: list[str] = []

    # Conserver les commentaires d'en-tête si le fichier existe
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                lines.append(line)

    for k, v in env.items():
        lines.append(f"{k}={v}")

    _ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_current_device() -> str:
    """Retourne le device actuel depuis .env, ou 'cpu' par défaut."""
    return _read_env().get("SECURAI_DEVICE", "cpu").lower()


def set_device(device: str) -> None:
    """Écrit SECURAI_DEVICE dans .env."""
    if device not in VALID_DEVICES:
        raise ValueError(f"Device invalide : '{device}'. Valeurs : {VALID_DEVICES}")
    env = _read_env()
    env["SECURAI_DEVICE"] = device
    _write_env(env)


# ── Détection CUDA ────────────────────────────────────────────────────────────

def cuda_available() -> bool:
    """Retourne True si torch.cuda.is_available() — sans crasher si torch absent."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def cuda_info() -> str:
    """Retourne un résumé du GPU disponible (nom + mémoire)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "Aucun GPU CUDA détecté"
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return f"{name} ({mem} Mo)"
    except ImportError:
        return "torch non installé"


# ── Gestion du serveur Flask ──────────────────────────────────────────────────

def _read_pid() -> int | None:
    """Lit le PID du serveur depuis .flask.pid."""
    if _PID_FILE.exists():
        try:
            return int(_PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def _write_pid(pid: int) -> None:
    _PID_FILE.write_text(str(pid))


def _kill_server(pid: int) -> bool:
    """Arrête le processus Flask (SIGTERM sur POSIX, taskkill sur Windows)."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                check=True, capture_output=True,
            )
        else:
            os.kill(pid, signal.SIGTERM)
        print(f"  ✓ Serveur (PID {pid}) arrêté.")
        return True
    except (ProcessLookupError, subprocess.CalledProcessError, PermissionError) as e:
        print(f"  ⚠ Impossible d'arrêter PID {pid} : {e}")
        return False


def stop_server() -> None:
    """Arrête le serveur Flask si un PID est enregistré."""
    pid = _read_pid()
    if pid is None:
        print("  ℹ Aucun serveur Flask enregistré dans .flask.pid.")
        return
    _kill_server(pid)
    _PID_FILE.unlink(missing_ok=True)


def start_server(device: str) -> int:
    """Lance le serveur Flask avec le device choisi et enregistre le PID."""
    if not _APP_FILE.exists():
        print(f"  ✗ Fichier introuvable : {_APP_FILE}")
        sys.exit(1)

    env = os.environ.copy()
    env["SECURAI_DEVICE"] = device
    env["FLASK_APP"]      = str(_APP_FILE)

    # Lance Flask dans un sous-processus détaché
    kwargs: dict = dict(
        env=env,
        cwd=str(_HERE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [sys.executable, str(_APP_FILE)],
        **kwargs,
    )
    _write_pid(proc.pid)
    print(f"  ✓ Serveur Flask démarré (PID {proc.pid}) sur le port {_PORT}.")
    return proc.pid


# ── Commande principale ───────────────────────────────────────────────────────

def toggle(target: str | None = None, restart: bool = True) -> str:
    """
    Bascule le device et (optionnellement) redémarre le serveur.

    Parameters
    ----------
    target  : 'cpu' | 'cuda' | None (bascule automatique)
    restart : si True, arrête et relance le serveur Flask

    Returns
    -------
    Le nouveau device actif.
    """
    current = get_current_device()

    if target is None:
        # Bascule automatique
        target = "cuda" if current == "cpu" else "cpu"

    if target == "cuda" and not cuda_available():
        print("  ✗ CUDA demandé mais aucun GPU disponible — on reste sur CPU.")
        target = "cpu"

    if target == current:
        print(f"  ℹ Device déjà sur '{current}', aucun changement.")
        if not restart:
            return current

    print(f"\n  {current.upper()} → {target.upper()}")
    set_device(target)
    print(f"  ✓ .env mis à jour : SECURAI_DEVICE={target}")

    if restart:
        print("\n  ── Redémarrage du serveur ──────────────────────")
        stop_server()
        time.sleep(1.0)          # laisser le port se libérer
        start_server(target)

    return target


def status() -> None:
    """Affiche l'état actuel du device et du serveur."""
    current = get_current_device()
    pid     = _read_pid()
    avail   = cuda_info()

    print("\n┌─ SecurAI — état GPU ──────────────────────────────")
    print(f"│  Device actuel    : {current.upper()}")
    print(f"│  CUDA disponible  : {avail}")
    print(f"│  Serveur Flask    : PID {pid if pid else 'non enregistré'}")
    print(f"│  .env             : {_ENV_FILE}")
    print("└───────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SecurAI T4 — Bascule CPU/GPU et redémarre Flask.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--gpu",    action="store_true", help="Force CUDA (GPU T4)")
    group.add_argument("--cpu",    action="store_true", help="Force CPU")
    group.add_argument("--status", action="store_true", help="Affiche l'état actuel")
    p.add_argument(
        "--no-restart",
        action="store_true",
        help="Modifie .env sans redémarrer le serveur",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.status:
        status()
        return 0

    target  = "cuda" if args.gpu else ("cpu" if args.cpu else None)
    restart = not args.no_restart

    new_device = toggle(target=target, restart=restart)
    print(f"\n  Device actif : {new_device.upper()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())