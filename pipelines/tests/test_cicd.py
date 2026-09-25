"""Garde-fous de la chaîne d'intégration et de livraison continues.

Les flux GitHub Actions ne s'exécutent pas ici : ce qui est vérifié, ce sont les propriétés
qui se perdent en silence à la première modification pressée — un secret qui cesse d'être
ignoré, une permission élargie « le temps d'un essai », un déploiement qui ne sait plus
revenir en arrière. Ce sont exactement les régressions qu'aucun relecteur ne voit passer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
FLUX = ROOT / ".github/workflows"


def _flux(nom: str) -> dict:
    return yaml.safe_load((FLUX / nom).read_text(encoding="utf-8"))


def test_the_workflows_read_the_repository_and_nothing_else():
    """Moindre privilège : un jeton de flux en écriture est une porte ouverte inutile."""
    for fichier in sorted(FLUX.glob("*.yml")):
        flux = yaml.safe_load(fichier.read_text(encoding="utf-8"))
        assert flux["permissions"] == {"contents": "read"}, fichier.name
        for job in flux["jobs"].values():
            assert "self-hosted" in job["runs-on"] and "ondine" in job["runs-on"]
            assert job.get("timeout-minutes"), "un flux sans limite de temps bloque la machine"


def test_a_deployment_is_never_interrupted_midway():
    """Une bascule annulée au milieu laisse la plateforme dans un état indéterminé."""
    livraison = _flux("cd.yml")
    assert livraison["concurrency"]["cancel-in-progress"] is False
    # L'intégration, elle, peut être annulée : seul le dernier commit d'une branche compte.
    assert _flux("ci.yml")["concurrency"]["cancel-in-progress"] is True


def test_the_deployment_can_roll_back_and_is_valid_shell():
    """Un déploiement qui ne sait pas revenir en arrière est un pari, pas un déploiement."""
    script = ROOT / "scripts/deployer.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)  # noqa: S603, S607
    texte = script.read_text(encoding="utf-8")
    assert "marquer local precedent" in texte, "la version en service doit être conservée"
    assert "marquer precedent local" in texte, "le retour arrière doit remettre ces images"
    assert "set -euo pipefail" in texte, "une étape en échec ne doit pas être ignorée"
    # Les sondes portent sur la mise en service, pas sur la simple vivacité.
    assert "/pret" in texte and "exit 1" in texte
    assert _flux("cd.yml")["jobs"]["deployer"]["steps"], "le flux doit appeler le script"
    assert "scripts/deployer.sh" in (FLUX / "cd.yml").read_text(encoding="utf-8")


def test_secrets_never_enter_the_repository():
    """Un secret commis ne s'efface pas d'un `git revert` : il ne doit jamais entrer."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in ignore
    assert not (ROOT / ".env").exists() or ".env" in ignore
    # Les secrets viennent de la machine de production, le temps du déploiement seulement,
    # et sont effacés de l'espace de travail quelle que soit l'issue.
    livraison = (FLUX / "cd.yml").read_text(encoding="utf-8")
    assert "vars.ONDINE_INSTALL" in livraison
    assert "Copy-Item $source '.env'" in livraison
    assert "Remove-Item '.env'" in livraison
    assert "if: always()" in livraison, "un secret laissé sur disque après un échec"
    assert "cat .env" not in livraison and "Get-Content" not in livraison


def test_the_workflows_use_a_shell_the_production_machine_actually_has():
    """Deux interpréteurs manquent sur la machine de production, et l'échec est muet.

    `bash` y désigne celui de WSL, qui ne sait pas lire les chemins Windows que le runner
    lui passe ; `pwsh` (PowerShell 7) n'y est pas installé du tout. Seul le PowerShell livré
    avec Windows est garanti — et le bash de Git, appelé par son chemin.
    """
    for fichier in sorted(FLUX.glob("*.yml")):
        texte = fichier.read_text(encoding="utf-8")
        assert "shell: bash" not in texte and "shell: pwsh" not in texte, fichier.name
        flux = yaml.safe_load(texte)
        assert flux["defaults"]["run"]["shell"] == "powershell", fichier.name
    assert "Git\\bin\\bash.exe" in (FLUX / "cd.yml").read_text(encoding="utf-8")


def test_a_warning_on_standard_error_does_not_kill_a_step():
    """GitHub exécute les étapes PowerShell avec `$ErrorActionPreference = 'Stop'`.

    La moindre ligne écrite sur la sortie d'erreur par un programme externe interrompt alors
    l'étape — et `docker compose` y écrit son avancement et ses avertissements. L'étape échoue
    en une fraction de seconde, sans que rien n'ait réellement échoué. Chaque bloc rétablit
    donc `Continue` et vérifie explicitement le code de sortie.
    """
    for fichier in sorted(FLUX.glob("*.yml")):
        flux = yaml.safe_load(fichier.read_text(encoding="utf-8"))
        for job in flux["jobs"].values():
            for etape in job["steps"]:
                script = etape.get("run")
                if script is None:
                    continue
                assert "$ErrorActionPreference = 'Continue'" in script, (
                    f"{fichier.name} / {etape.get('name', '?')}"
                )


def test_the_tests_never_run_against_the_production_database():
    """Les tests créent et suppriment des bases entières : jamais sur celle des données."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    tests = compose["services"]["tests"]
    assert "postgres-ci" in tests["environment"]["TEST_PG_ADMIN_DSN"]
    assert "postgres-ci" in tests["depends_on"]
    # Base jetable : rien n'en survit, et elle n'occupe aucun volume.
    assert compose["services"]["postgres-ci"]["tmpfs"]
    assert "ci" in compose["services"]["postgres-ci"]["profiles"]
