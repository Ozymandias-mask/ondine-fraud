# Chaîne d'intégration et de livraison continues

Deux flux GitHub Actions, exécutés par un **runner auto-hébergé** installé sur la machine qui
porte la plateforme. Ce choix n'est pas un pis-aller : la solution se déploie en conteneurs
sur cette machine, et un déploiement qu'on ne peut pas vérifier sur la vraie plateforme n'est
pas un déploiement. Le runner voit les mêmes images, les mêmes volumes et le même `.env` que
la production.

| Flux | Déclencheur | Ce qu'il fait | Ce qu'il ne fait pas |
|---|---|---|---|
| `ci.yml` | tout `push`, toute proposition de fusion | style, erreurs probables, tests automatisés, configuration de la plateforme, règles d'alerte, vulnérabilités des images, recherche de secrets | ne déploie rien |
| `cd.yml` | `push` sur `main`, étiquette `v*` | construit, applique les migrations, bascule, vérifie, **revient en arrière si la mise en service échoue** | ne saute aucun test : `ci.yml` tourne sur le même commit |

## Ce qui est vérifié avant `main`

- **Style et erreurs probables** — `ruff check` et `ruff format --check`, avec les règles de
  sécurité (Bandit) et de docstrings activées dans `pyproject.toml`.
- **Tests automatisés** — dans l'**image des traitements**, donc sur exactement les versions
  de bibliothèques qui tournent en production. Un environnement Python reconstruit sur la
  machine d'intégration ne le garantirait pas, et c'est précisément là que se cachent les
  écarts. Les tests construisent de vraies données avec le simulateur, les font passer par
  les vrais traitements jusqu'à la couche gold, entraînent un modèle et vérifient l'API.
- **Base jetable** — `postgres-ci`, en mémoire (`tmpfs`) : les tests créent et suppriment des
  bases entières, ils ne doivent jamais le faire sur l'instance qui porte les données.
- **Configuration** — `docker compose config` et `promtool check rules` : une erreur de
  syntaxe dans les règles d'alerte ne se découvre pas le jour où l'alerte devait se déclencher.
- **Vulnérabilités** — Trivy sur l'image de service. Seules les vulnérabilités **critiques et
  corrigeables** bloquent : au-delà, la chaîne échouerait en permanence sur des alertes que
  personne ne peut corriger, et plus personne ne la regarderait.
- **Secrets commis** — gitleaks sur l'historique complet : un secret publié ne s'efface pas
  d'un `git revert`.

## Ce que fait un déploiement

`scripts/deployer.sh` — la logique est dans un script, pas dans le YAML : on doit pouvoir la
lire, la relancer et la déboguer sans passer par l'orchestrateur.

1. les images en service sont **marquées `:precedent`** ;
2. construction, avec l'empreinte du commit inscrite dans l'image (`ONDINE_BUILD`) ;
3. migrations de base de données (`db-migrate`, idempotentes et numérotées) ;
4. bascule (`docker compose up -d`) ;
5. sondes de mise en service : API de scoring (`/pret`, qui exige modèle **et** magasin en
   ligne), MLflow, Prometheus. Si l'une ne répond pas au bout de 100 s, **les images
   `:precedent` sont remises et la plateforme redémarre dessus**, puis le flux échoue.

Un déploiement qui ne sait pas revenir en arrière n'est pas un déploiement, c'est un pari.

## Versions

- Chaque image est étiquetée avec l'empreinte courte du commit (`ondine/api:3f2a91c`) en plus
  de `:local`. « latest » ne dit pas quel code tourne.
- L'API renvoie cette empreinte sur `/sante` (`build`) : savoir *quel* code répond est la
  première question d'une astreinte, et la seule à laquelle les journaux ne répondent pas
  toujours.
- Les livraisons notables sont étiquetées `vX.Y.Z` dans Git ; l'étiquette déclenche le même
  flux de déploiement.
- Les modèles ont leur propre versionnement, dans le registre MLflow (alias `champion`,
  `challenger`, `ancien_champion`) : le code et le modèle se déploient indépendamment, et
  c'est voulu — un modèle se remplace sans reconstruire l'API.

## Secrets et moindre privilège

- Aucun secret dans le dépôt. Ils vivent dans le `.env` de la machine de production, créé une
  fois par `python scripts/generate_env.py` et jamais versionné (`.gitignore`). Le flux le
  copie dans son espace de travail le temps du déploiement, **sans jamais en afficher le
  contenu**, et l'efface ensuite quelle que soit l'issue. Le dossier d'installation est une
  variable de dépôt (`ONDINE_INSTALL`), pas un chemin gravé dans le flux.
- `permissions: contents: read` sur les deux flux : ils lisent le dépôt et rien d'autre.
- Les journaux de flux ne contiennent ni mot de passe ni donnée personnelle ; les commandes
  qui liraient des données de production n'y figurent pas.
- `concurrency: cd-production` sans annulation : une bascule interrompue au milieu laisserait
  la plateforme dans un état indéterminé.
- Le runner tourne sous un compte dédié, dans le dossier du dépôt, et n'a accès qu'à Docker.

## Installer le runner (une seule fois)

Sur la machine de production, dans **Settings → Actions → Runners → New self-hosted runner**
du dépôt, puis en suivant les commandes affichées. Le jeton proposé est à usage unique et
n'est pas à conserver.

Deux points qui comptent :

- donner au runner les étiquettes **`self-hosted`** et **`ondine`** (les flux les exigent) ;
- l'installer en service pour qu'il redémarre avec la machine (`./svc.sh install` puis
  `./svc.sh start`, ou `.\svc.cmd install` sous Windows).

Le runner a besoin de `docker` et `git` dans son `PATH`. Rien d'autre : tout le reste tourne
en conteneur.

Deux pièges rencontrés, qui valent d'être connus :

- **`shell: bash` désigne le bash de WSL**, pas celui de Git : `C:\Windows\System32\bash.exe`
  passe avant dans le `PATH`, et il ne sait pas lire les chemins Windows que le runner lui
  transmet. Les étapes sont donc en PowerShell, et `deployer.sh` est appelé explicitement par
  le bash de Git.
- **`shell: pwsh` n'existe pas** sur une installation Windows standard : `pwsh` est
  PowerShell 7, qui s'installe séparément. Seul `powershell` (5.1, livré avec le système) est
  garanti.
- **GitHub exécute les étapes PowerShell avec `$ErrorActionPreference = 'Stop'`**, ce qui fait
  qu'une simple ligne écrite sur la sortie d'erreur par un programme externe interrompt
  l'étape. `docker compose` y écrit son avancement et ses avertissements : l'étape échouait en
  0,3 s alors que la commande, relancée à la main, réussissait. Chaque bloc rétablit
  `Continue` et vérifie explicitement `$LASTEXITCODE`.

Les trois pièges échouent de la même façon : le flux démarre, donc tout semble en place, puis
l'étape s'arrête en quelques secondes sans message exploitable. C'est ce qui rend la première
mise en place d'un runner auto-hébergé plus longue qu'elle n'en a l'air.
- **Ne pas installer le runner en service** sur ce poste : le service tourne sous
  `NT AUTHORITY\NETWORK SERVICE`, qui n'a pas accès à Docker Desktop. Il est lancé au premier
  plan (`.\run.cmd`), ce qui a l'avantage de rendre les exécutions visibles pendant une
  démonstration.

## Limites assumées

- **Un seul environnement.** Il n'y a pas de pré-production : la machine est unique. Le
  retour arrière automatique est ce qui tient lieu de filet, et les migrations sont écrites
  pour être rejouables. Sur une plateforme réelle, `cd.yml` déploierait d'abord sur un
  environnement de recette, avec la même mécanique.
- **Pas de registre d'images distant.** Les images restent locales à la machine ; sur une
  cible cloud, elles seraient poussées vers un registre privé et signées.
- Le réentraînement du modèle n'est **pas** dans la chaîne de livraison : il a son propre
  cycle (DAG `ondine_model_refresh`), déclenché par la dérive et non par un commit. Mélanger
  les deux ferait dépendre la mise à jour d'un modèle d'une modification de code.
