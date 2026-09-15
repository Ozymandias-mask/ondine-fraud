"""Référentiels statiques : villes, catégories de marchands, pays, profils horaires."""

from __future__ import annotations

import numpy as np

# Villes françaises : (nom, code postal, région, latitude, longitude, poids de population).
CITIES: list[tuple[str, str, str, float, float, float]] = [
    ("Paris", "75000", "Île-de-France", 48.8566, 2.3522, 21.0),
    ("Boulogne-Billancourt", "92100", "Île-de-France", 48.8397, 2.2399, 1.2),
    ("Saint-Denis", "93200", "Île-de-France", 48.9362, 2.3574, 1.1),
    ("Versailles", "78000", "Île-de-France", 48.8049, 2.1204, 0.9),
    ("Créteil", "94000", "Île-de-France", 48.7904, 2.4556, 0.9),
    ("Marseille", "13000", "Provence-Alpes-Côte d'Azur", 43.2965, 5.3698, 8.7),
    ("Nice", "06000", "Provence-Alpes-Côte d'Azur", 43.7102, 7.2620, 3.4),
    ("Toulon", "83000", "Provence-Alpes-Côte d'Azur", 43.1242, 5.9280, 1.7),
    ("Aix-en-Provence", "13090", "Provence-Alpes-Côte d'Azur", 43.5297, 5.4474, 1.4),
    ("Avignon", "84000", "Provence-Alpes-Côte d'Azur", 43.9493, 4.8055, 0.9),
    ("Lyon", "69000", "Auvergne-Rhône-Alpes", 45.7640, 4.8357, 5.2),
    ("Grenoble", "38000", "Auvergne-Rhône-Alpes", 45.1885, 5.7245, 1.6),
    ("Saint-Étienne", "42000", "Auvergne-Rhône-Alpes", 45.4397, 4.3872, 1.7),
    ("Clermont-Ferrand", "63000", "Auvergne-Rhône-Alpes", 45.7772, 3.0870, 1.4),
    ("Annecy", "74000", "Auvergne-Rhône-Alpes", 45.8992, 6.1294, 1.2),
    ("Toulouse", "31000", "Occitanie", 43.6047, 1.4442, 4.8),
    ("Montpellier", "34000", "Occitanie", 43.6108, 3.8767, 2.9),
    ("Nîmes", "30000", "Occitanie", 43.8367, 4.3601, 1.4),
    ("Perpignan", "66000", "Occitanie", 42.6887, 2.8948, 1.2),
    ("Bordeaux", "33000", "Nouvelle-Aquitaine", 44.8378, -0.5792, 2.6),
    ("Limoges", "87000", "Nouvelle-Aquitaine", 45.8336, 1.2611, 1.3),
    ("Poitiers", "86000", "Nouvelle-Aquitaine", 46.5802, 0.3404, 0.9),
    ("La Rochelle", "17000", "Nouvelle-Aquitaine", 46.1603, -1.1511, 0.8),
    ("Nantes", "44000", "Pays de la Loire", 47.2184, -1.5536, 3.2),
    ("Angers", "49000", "Pays de la Loire", 47.4784, -0.5632, 1.5),
    ("Le Mans", "72000", "Pays de la Loire", 48.0061, 0.1996, 1.4),
    ("Rennes", "35000", "Bretagne", 48.1173, -1.6778, 2.2),
    ("Brest", "29200", "Bretagne", 48.3904, -4.4861, 1.4),
    ("Lille", "59000", "Hauts-de-France", 50.6292, 3.0573, 2.3),
    ("Amiens", "80000", "Hauts-de-France", 49.8941, 2.2958, 1.3),
    ("Roubaix", "59100", "Hauts-de-France", 50.6942, 3.1746, 1.0),
    ("Strasbourg", "67000", "Grand Est", 48.5734, 7.7521, 2.8),
    ("Reims", "51100", "Grand Est", 49.2583, 4.0317, 1.8),
    ("Metz", "57000", "Grand Est", 49.1193, 6.1757, 1.2),
    ("Mulhouse", "68100", "Grand Est", 47.7508, 7.3359, 1.1),
    ("Nancy", "54000", "Grand Est", 48.6921, 6.1844, 1.1),
    ("Dijon", "21000", "Bourgogne-Franche-Comté", 47.3220, 5.0415, 1.6),
    ("Besançon", "25000", "Bourgogne-Franche-Comté", 47.2378, 6.0241, 1.2),
    ("Rouen", "76000", "Normandie", 49.4432, 1.0999, 1.1),
    ("Le Havre", "76600", "Normandie", 49.4944, 0.1079, 1.7),
    ("Caen", "14000", "Normandie", 49.1829, -0.3707, 1.1),
    ("Tours", "37000", "Centre-Val de Loire", 47.3941, 0.6848, 1.4),
    ("Orléans", "45000", "Centre-Val de Loire", 47.9030, 1.9093, 1.2),
    ("Ajaccio", "20000", "Corse", 41.9192, 8.7386, 0.7),
]

# Catégories de marchands : (catégorie, code MCC, mu et sigma du log-montant,
# part des marchands en ligne, poids d'utilisation, sensible à la fraude).
MERCHANT_CATEGORIES: list[tuple[str, int, float, float, float, float, bool]] = [
    ("grocery", 5411, 3.3, 0.8, 0.10, 26.0, False),
    ("restaurant", 5812, 3.1, 0.6, 0.05, 11.0, False),
    ("fast_food", 5814, 2.4, 0.5, 0.15, 9.0, False),
    ("fuel", 5541, 4.0, 0.4, 0.00, 7.0, False),
    ("pharmacy", 5912, 2.9, 0.7, 0.10, 4.0, False),
    ("clothing", 5651, 4.0, 0.7, 0.45, 6.0, False),
    ("electronics", 5732, 4.6, 1.0, 0.60, 3.0, True),
    ("airline", 4511, 5.4, 0.7, 0.95, 1.0, True),
    ("hotel", 7011, 5.0, 0.6, 0.70, 1.2, False),
    ("taxi_vtc", 4121, 3.0, 0.5, 0.90, 3.5, False),
    ("digital_goods", 5818, 2.3, 0.9, 1.00, 5.0, True),
    ("marketplace", 5399, 3.7, 0.9, 1.00, 7.0, False),
    ("jewelry", 5944, 5.6, 0.8, 0.30, 0.4, True),
    ("gambling", 7995, 3.5, 1.0, 1.00, 0.9, True),
    ("atm_cash", 6011, 4.2, 0.6, 0.00, 5.0, True),
    ("subscription", 4899, 2.6, 0.5, 1.00, 5.0, False),
    ("telecom", 4814, 3.4, 0.4, 0.80, 2.0, False),
    ("home_improvement", 5200, 4.1, 0.9, 0.30, 3.0, False),
]

# Pays étrangers : pays touristiques (achats légitimes) et pays à risque (fraude).
TRAVEL_COUNTRIES: list[tuple[str, float, float]] = [
    ("ES", 40.4168, -3.7038),
    ("IT", 41.9028, 12.4964),
    ("DE", 52.5200, 13.4050),
    ("BE", 50.8503, 4.3517),
    ("PT", 38.7223, -9.1393),
    ("GB", 51.5074, -0.1278),
    ("CH", 46.2044, 6.1432),
    ("NL", 52.3676, 4.9041),
]
RISKY_COUNTRIES: list[tuple[str, float, float]] = [
    ("RO", 44.4268, 26.1025),
    ("BR", -23.5505, -46.6333),
    ("TH", 13.7563, 100.5018),
    ("NG", 6.5244, 3.3792),
    ("ID", -6.2088, 106.8456),
    ("MX", 19.4326, -99.1332),
    ("VN", 10.8231, 106.6297),
    ("PH", 14.5995, 120.9842),
]

# Probabilité d'une transaction légitime selon l'heure de la journée (0 h à 23 h).
# fmt: off
_LEGIT_HOURS = np.array([
    0.4, 0.2, 0.1, 0.1, 0.1, 0.2, 0.8, 2.0, 3.8, 4.8, 5.6, 7.0,
    8.6, 7.2, 5.6, 5.4, 6.0, 7.4, 8.2, 7.6, 5.4, 3.4, 2.0, 1.1,
])
# fmt: on
LEGIT_HOUR_PROBS = _LEGIT_HOURS / _LEGIT_HOURS.sum()

# Les fraudeurs agissent plus souvent la nuit, quand la victime dort.
# fmt: off
_FRAUD_HOURS = np.array([
    5.0, 6.0, 6.5, 6.5, 6.0, 5.0, 3.0, 2.5, 3.0, 3.5, 4.0, 4.0,
    4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 5.0,
])
# fmt: on
FRAUD_HOUR_PROBS = _FRAUD_HOURS / _FRAUD_HOURS.sum()

# Saisonnalité mensuelle (janvier à décembre) : soldes, Black Friday, Noël.
_SEASONALITY = np.array([1.00, 0.88, 0.95, 0.97, 1.00, 1.00, 1.05, 0.93, 0.98, 1.00, 1.08, 1.26])
SEASONALITY = _SEASONALITY / _SEASONALITY.mean()

EMAIL_DOMAINS = [
    "gmail.com",
    "orange.fr",
    "free.fr",
    "outlook.fr",
    "yahoo.fr",
    "laposte.net",
    "sfr.fr",
    "hotmail.fr",
    "icloud.com",
    "proton.me",
]

# Segments clients : (nom, poids, âge min, âge max, multiplicateur d'activité).
SEGMENTS: list[tuple[str, float, int, int, float]] = [
    ("student", 0.14, 18, 25, 0.8),
    ("young_active", 0.30, 26, 35, 1.2),
    ("active", 0.36, 36, 60, 1.05),
    ("senior", 0.20, 61, 90, 0.7),
]

MOBILE_OS = ["android", "ios"]
