"""
Configuration for the 8 North Eastern Region (NER) states.
Includes bounding boxes and base metadata required for the validation pipeline.
"""

NER_STATES_CONFIG = {
    "Sikkim": {
        "id": "sikkim",
        "min_lat": 27.0,
        "max_lat": 28.2,
        "min_lon": 88.0,
        "max_lon": 89.0,
        "is_pilot": True,
        "pilot_area": "East Sikkim"
    },
    "Arunachal Pradesh": {
        "id": "arunachal_pradesh",
        "min_lat": 26.5,
        "max_lat": 29.5,
        "min_lon": 91.5,
        "max_lon": 97.5,
        "is_pilot": False
    },
    "Assam": {
        "id": "assam",
        "min_lat": 24.0,
        "max_lat": 28.0,
        "min_lon": 89.5,
        "max_lon": 96.0,
        "is_pilot": False
    },
    "Manipur": {
        "id": "manipur",
        "min_lat": 23.8,
        "max_lat": 25.7,
        "min_lon": 93.0,
        "max_lon": 94.8,
        "is_pilot": False
    },
    "Meghalaya": {
        "id": "meghalaya",
        "min_lat": 25.0,
        "max_lat": 26.1,
        "min_lon": 89.8,
        "max_lon": 92.8,
        "is_pilot": False
    },
    "Mizoram": {
        "id": "mizoram",
        "min_lat": 21.9,
        "max_lat": 24.5,
        "min_lon": 92.2,
        "max_lon": 93.4,
        "is_pilot": False
    },
    "Nagaland": {
        "id": "nagaland",
        "min_lat": 25.2,
        "max_lat": 27.0,
        "min_lon": 93.3,
        "max_lon": 95.3,
        "is_pilot": False
    },
    "Tripura": {
        "id": "tripura",
        "min_lat": 22.9,
        "max_lat": 24.5,
        "min_lon": 91.1,
        "max_lon": 92.4,
        "is_pilot": False
    }
}
