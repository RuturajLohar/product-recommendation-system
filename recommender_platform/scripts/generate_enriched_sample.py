import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.db import models
from app.db.session import SessionLocal


OUTPUT_COLUMNS = [
    "product_id", "title", "brand", "category", "subcategory", "description",
    "features", "specifications", "keywords/tags", "price", "rating",
    "review_count", "popularity", "best_seller", "availability", "image_url",
    "product_url",
]

DOMAIN_PROFILES = {
    "technology": {
        "brands": ["Nexora", "Voltaris", "Aveniq", "Orbyte", "Zenithra", "Kinetiq", "Lumetrix", "Tekvora"],
        "subcategories": ["Smart Electronics", "Computing Accessories", "Connected Devices", "Digital Storage", "Personal Computing", "Power & Connectivity"],
        "specs": {
            "connectivity": ["Bluetooth 5.3", "Wi-Fi 6", "USB-C", "Multi-platform wireless"],
            "compatibility": ["Windows and macOS", "Android and iOS", "Multi-platform", "USB-enabled devices"],
            "warranty": ["12 months", "18 months", "24 months"],
            "finish": ["Matte graphite", "Satin silver", "Midnight black", "Pearl white"],
        },
    },
    "beauty": {
        "brands": ["Velmora", "Aurevia", "Elowen Beauty", "Seranova", "Lumielle", "Rosavie", "Nuvessa", "Calithea"],
        "subcategories": ["Daily Skin Care", "Professional Make-up", "Hair Treatment", "Bath Essentials", "Personal Fragrance", "Nail Care"],
        "specs": {
            "formulation": ["Lightweight cream", "Fast-absorbing serum", "Silky powder", "Gentle liquid"],
            "suitable_for": ["All skin types", "Normal to dry skin", "Combination skin", "Sensitive skin"],
            "size": ["30 ml", "50 ml", "100 ml", "200 ml"],
            "finish": ["Natural", "Matte", "Radiant", "Satin"],
        },
    },
    "fashion": {
        "brands": ["Avelune", "North & Vale", "Mirello", "Crestwyn", "Solvane", "Elaris", "Vantelle", "Oakmere"],
        "subcategories": ["Everyday Apparel", "Performance Footwear", "Fashion Accessories", "Travel Accessories", "Occasion Wear", "Contemporary Jewellery"],
        "specs": {
            "material": ["Cotton blend", "Premium synthetic", "Soft-touch textile", "Durable woven fabric"],
            "fit": ["Regular fit", "Relaxed fit", "Tailored fit", "Adjustable fit"],
            "care": ["Machine washable", "Hand wash recommended", "Wipe clean", "Gentle cycle"],
            "style": ["Contemporary", "Classic", "Minimal", "Sport-inspired"],
        },
    },
    "home": {
        "brands": ["Hearthwyn", "Casa Verdan", "Oak & Ember", "Lunora Home", "Nestoria", "Brindle & Co.", "Virelia", "Homestead Lane"],
        "subcategories": ["Kitchen Essentials", "Home Organisation", "Furniture & Decor", "Lighting Solutions", "Garden Living", "Bedding & Bath"],
        "specs": {
            "material": ["Stainless steel", "Engineered wood", "BPA-free polymer", "Cotton blend"],
            "style": ["Modern", "Scandinavian", "Classic", "Minimalist"],
            "recommended_room": ["Kitchen", "Living room", "Bedroom", "Home office"],
            "care": ["Wipe clean", "Hand wash", "Machine washable", "Low-maintenance finish"],
        },
    },
    "sports": {
        "brands": ["Strivex", "Peakforge", "AeroStride", "Vantage Athletics", "Trailnova", "CoreSprint", "Ridgewell", "Motionary"],
        "subcategories": ["Training Equipment", "Team Sports", "Outdoor Recreation", "Fitness Accessories", "Performance Footwear", "Winter Sports"],
        "specs": {
            "material": ["Performance polymer", "Reinforced textile", "Lightweight alloy", "High-grip rubber"],
            "skill_level": ["Beginner to intermediate", "Intermediate", "Recreational", "All skill levels"],
            "use_case": ["Indoor training", "Outdoor training", "Competition practice", "Everyday fitness"],
            "weather_rating": ["Indoor use", "Weather resistant", "All-season", "Quick-dry construction"],
        },
    },
    "toys": {
        "brands": ["Brightkin", "Playvora", "Wondermint", "Kiddara", "Imaginest", "Joyfoundry", "CleverSprout", "TinyQuest"],
        "subcategories": ["Creative Play", "Learning Toys", "Construction Toys", "Puzzles & Games", "Pretend Play", "Collectible Toys"],
        "specs": {
            "age_range": ["3 years and up", "5 years and up", "6 years and up", "8 years and up"],
            "material": ["Durable ABS", "Cardboard and paper", "Soft textile", "Responsibly sourced wood"],
            "play_style": ["Creative", "Educational", "Collaborative", "Imaginative"],
            "safety": ["Rounded-edge design", "Non-toxic materials", "Child-safe construction", "Easy-clean surface"],
        },
    },
    "automotive": {
        "brands": ["Motoryn", "Roadvanta", "Axelcrest", "Drivevera", "Torqline", "Autonexis", "Gearmont", "Velocraft"],
        "subcategories": ["Vehicle Accessories", "Motorcycle Accessories", "Car Care", "Replacement Components", "In-car Electronics", "Workshop Equipment"],
        "specs": {
            "compatibility": ["Universal vehicle fit", "Passenger vehicles", "Motorcycles and scooters", "12 V vehicle systems"],
            "material": ["Automotive-grade polymer", "Corrosion-resistant steel", "Reinforced rubber", "Aluminium alloy"],
            "installation": ["Tool-free installation", "Standard-fit installation", "Self-adhesive mounting", "Professional installation recommended"],
            "finish": ["Matte black", "Brushed metal", "Carbon-look", "Gloss black"],
        },
    },
    "media_av": {
        "brands": ["Sonivara", "Optivue", "Audion Crest", "Framewave", "Resonix", "Clarityn", "Voxmere", "Lensora"],
        "subcategories": ["Personal Audio", "Home Audio", "Cameras & Optics", "Streaming Devices", "Studio Equipment", "Photo Accessories"],
        "specs": {
            "connectivity": ["Bluetooth 5.2", "3.5 mm audio", "HDMI", "USB-C"],
            "compatibility": ["Phones and tablets", "Cameras and tripods", "Home entertainment systems", "PC and console"],
            "form_factor": ["Compact", "Portable", "Desktop", "Full-size"],
            "warranty": ["12 months", "18 months", "24 months"],
        },
    },
    "tools": {
        "brands": ["Forgewell", "Duracrest", "Ironvale", "Protorq", "Buildora", "Craftbolt", "Workspire", "Hammerlyn"],
        "subcategories": ["Power Tools", "Hand Tools", "Workshop Supplies", "Measuring Equipment", "Hardware", "Safety Equipment"],
        "specs": {
            "power_source": ["Corded electric", "Rechargeable battery", "Manual", "Mains powered"],
            "material": ["Hardened steel", "Impact-resistant polymer", "Chrome vanadium steel", "Aluminium alloy"],
            "duty_level": ["Home DIY", "Workshop", "Heavy duty", "Precision work"],
            "warranty": ["12 months", "24 months", "36 months"],
        },
    },
    "office": {
        "brands": ["Scriptora", "Desklyn", "Paperwell", "Notivue", "Clearfolio", "Worknest", "Inkwright", "Organique"],
        "subcategories": ["Writing Supplies", "Desk Organisation", "Paper Products", "School Supplies", "Office Equipment", "Presentation Supplies"],
        "specs": {
            "material": ["Recycled paper", "Durable polymer", "Anodised aluminium", "Card and paper"],
            "format": ["A4", "A5", "Desktop size", "Portable size"],
            "use_case": ["Office", "School", "Home study", "Professional presentation"],
            "pack_size": ["Single item", "Pack of 3", "Pack of 6", "Pack of 12"],
        },
    },
    "care": {
        "brands": ["Pawvera", "Little Haven", "Nurturely", "Petmora", "Kindred Care", "Bloomcub", "Whiskerwell", "Tendernest"],
        "subcategories": ["Baby Essentials", "Nursery Accessories", "Pet Care", "Pet Feeding", "Wildlife Care", "Safety & Hygiene"],
        "specs": {
            "material": ["BPA-free polymer", "Soft cotton blend", "Food-grade silicone", "Easy-clean textile"],
            "life_stage": ["Newborn and infant", "Toddler", "Adult pets", "All life stages"],
            "care": ["Wipe clean", "Machine washable", "Hand wash", "Dishwasher safe"],
            "safety": ["Rounded-edge design", "Non-toxic materials", "Secure closure", "Skin-friendly construction"],
        },
    },
    "arts": {
        "brands": ["Artivane", "Craftelle", "Makers Grove", "Canvas & Loom", "Huefoundry", "Stitchora", "Form & Fibre", "Musewell"],
        "subcategories": ["Drawing & Painting", "Craft Supplies", "Sewing & Textiles", "Handmade Decor", "Model Making", "Creative Kits"],
        "specs": {
            "material": ["Artist-grade paper", "Cotton textile", "Water-based pigment", "Craft wood"],
            "skill_level": ["Beginner", "Intermediate", "All skill levels", "Experienced maker"],
            "craft_style": ["Contemporary", "Traditional", "Mixed media", "Decorative"],
            "pack_size": ["Single item", "Starter set", "12-piece set", "24-piece set"],
        },
    },
    "media": {
        "brands": ["Storyvale", "Pagecrest", "Northlight Press", "Silverleaf Media", "Arc & Quill", "Brightword", "Cinevista", "Melody House"],
        "subcategories": ["Fiction", "Non-fiction", "Reference", "Children's Media", "Film & Television", "Music & Audio"],
        "specs": {
            "format": ["Paperback", "Hardcover", "Digital media", "Audio edition"],
            "language": ["English", "English language edition", "Bilingual edition"],
            "audience": ["General audience", "Young adult", "Children", "Specialist reader"],
            "edition": ["Standard edition", "Revised edition", "Collector edition", "Illustrated edition"],
        },
    },
    "lifestyle": {
        "brands": ["Evermere", "Novelle & Co.", "Viremont", "Asterlane", "Morrow & Finch", "Celandor", "Rivensa", "Elmridge"],
        "subcategories": ["Everyday Essentials", "Personal Accessories", "Leisure Products", "Gift Collections", "Seasonal Essentials", "Speciality Products"],
        "specs": {
            "material": ["Premium composite", "Durable textile", "Lightweight polymer", "Natural fibre blend"],
            "style": ["Contemporary", "Classic", "Minimal", "Decorative"],
            "use_case": ["Everyday use", "Home and travel", "Gifting", "Leisure"],
            "warranty": ["12 months", "18 months", "24 months"],
        },
    },
}

DOMAIN_RULES = [
    ("beauty", ("beauty", "skin care", "hair care", "make-up", "fragrance", "manicure", "pedicure", "bath & body")),
    ("technology", ("computer", "pc", "cpu", "gpu", "laptop", "mobile phone", "electronics", "keyboard", "memory", "data storage", "monitor", "printer", "scanner", "network", "software", "video game")),
    ("media_av", ("camera", "photo", "audio", "hi-fi", "radio", "speaker", "headphone", "music equipment", "streaming", "cinema", "tv", "video", "lens", "digital frame")),
    ("toys", ("toy", "game", "puzzle", "kids", "play", "learning & education")),
    ("office", ("office", "stationery", "pens", "pencils", "writing supplies", "school supply", "education supplies", "paper products")),
    ("care", ("baby", "pet", "dog", "cat", "bird", "wildlife", "nursery")),
    ("automotive", ("automotive", "motorcycle", "motorbike", "vehicle", "car accessory", "car care")),
    ("tools", ("tool", "hardware", "diy", "industrial", "building supply", "measuring")),
    ("sports", ("sport", "fitness", "cycling", "football", "cricket", "ski", "golf", "exercise", "camping", "hiking")),
    ("fashion", ("men", "women", "boys", "girls", "clothing", "shoe", "fashion", "watch", "luggage", "jewellery", "jewelry", "accessories")),
    ("home", ("home", "kitchen", "furniture", "lighting", "garden", "storage", "bedding", "cook", "appliance", "vacuum", "floorcare", "cooler", "filter")),
    ("arts", ("handmade", "art", "craft", "sewing", "knitting", "model making")),
    ("media", ("book", "dvd", "blu-ray", "film", "music", "literature")),
]

TYPE_RULES = {
    "technology": [("3d printer", "3D Printing"), ("gaming laptop", "Gaming Laptops"), ("laptop", "Laptops"), ("smartphone", "Smartphones"), ("mobile phone", "Mobile Phones"), ("memory card", "Memory Cards"), ("ssd", "Solid State Drives"), ("keyboard", "Keyboards"), ("mouse", "Computer Mice"), ("monitor", "Monitors"), ("printer", "Printers"), ("scanner", "Scanners"), ("joystick", "Gaming Controllers"), ("router", "Routers"), ("charger", "Chargers")],
    "beauty": [("serum", "Face Serums"), ("cleanser", "Cleansers"), ("shampoo", "Shampoo"), ("conditioner", "Conditioner"), ("lipstick", "Lipstick"), ("eyebrow", "Eyebrow Makeup"), ("mascara", "Mascara"), ("foundation", "Foundation"), ("nail polish", "Nail Polish")],
    "fashion": [("sneaker", "Sneakers"), ("shoe", "Shoes"), ("boot", "Boots"), ("dress", "Dresses"), ("shirt", "Shirts"), ("jacket", "Jackets"), ("scarf", "Scarves"), ("backpack", "Backpacks"), ("wallet", "Wallets"), ("necklace", "Necklaces"), ("earring", "Earrings")],
    "home": [("frying pan", "Frying Pans"), ("cookware", "Cookware"), ("lamp", "Lamps"), ("chair", "Chairs"), ("table", "Tables"), ("mattress", "Mattresses"), ("vacuum", "Vacuum Cleaners"), ("storage", "Storage & Organisation")],
    "sports": [("football", "Football Equipment"), ("soccer", "Football Equipment"), ("cricket", "Cricket Equipment"), ("ski", "Ski Equipment"), ("tent", "Outdoor Shelters"), ("yoga", "Yoga Equipment")],
    "toys": [("puzzle", "Puzzles"), ("board game", "Board Games"), ("action figure", "Action Figures"), ("doll", "Dolls"), ("building", "Building Sets")],
    "media_av": [("wall mount", "Home AV Accessories"), ("soundbar", "Home Audio"), ("headphone", "Headphones"), ("earbud", "Earbuds"), ("speaker", "Speakers"), ("microphone", "Microphones"), ("camera", "Cameras"), ("lens", "Camera Lenses"), ("tripod", "Tripods")],
    "tools": [("drill", "Power Drills"), ("saw", "Power Saws"), ("screwdriver", "Screwdrivers"), ("wrench", "Wrenches"), ("measure", "Measuring Tools")],
    "office": [("pen", "Writing Instruments"), ("pencil", "Writing Instruments"), ("notebook", "Notebooks"), ("folder", "Folders & Filing"), ("stapler", "Desk Equipment")],
}

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "compatible", "for", "from", "in", "is", "it", "made", "of", "on", "or", "pack", "set", "the", "to", "with", "your", "black", "white", "new"}
FORBIDDEN_VALUES = {"", "unknown", "n/a", "na", "none", "null", "nil", "not available"}

SPEC_PATTERNS = [
    ("storage", re.compile(r"\b(\d+(?:\.\d+)?\s?(?:TB|GB|MB))\b", re.I)),
    ("memory", re.compile(r"\b(\d+\s?GB\s+(?:DDR[345]|RAM))\b", re.I)),
    ("display_size", re.compile(r"\b(\d{1,2}(?:\.\d+)?\s?(?:inch|inches|\"))\b", re.I)),
    ("resolution", re.compile(r"\b(\d{3,4}\s?[xX]\s?\d{3,4}|4K|8K|FHD|QHD|UHD)\b", re.I)),
    ("frequency", re.compile(r"\b(\d+(?:\.\d+)?\s?(?:GHz|MHz|Hz))\b", re.I)),
    ("power", re.compile(r"\b(\d+(?:\.\d+)?\s?(?:W|watt|watts))\b", re.I)),
    ("quantity", re.compile(r"\b(\d+\s?(?:pcs|pieces|pack|count))\b", re.I)),
]


def _stable_int(seed: int, product_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}:{product_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_pick(values: Sequence[str], seed: int, product_id: str, salt: str) -> str:
    return values[_stable_int(seed, product_id, salt) % len(values)]


def classify_domain(category: str, title: str) -> str:
    category_text = category.lower()
    for domain, keywords in DOMAIN_RULES:
        if any(_keyword_matches(category_text, keyword) for keyword in keywords):
            return domain
    return "lifestyle"


def _keyword_matches(text: str, keyword: str) -> bool:
    if keyword in {"art", "car", "cat", "dog", "men", "pc", "pet", "tv"}:
        return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None
    return keyword in text


def _allocate_quotas(counts: Dict[str, int], target: int) -> Dict[str, int]:
    target = min(target, sum(counts.values()))
    quotas = {category: min(count, 5) for category, count in counts.items()}
    remaining = target - sum(quotas.values())
    capacities = {category: counts[category] - quotas[category] for category in counts}
    while remaining > 0:
        total_capacity = sum(capacities.values())
        if total_capacity <= 0:
            break
        additions = {
            category: min(capacity, int(remaining * capacity / total_capacity))
            for category, capacity in capacities.items()
        }
        allocated = sum(additions.values())
        if allocated == 0:
            category = max(capacities, key=capacities.get)
            additions[category] = 1
            allocated = 1
        for category, addition in additions.items():
            quotas[category] += addition
            capacities[category] -= addition
        remaining -= allocated
    return quotas


def _select_products(rows: Sequence[Any], sample_size: int, seed: int) -> List[Any]:
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for row in rows:
        if not row.category:
            raise ValueError(f"Product {row.asin} has no category")
        grouped[row.category].append(row)
    quotas = _allocate_quotas({key: len(value) for key, value in grouped.items()}, sample_size)
    selected = []
    for category, products in grouped.items():
        products.sort(key=lambda row: _stable_int(seed, row.asin, "sample"))
        selected.extend(products[:quotas[category]])
    selected.sort(key=lambda row: _stable_int(seed, row.asin, "output-order"))
    return selected


def _generate_brand(domain: str, product_id: str, seed: int) -> str:
    return _stable_pick(DOMAIN_PROFILES[domain]["brands"], seed, product_id, "brand")


def _infer_subcategory(domain: str, category: str, title: str, product_id: str, seed: int) -> str:
    category_text = category.lower()
    strong_category_matches = [
        (("outdoor lighting",), "Lighting Solutions"),
        (("vacuum", "floorcare"), "Vacuum Cleaners"),
        (("water cooler", "filter cartridge"), "Kitchen Essentials"),
        (("lens",), "Camera Lenses"),
        (("kids' play vehicle",), "Pretend Play"),
        (("professional education",), "School Supplies"),
    ]
    for category_keywords, subcategory in strong_category_matches:
        if any(keyword in category_text for keyword in category_keywords):
            return subcategory

    lowered = title.lower()
    for keyword, subcategory in TYPE_RULES.get(domain, []):
        if keyword in lowered:
            return subcategory
    candidates = DOMAIN_PROFILES[domain]["subcategories"]
    if domain == "care":
        candidates = candidates[:2] + candidates[-1:] if "baby" in category_text or "nursery" in category_text else candidates[2:]
    elif domain == "media_av":
        candidates = candidates[2:] if any(word in category_text for word in ("camera", "photo", "lens", "frame")) else candidates[:2] + candidates[3:5]
    elif domain == "beauty":
        if "skin" in category_text:
            candidates = ["Daily Skin Care"]
        elif "hair" in category_text:
            candidates = ["Hair Treatment"]
        elif "fragrance" in category_text:
            candidates = ["Personal Fragrance"]
        elif any(word in category_text for word in ("make-up", "manicure", "pedicure")):
            candidates = ["Professional Make-up", "Nail Care"]
    elif domain == "home":
        if any(word in category_text for word in ("kitchen", "cook")):
            candidates = ["Kitchen Essentials"]
        elif "lighting" in category_text:
            candidates = ["Lighting Solutions"]
        elif "storage" in category_text:
            candidates = ["Home Organisation"]
        elif "garden" in category_text:
            candidates = ["Garden Living"]
    elif domain == "fashion":
        if any(word in category_text for word in ("jewellery", "jewelry")):
            candidates = ["Contemporary Jewellery"]
        elif "luggage" in category_text:
            candidates = ["Travel Accessories"]
        elif "shoe" in category_text:
            candidates = ["Performance Footwear"]
    return _stable_pick(candidates, seed, product_id, "subcategory")


def _extract_specifications(title: str) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    for key, pattern in SPEC_PATTERNS:
        match = pattern.search(title)
        if match:
            specs.setdefault(key, re.sub(r"\s+", " ", match.group(1)).strip())
    for connectivity in ("Bluetooth", "Wi-Fi", "USB-C", "USB 3.0", "HDMI", "NFC"):
        if connectivity.lower() in title.lower():
            specs.setdefault("connectivity", connectivity)
            break
    return specs


def _spec_profile(domain: str, subcategory: str) -> Dict[str, Sequence[str]]:
    if domain == "media_av" and subcategory == "Home AV Accessories":
        return {
            "compatibility": ["TV and media devices", "Home audio equipment", "Streaming devices", "Display equipment"],
            "material": ["Powder-coated steel", "Reinforced aluminium", "Impact-resistant polymer", "Steel and polymer"],
            "installation": ["Wall mounted", "Tool-assisted installation", "Adjustable mounting", "Cable-managed installation"],
            "warranty": ["12 months", "18 months", "24 months"],
        }
    if domain == "media_av" and subcategory == "Streaming Devices":
        return {
            "connectivity": ["HDMI and Wi-Fi", "Wi-Fi and Bluetooth", "HDMI", "USB-C and Wi-Fi"],
            "compatibility": ["HD televisions", "Home entertainment systems", "Phones and televisions", "TV and monitor displays"],
            "resolution": ["Full HD", "4K UHD", "Up to 4K", "1080p"],
            "warranty": ["12 months", "18 months", "24 months"],
        }
    if domain == "media_av" and subcategory in {"Cameras", "Camera Lenses", "Tripods", "Cameras & Optics", "Photo Accessories"}:
        return {
            "compatibility": ["Mirrorless cameras", "DSLR cameras", "Standard camera mounts", "Photo and video equipment"],
            "construction": ["Optical glass and alloy", "Reinforced aluminium", "Lightweight composite", "Weather-resistant polymer"],
            "intended_use": ["Portrait photography", "Travel photography", "Studio photography", "Photo and video capture"],
            "warranty": ["12 months", "18 months", "24 months"],
        }
    if domain == "beauty" and subcategory in {"Professional Make-up", "Mascara", "Lipstick", "Foundation", "Eyebrow Makeup", "Nail Polish", "Nail Care"}:
        return {
            "formulation": ["Buildable cream", "Long-wear liquid", "Fine pressed powder", "Smooth gel"],
            "suitable_for": ["Everyday wear", "Professional make-up", "Sensitive skin", "All skin types"],
            "size": ["5 ml", "8 ml", "12 ml", "15 ml"],
            "finish": ["Natural", "Matte", "Radiant", "Satin"],
        }
    if domain == "fashion" and subcategory in {"Contemporary Jewellery", "Necklaces", "Earrings"}:
        return {
            "material": ["Stainless steel", "Gold-tone alloy", "Sterling silver", "Silver-tone alloy"],
            "finish": ["Polished", "Brushed", "Hammered", "High shine"],
            "style": ["Contemporary", "Classic", "Minimal", "Statement"],
            "care": ["Polish with a soft cloth", "Keep dry", "Store in a soft pouch", "Avoid direct contact with perfume"],
        }
    if domain == "home" and subcategory == "Lighting Solutions":
        return {
            "material": ["Aluminium alloy", "Powder-coated steel", "Durable polymer", "Glass and metal"],
            "power_source": ["Mains powered", "Rechargeable battery", "Solar powered", "Low-voltage supply"],
            "placement": ["Indoor and outdoor", "Indoor", "Outdoor", "Wall mounted"],
            "style": ["Modern", "Industrial", "Classic", "Minimalist"],
        }
    if domain == "care" and subcategory in {"Baby Essentials", "Nursery Accessories"}:
        return {
            "material": ["BPA-free polymer", "Soft cotton blend", "Food-grade silicone", "Breathable textile"],
            "age_range": ["Newborn and up", "3 months and up", "6 months and up", "Toddler"],
            "care": ["Wipe clean", "Machine washable", "Hand wash", "Dishwasher safe"],
            "safety": ["Rounded-edge design", "Non-toxic materials", "Secure closure", "Skin-friendly construction"],
        }
    return DOMAIN_PROFILES[domain]["specs"]


def _complete_specifications(domain: str, subcategory: str, product_id: str, title: str, seed: int) -> Dict[str, str]:
    specs = _extract_specifications(title)
    if domain == "home":
        for room in ("Living room", "Bedroom", "Kitchen", "Bathroom", "Home office"):
            if room.lower() in title.lower():
                specs["recommended_room"] = room
                break
    for key, values in _spec_profile(domain, subcategory).items():
        if len(specs) >= 4:
            break
        specs.setdefault(key, _stable_pick(values, seed, product_id, f"spec:{key}"))
    if len(specs) < 3:
        raise RuntimeError(f"Could not generate enough specifications for {product_id}")
    return specs


def _synthetic_rating(row: Any, seed: int) -> float:
    if row.stars and 1 <= float(row.stars) <= 5:
        return round(float(row.stars), 1)
    return round(3.4 + (_stable_int(seed, row.asin, "rating") % 16) / 10, 1)


def _synthetic_reviews(row: Any, seed: int) -> int:
    if row.reviews and int(row.reviews) > 0:
        return int(row.reviews)
    fraction = (_stable_int(seed, row.asin, "reviews") % 10_000) / 9_999
    return 5 + int((fraction ** 2) * 4_995)


def _best_seller(row: Any, rating: float, reviews: int, seed: int) -> bool:
    if row.is_best_seller:
        return True
    qualifies = rating >= 4.4 and reviews >= 500
    return qualifies and _stable_int(seed, row.asin, "best-seller") % 100 < 8


def _availability(row: Any, best_seller: bool, seed: int) -> str:
    if best_seller or (row.bought_in_last_month or 0) > 0:
        return "in_stock"
    bucket = _stable_int(seed, row.asin, "availability") % 100
    return "in_stock" if bucket < 90 else "limited_stock" if bucket < 98 else "out_of_stock"


def _build_features(category: str, subcategory: str, specs: Dict[str, str], rating: float, reviews: int, best_seller: bool) -> List[str]:
    features = [f"Purpose-built for {subcategory.lower()}", f"Suitable for the {category.lower()} category"]
    features.extend(f"{key.replace('_', ' ').title()}: {value}" for key, value in specs.items())
    if rating >= 4.5:
        features.append(f"Highly rated at {rating:.1f}/5")
    if reviews >= 1000:
        features.append(f"Popular with {reviews:,} reviews")
    if best_seller:
        features.append("Best-selling product")
    return features[:8]


def _build_keywords(title: str, brand: str, category: str, subcategory: str, specs: Dict[str, str]) -> str:
    candidates: Iterable[str] = [brand, category, subcategory, *specs.values()]
    candidates = list(candidates) + re.findall(r"[A-Za-z0-9][A-Za-z0-9+.-]{2,}", title.lower())
    keywords, seen = [], set()
    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", str(candidate).strip()).lower()
        if not cleaned or cleaned in STOPWORDS or cleaned in seen:
            continue
        seen.add(cleaned)
        keywords.append(cleaned)
        if len(keywords) == 20:
            break
    return "|".join(keywords)


def _description(row: Any, brand: str, category: str, subcategory: str, specs: Dict[str, str], rating: float, reviews: int) -> str:
    details = ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in list(specs.items())[:4])
    return (
        f"{brand} {subcategory.lower()} designed for shoppers exploring {category.lower()}. "
        f"Based on the product listing '{row.title}', its key specifications are {details}. "
        f"Rated {rating:.1f} out of 5 from {reviews:,} reviews."
    )


def _validate_source(row: Any) -> None:
    required = {"product_id": row.asin, "title": row.title, "category": row.category, "image_url": row.img_url, "product_url": row.product_url}
    invalid = [key for key, value in required.items() if str(value or "").strip().lower() in FORBIDDEN_VALUES]
    if invalid:
        raise ValueError(f"Product {row.asin} has unusable source fields: {', '.join(invalid)}")
    if not row.price or float(row.price) <= 0:
        raise ValueError(f"Product {row.asin} has an invalid price")


def generate(output_path: Path, sample_size: int, seed: int) -> None:
    db = SessionLocal()
    try:
        rows = db.query(models.Item).order_by(models.Item.asin).all()
    finally:
        db.close()
    if len(rows) < sample_size:
        raise ValueError(f"Database has {len(rows):,} products; requested {sample_size:,}.")

    selected = _select_products(rows, sample_size, seed)
    enriched = []
    for row in selected:
        _validate_source(row)
        domain = classify_domain(row.category, row.title)
        brand = _generate_brand(domain, row.asin, seed)
        subcategory = _infer_subcategory(domain, row.category, row.title, row.asin, seed)
        specs = _complete_specifications(domain, subcategory, row.asin, row.title, seed)
        rating = _synthetic_rating(row, seed)
        reviews = _synthetic_reviews(row, seed)
        best_seller = _best_seller(row, rating, reviews, seed)
        enriched.append((row, domain, brand, subcategory, specs, rating, reviews, best_seller))

    max_reviews = max(item[6] for item in enriched)
    max_bought = max(int(item[0].bought_in_last_month or 0) for item in enriched)
    category_counts, domain_counts, brand_counts = Counter(), Counter(), Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row, domain, brand, subcategory, specs, rating, reviews, best_seller in enriched:
            bought = int(row.bought_in_last_month or 0)
            popularity = 100 * (
                0.45 * math.log1p(reviews) / math.log1p(max_reviews)
                + 0.30 * math.log1p(bought) / math.log1p(max_bought or 1)
                + 0.15 * rating / 5
                + 0.10 * int(best_seller)
            )
            features = _build_features(row.category, subcategory, specs, rating, reviews, best_seller)
            writer.writerow({
                "product_id": row.asin,
                "title": row.title,
                "brand": brand,
                "category": row.category,
                "subcategory": subcategory,
                "description": _description(row, brand, row.category, subcategory, specs, rating, reviews),
                "features": json.dumps(features, ensure_ascii=True),
                "specifications": json.dumps(specs, ensure_ascii=True, sort_keys=True),
                "keywords/tags": _build_keywords(row.title, brand, row.category, subcategory, specs),
                "price": f"{float(row.price):.2f}",
                "rating": f"{rating:.1f}",
                "review_count": reviews,
                "popularity": f"{popularity:.2f}",
                "best_seller": best_seller,
                "availability": _availability(row, best_seller, seed),
                "image_url": row.img_url,
                "product_url": row.product_url,
            })
            category_counts[row.category] += 1
            domain_counts[domain] += 1
            brand_counts[brand] += 1

    print(f"Generated: {output_path}")
    print(f"Products: {len(enriched):,}")
    print(f"Categories: {len(category_counts):,}")
    print(f"Domains: {len(domain_counts):,}")
    print(f"Synthetic brands: {len(brand_counts):,}")
    print(f"Category sample range: {min(category_counts.values()):,}-{max(category_counts.values()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an intelligently enriched product sample CSV.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    generate(args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
