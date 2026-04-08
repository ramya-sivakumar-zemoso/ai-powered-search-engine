"""Generate a 10,000-product e-commerce dataset (data/ecommerce.json).

Each document has: id, title, description, category, subcategory, tags,
brand, price, currency, rating, review_count, in_stock, image_url.

Run:  python scripts/generate_ecommerce_data.py [--count 10000] [--out data/ecommerce.json]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Catalog data ──────────────────────────────────────────────────────────────

BLACK_DECKER = "Black+Decker"

CATEGORIES: dict[str, dict] = {
    "Electronics": {
        "subcategories": [
            "Smartphones", "Laptops", "Tablets", "Headphones", "Smartwatches",
            "Cameras", "Speakers", "Monitors", "Keyboards", "Mice",
            "USB Hubs", "Chargers", "Power Banks", "Webcams", "Microphones",
        ],
        "brands": [
            "Sony", "Samsung", "Apple", "LG", "Bose", "JBL", "Anker",
            "Logitech", "Dell", "HP", "Lenovo", "ASUS", "Razer", "HyperX",
            "Corsair", "SteelSeries", "Sennheiser", "Audio-Technica",
        ],
        "price_range": (9.99, 1999.99),
        "adjectives": [
            "Premium", "Ultra", "Pro", "Advanced", "Compact", "Wireless",
            "Portable", "High-Performance", "Noise-Cancelling", "4K",
            "Ergonomic", "Slim", "Foldable", "Gaming", "Professional",
        ],
    },
    "Clothing": {
        "subcategories": [
            "T-Shirts", "Jeans", "Jackets", "Dresses", "Hoodies",
            "Sneakers", "Boots", "Sandals", "Shorts", "Sweaters",
            "Activewear", "Formal Wear", "Underwear", "Socks", "Hats",
        ],
        "brands": [
            "Nike", "Adidas", "Puma", "Levi's", "H&M", "Zara", "Uniqlo",
            "Gap", "Under Armour", "North Face", "Patagonia", "Columbia",
            "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein",
        ],
        "price_range": (7.99, 299.99),
        "adjectives": [
            "Classic", "Slim-Fit", "Relaxed", "Vintage", "Modern",
            "Breathable", "Waterproof", "Stretch", "Organic Cotton",
            "Recycled", "Oversized", "Tailored", "Lightweight", "Thermal",
        ],
    },
    "Home & Kitchen": {
        "subcategories": [
            "Cookware", "Bakeware", "Blenders", "Coffee Makers", "Toasters",
            "Air Fryers", "Knife Sets", "Cutting Boards", "Storage Containers",
            "Dish Racks", "Towels", "Candles", "Vases", "Clocks", "Rugs",
        ],
        "brands": [
            "KitchenAid", "Cuisinart", "Instant Pot", "Ninja", "OXO",
            "Le Creuset", "Lodge", "Pyrex", "Rubbermaid", "Calphalon",
            "Breville", "Hamilton Beach", BLACK_DECKER, "Dyson",
        ],
        "price_range": (5.99, 499.99),
        "adjectives": [
            "Stainless Steel", "Non-Stick", "Ceramic", "BPA-Free",
            "Dishwasher-Safe", "Heat-Resistant", "Multi-Function",
            "Space-Saving", "Heavy-Duty", "Eco-Friendly", "Elegant",
            "Stackable", "Airtight", "Professional-Grade",
        ],
    },
    "Sports & Outdoors": {
        "subcategories": [
            "Yoga Mats", "Dumbbells", "Resistance Bands", "Running Shoes",
            "Cycling Gear", "Camping Tents", "Sleeping Bags", "Backpacks",
            "Water Bottles", "Fitness Trackers", "Jump Ropes", "Foam Rollers",
            "Hiking Poles", "Fishing Rods", "Soccer Balls",
        ],
        "brands": [
            "Nike", "Adidas", "Under Armour", "Coleman", "The North Face",
            "Patagonia", "Osprey", "CamelBak", "Yeti", "Hydro Flask",
            "Garmin", "Fitbit", "Manduka", "Bowflex", "TRX",
        ],
        "price_range": (8.99, 599.99),
        "adjectives": [
            "Lightweight", "Durable", "Foldable", "Adjustable", "Anti-Slip",
            "Quick-Dry", "UV-Protected", "Insulated", "Shock-Absorbing",
            "Ergonomic", "All-Weather", "Competition-Grade", "Reinforced",
        ],
    },
    "Books": {
        "subcategories": [
            "Fiction", "Non-Fiction", "Science Fiction", "Biography",
            "Self-Help", "Cookbooks", "History", "Science", "Technology",
            "Business", "Children's Books", "Mystery", "Romance", "Fantasy",
            "Graphic Novels",
        ],
        "brands": [
            "Penguin Random House", "HarperCollins", "Simon & Schuster",
            "Macmillan", "Hachette", "Scholastic", "Wiley", "O'Reilly",
            "Oxford University Press", "Cambridge University Press",
            "MIT Press", "No Starch Press", "Packt", "Manning",
        ],
        "price_range": (4.99, 79.99),
        "adjectives": [
            "Bestselling", "Award-Winning", "Illustrated", "Hardcover",
            "Paperback", "Revised Edition", "Complete", "Annotated",
            "Collector's Edition", "Pocket", "Deluxe", "Unabridged",
        ],
    },
    "Beauty & Personal Care": {
        "subcategories": [
            "Moisturizers", "Shampoos", "Conditioners", "Sunscreen",
            "Lipstick", "Foundation", "Mascara", "Perfume", "Deodorant",
            "Face Wash", "Serums", "Hair Oil", "Nail Polish", "Body Lotion",
            "Face Masks",
        ],
        "brands": [
            "L'Oreal", "Maybelline", "Nivea", "Dove", "Neutrogena",
            "CeraVe", "The Ordinary", "Olay", "Garnier", "Clinique",
            "Estée Lauder", "MAC", "NYX", "Bath & Body Works", "Aveeno",
        ],
        "price_range": (3.99, 149.99),
        "adjectives": [
            "Hydrating", "Anti-Aging", "Organic", "Paraben-Free",
            "Fragrance-Free", "Dermatologist-Tested", "SPF 50", "Vegan",
            "Cruelty-Free", "Nourishing", "Gentle", "Deep-Cleansing",
            "Long-Lasting", "Matte",
        ],
    },
    "Toys & Games": {
        "subcategories": [
            "Board Games", "Puzzles", "Building Sets", "Action Figures",
            "Dolls", "Remote Control Cars", "Card Games", "Educational Toys",
            "Outdoor Toys", "Plush Toys", "Science Kits", "Art Supplies",
            "Video Games", "Ride-On Toys", "Play Sets",
        ],
        "brands": [
            "LEGO", "Hasbro", "Mattel", "Fisher-Price", "Nintendo",
            "Ravensburger", "Melissa & Doug", "Hot Wheels", "Nerf",
            "Barbie", "Play-Doh", "Crayola", "VTech", "LeapFrog",
        ],
        "price_range": (4.99, 249.99),
        "adjectives": [
            "Interactive", "Educational", "Creative", "Collectible",
            "Award-Winning", "Classic", "Glow-in-the-Dark", "Magnetic",
            "Battery-Powered", "STEM", "Eco-Friendly", "Jumbo",
        ],
    },
    "Automotive": {
        "subcategories": [
            "Car Chargers", "Dash Cams", "Floor Mats", "Seat Covers",
            "Air Fresheners", "Tire Inflators", "Jump Starters", "Tool Kits",
            "Phone Mounts", "Trunk Organizers", "Wipers", "LED Bulbs",
            "Car Wash Kits", "Oil Filters", "Battery Chargers",
        ],
        "brands": [
            "Bosch", "Michelin", "3M", "WeatherTech", "Garmin", "Pioneer",
            "Kenwood", "Meguiar's", "Chemical Guys", "NOCO", "DeWalt",
            BLACK_DECKER, "Anker", "Scosche",
        ],
        "price_range": (5.99, 399.99),
        "adjectives": [
            "Universal-Fit", "Heavy-Duty", "Waterproof", "Anti-Slip",
            "12V", "Quick-Charge", "Ultra-Bright", "Scratch-Resistant",
            "All-Season", "Compact", "Professional", "OEM-Quality",
        ],
    },
    "Garden & Outdoor": {
        "subcategories": [
            "Garden Hoses", "Planters", "Seeds", "Pruning Shears",
            "Lawn Mowers", "Sprinklers", "Compost Bins", "Outdoor Furniture",
            "Grills", "Fire Pits", "Solar Lights", "Bird Feeders",
            "Raised Garden Beds", "Wheelbarrows", "Gloves",
        ],
        "brands": [
            "Scotts", "Miracle-Gro", "Fiskars", "Weber", "Traeger",
            BLACK_DECKER, "Sun Joe", "Greenworks", "Husqvarna",
            "DeWalt", "Gardena", "Keter", "Suncast", "Rain Bird",
        ],
        "price_range": (3.99, 799.99),
        "adjectives": [
            "Weather-Resistant", "Rust-Proof", "Solar-Powered", "Organic",
            "Adjustable", "Heavy-Duty", "Collapsible", "UV-Resistant",
            "Self-Watering", "Expandable", "Ergonomic", "Professional",
        ],
    },
    "Office Supplies": {
        "subcategories": [
            "Pens", "Notebooks", "Desk Organizers", "Staplers", "Tape",
            "Folders", "Whiteboards", "Markers", "Scissors", "Calculators",
            "Label Makers", "Paper Shredders", "Desk Lamps", "Chairs",
            "Standing Desks",
        ],
        "brands": [
            "Pilot", "Moleskine", "3M", "Post-it", "Sharpie", "BIC",
            "Brother", "HP", "Canon", "Staples", "Fellowes", "Swingline",
            "Herman Miller", "Steelcase", "FlexiSpot",
        ],
        "price_range": (1.99, 899.99),
        "adjectives": [
            "Ergonomic", "Recycled", "Refillable", "Quick-Dry",
            "Smudge-Proof", "Adjustable", "Ultra-Fine", "Heavy-Duty",
            "Compact", "Wireless", "USB-C", "Premium",
        ],
    },
}

COLORS = [
    "Black", "White", "Silver", "Gray", "Navy", "Blue", "Red", "Green",
    "Pink", "Gold", "Rose Gold", "Midnight Blue", "Forest Green", "Coral",
    "Teal", "Burgundy", "Charcoal", "Ivory", "Olive", "Sky Blue",
]

DESC_TEMPLATES = [
    "The {adj} {brand} {sub} delivers exceptional quality and value. "
    "Designed for everyday use, it features {feature1} and {feature2}. "
    "Whether you're at home or on the go, this product ensures reliability and comfort.",

    "Introducing the {adj} {sub} by {brand}. Built with {feature1}, "
    "this product stands out with its {feature2}. Perfect for anyone "
    "looking for a dependable, stylish option in the {cat} category.",

    "Upgrade your routine with the {brand} {adj} {sub}. "
    "It combines {feature1} with {feature2} for a seamless experience. "
    "Crafted with attention to detail, this product is built to last.",

    "The {brand} {sub} is a {adj} choice for discerning shoppers. "
    "Featuring {feature1} and {feature2}, it offers outstanding "
    "performance at a competitive price point in {cat}.",

    "Experience the difference with this {adj} {sub} from {brand}. "
    "Engineered with {feature1}, it provides {feature2} that sets it "
    "apart from the competition. A top pick in {cat}.",

    "Looking for the perfect {sub}? The {brand} {adj} edition has you covered. "
    "With {feature1} and {feature2}, it's designed to meet the highest "
    "standards of quality and usability.",

    "The {adj} {brand} {sub} combines modern design with practical functionality. "
    "Key highlights include {feature1} and {feature2}. "
    "Ideal for both beginners and experienced users in the {cat} space.",

    "Discover the {brand} {adj} {sub} — a versatile product that features "
    "{feature1} alongside {feature2}. Whether for personal or professional "
    "use, it delivers consistent results every time.",
]

FEATURES = [
    "premium build quality", "an intuitive design", "long-lasting durability",
    "a sleek modern finish", "easy-to-use controls", "energy-efficient operation",
    "advanced safety features", "quick setup and installation",
    "a comfortable ergonomic grip", "superior noise reduction",
    "fast charging capability", "eco-friendly materials",
    "a compact space-saving design", "high-resolution display",
    "smart connectivity options", "temperature regulation",
    "stain-resistant coating", "anti-microbial protection",
    "reinforced stitching", "shock-proof construction",
    "fade-resistant colors", "moisture-wicking fabric",
    "precision-engineered components", "a whisper-quiet motor",
    "an extra-long warranty", "scratch-resistant surfaces",
    "a travel-friendly form factor", "multi-device compatibility",
    "seamless Bluetooth pairing", "auto-shutoff for safety",
]


def _pick_tags(category: str, subcategory: str, adjective: str, brand: str) -> list[str]:
    """Build a 3-5 tag list from category metadata."""
    pool = [category, subcategory, brand, adjective]
    extra = random.choice([
        "best seller", "new arrival", "trending", "top rated",
        "editor's choice", "value pick", "limited edition", "everyday essential",
    ])
    pool.append(extra)
    random.shuffle(pool)
    return pool[:random.randint(3, 5)]


def generate_product(product_id: int) -> dict:
    """Generate a single realistic e-commerce product document."""
    cat_name = random.choice(list(CATEGORIES.keys()))
    cat = CATEGORIES[cat_name]

    sub = random.choice(cat["subcategories"])
    brand = random.choice(cat["brands"])
    adj = random.choice(cat["adjectives"])
    color = random.choice(COLORS)

    lo, hi = cat["price_range"]
    price = round(random.uniform(lo, hi), 2)

    title_formats = [
        f"{brand} {adj} {sub}",
        f"{adj} {sub} by {brand}",
        f"{brand} {sub} — {adj} Edition",
        f"{brand} {color} {sub}",
        f"{adj} {color} {sub} — {brand}",
    ]
    title = random.choice(title_formats)

    f1, f2 = random.sample(FEATURES, 2)
    desc = random.choice(DESC_TEMPLATES).format(
        adj=adj, brand=brand, sub=sub, cat=cat_name,
        feature1=f1, feature2=f2,
    )

    rating = round(random.triangular(1.0, 5.0, 4.2), 1)
    rating = min(5.0, max(1.0, rating))

    tags = _pick_tags(cat_name, sub, adj, brand)

    now = int(time.time())
    indexed_at = now - random.randint(0, 365 * 24 * 3600)

    return {
        "id": product_id,
        "title": title,
        "description": desc,
        "category": cat_name,
        "subcategory": sub,
        "tags": tags,
        "brand": brand,
        "price": price,
        "currency": "USD",
        "rating": rating,
        "review_count": random.randint(0, 12000),
        "in_stock": random.random() < 0.85,
        "color": color,
        "image_url": f"https://placehold.co/400x400?text={sub.replace(' ', '+')}",
        "indexed_at": indexed_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate e-commerce product dataset.")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--out", default="data/ecommerce.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count:,} products …")
    products = [generate_product(i + 1) for i in range(args.count)]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(products):,} products → {out_path}  ({size_mb:.1f} MB)")
    print(f"\nTo index:  python -m src.tools.setup_index --file {out_path} --schema ecommerce")


if __name__ == "__main__":
    main()
