"""General care guidance for popular plants, grouped by plant type.

This is reference content, not model output: it lets the app give useful,
plant-appropriate inspection and care guidance for plants the disease
classifier was never trained on. Guidance is deliberately generic at the
group level; plant-specific notes are limited to well-established facts.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.plants.catalog import normalize_name


@dataclass(frozen=True)
class CareGroup:
    label: str
    watering: str
    light: str
    pests: str
    problems: str


GROUPS: dict[str, CareGroup] = {
    "tropical_foliage": CareGroup(
        "Tropical foliage houseplant",
        "Water when the top 2-5 cm of soil is dry; empty saucers so roots never sit in water.",
        "Bright, indirect light; harsh direct sun can scorch leaves.",
        "Spider mites, mealybugs, scale insects, thrips, fungus gnats.",
        "Root rot from overwatering, leaf spots, brown tips from dry air or salt build-up.",
    ),
    "succulent": CareGroup(
        "Succulent / cactus",
        "Water thoroughly only when the soil is completely dry; water much less in winter.",
        "Bright light with some direct sun; stretched, pale growth means too little light.",
        "Mealybugs (white cottony tufts), scale insects, spider mites.",
        "Rot from overwatering (soft, translucent or blackened tissue), sunburn after sudden moves into sun.",
    ),
    "orchid": CareGroup(
        "Orchid",
        "Water when the potting mix is nearly dry and the roots look silvery; never leave water in the crown.",
        "Bright, indirect light.",
        "Mealybugs, scale insects, aphids, thrips, spider mites.",
        "Root and crown rot, bud drop after sudden changes in temperature or light.",
    ),
    "fern": CareGroup(
        "Fern",
        "Keep the soil evenly moist but not soggy; ferns dislike drying out.",
        "Indirect light or shade.",
        "Scale insects, mealybugs, spider mites in dry air.",
        "Brown, crispy fronds from low humidity or dry soil; root rot if waterlogged.",
    ),
    "herb_mediterranean": CareGroup(
        "Mediterranean herb",
        "Let the soil dry between waterings; these herbs dislike wet roots.",
        "Full sun (6+ hours).",
        "Aphids, spider mites, whiteflies.",
        "Root rot in wet or poorly drained soil, powdery mildew in humid, still air.",
    ),
    "herb_soft": CareGroup(
        "Leafy herb",
        "Keep the soil evenly moist but not soggy.",
        "Full sun to light shade.",
        "Aphids, whiteflies, leaf miners, slugs and snails, caterpillars.",
        "Downy or powdery mildew, rust, wilting in heat, bolting (flowering) in hot weather.",
    ),
    "fruiting_vegetable": CareGroup(
        "Fruiting vegetable",
        "Water deeply and consistently at the base; mulch to keep moisture even.",
        "Full sun (6-8+ hours).",
        "Aphids, whiteflies, thrips, spider mites, caterpillars (e.g. fruit borers, hornworms).",
        "Leaf spots and blights, wilts, viral leaf curl or mosaic, blossom-end rot from irregular watering.",
    ),
    "cucurbit": CareGroup(
        "Cucurbit (gourd family)",
        "Water deeply at the base, keeping leaves dry; consistent moisture during fruiting.",
        "Full sun.",
        "Aphids, whiteflies, cucumber beetles, squash bugs, fruit flies, spider mites, vine borers.",
        "Powdery and downy mildew, bacterial wilt, mosaic viruses.",
    ),
    "leafy_vegetable": CareGroup(
        "Leafy / cabbage-family vegetable",
        "Keep the soil consistently moist.",
        "Full sun to partial shade; most prefer cool weather.",
        "Aphids, caterpillars (e.g. cabbage worms, diamondback moth), flea beetles, slugs and snails.",
        "Downy mildew, leaf spots, black rot in brassicas, bolting in heat.",
    ),
    "root_vegetable": CareGroup(
        "Root, bulb or tuber crop",
        "Keep soil evenly moist; irregular watering can cause cracking or misshapen roots.",
        "Full sun.",
        "Aphids, thrips (onions), flea beetles, wireworms, root maggots.",
        "Leaf blights, bulb or root rots in waterlogged soil.",
    ),
    "legume": CareGroup(
        "Legume (bean / pea family)",
        "Water consistently, especially during flowering and pod fill; avoid waterlogging.",
        "Full sun.",
        "Aphids, whiteflies, spider mites, pod borers, bean beetles, leaf miners.",
        "Rust, powdery mildew, anthracnose, bacterial blight, yellow mosaic viruses.",
    ),
    "fruit_tree": CareGroup(
        "Fruit tree",
        "Water deeply and infrequently; young trees need more regular water than established ones.",
        "Full sun.",
        "Aphids, scale insects, mealybugs, mites, caterpillars, borers, fruit flies.",
        "Leaf spots, scab, rusts, powdery mildew, anthracnose, cankers.",
    ),
    "citrus": CareGroup(
        "Citrus",
        "Water deeply when the top few centimetres are dry; citrus needs good drainage.",
        "Full sun.",
        "Citrus psyllid, scale insects, mealybugs, leaf miners, aphids, mites.",
        "Citrus greening (HLB), canker, sooty mould, root rot; yellow leaves often relate to watering or nutrition.",
    ),
    "berry_vine": CareGroup(
        "Berry or vine fruit",
        "Keep soil consistently moist and mulched; water at the base.",
        "Full sun.",
        "Aphids, spider mites, beetles, fruit flies (e.g. spotted-wing drosophila), birds.",
        "Powdery and downy mildew, grey mould (botrytis), leaf spots, anthracnose.",
    ),
    "tropical_fruit": CareGroup(
        "Tropical fruit plant",
        "Needs regular, generous water in warm weather but good drainage.",
        "Full sun and warmth; protect from cold and strong wind.",
        "Aphids, mealybugs, scale insects, mites, weevils, fruit flies.",
        "Leaf spots, wilts, root rot, viral diseases spread by aphids or whiteflies.",
    ),
    "rose": CareGroup(
        "Rose",
        "Water deeply at the base 1-2 times a week, more in heat; keep leaves dry.",
        "Full sun (6+ hours).",
        "Aphids, spider mites, thrips, Japanese beetles, rose sawfly larvae.",
        "Black spot, powdery mildew, rust, downy mildew.",
    ),
    "garden_flower": CareGroup(
        "Flowering ornamental",
        "Water at the base when the top of the soil is dry.",
        "Mostly full sun; some (e.g. begonias, impatiens) prefer partial shade.",
        "Aphids, whiteflies, thrips, spider mites, mealybugs, caterpillars, slugs.",
        "Powdery mildew, grey mould (botrytis), leaf spots, root rot.",
    ),
    "tree_shrub": CareGroup(
        "Tree or shrub",
        "Water deeply and infrequently; established plants are fairly drought tolerant.",
        "Full sun to partial shade depending on species.",
        "Scale insects, aphids, mealybugs, caterpillars, borers.",
        "Leaf spots, powdery mildew, cankers, root rot in waterlogged soil.",
    ),
    "field_crop": CareGroup(
        "Field crop",
        "Irrigation needs depend strongly on crop stage and local conditions - follow local guidance.",
        "Full sun.",
        "Aphids, stem borers, armyworms and other caterpillars, leafhoppers and planthoppers, whiteflies.",
        "Rusts, leaf blights, smuts, blast and wilts; consult a local agronomist for field-scale problems.",
    ),
}


@dataclass(frozen=True)
class PlantProfile:
    name: str
    group: str
    aliases: tuple[str, ...] = ()
    note: str = ""

    @property
    def care(self) -> CareGroup:
        return GROUPS[self.group]


def _p(name, group, aliases=(), note=""):
    return PlantProfile(name, group, tuple(aliases), note)


PROFILES: list[PlantProfile] = [
    # --- Houseplants ---
    _p("Monstera", "tropical_foliage", ["swiss cheese plant", "monstera deliciosa"],
       "Needs a well-drained but moisture-retaining mix; yellow lower leaves are often a sign of overwatering."),
    _p("Pothos", "tropical_foliage", ["money plant", "devil's ivy", "devils ivy", "epipremnum"],
       "Tolerates lower light; yellowing leaves are commonly linked to overwatering."),
    _p("Philodendron", "tropical_foliage"),
    _p("Peace lily", "tropical_foliage", ["spathiphyllum"],
       "Droops visibly when dry and usually recovers within hours of watering; brown tips often come from dry air or salts."),
    _p("Snake plant", "succulent", ["sansevieria", "mother in law's tongue", "mother-in-law's tongue"],
       "Very drought tolerant; overwatering and root rot are the most common problems."),
    _p("ZZ plant", "succulent", ["zamioculcas", "zz"], "Stores water in thick rhizomes; let the soil dry out fully."),
    _p("Spider plant", "tropical_foliage", ["chlorophytum"],
       "Brown tips are common and often linked to dry air or minerals in tap water."),
    _p("Rubber plant", "tropical_foliage", ["ficus elastica", "rubber tree"]),
    _p("Fiddle leaf fig", "tropical_foliage", ["ficus lyrata"],
       "Often drops leaves after being moved or after sudden changes in light or temperature."),
    _p("Areca palm", "tropical_foliage", ["butterfly palm"]),
    _p("Chinese evergreen", "tropical_foliage", ["aglaonema"]),
    _p("Dieffenbachia", "tropical_foliage", ["dumb cane"]),
    _p("Calathea", "tropical_foliage", ["prayer plant", "maranta"],
       "Sensitive to dry air and minerals in tap water; crispy edges are common."),
    _p("Croton", "tropical_foliage", ["codiaeum"], "Prone to spider mites in dry indoor air."),
    _p("Dracaena", "tropical_foliage", ["corn plant", "dragon tree"]),
    _p("Lucky bamboo", "tropical_foliage", ["dracaena sanderiana"],
       "Not a true bamboo; yellowing is often linked to chlorinated water or too much sun."),
    _p("Anthurium", "tropical_foliage", ["flamingo flower"]),
    _p("English ivy", "tropical_foliage", ["hedera", "ivy"], "Prone to spider mites in warm, dry rooms."),
    _p("Arrowhead plant", "tropical_foliage", ["syngonium"]),
    _p("African violet", "tropical_foliage", ["saintpaulia"],
       "Cold water on the leaves causes pale spots; water at the soil or from below."),
    _p("Begonia", "garden_flower", [], "Prone to powdery mildew; avoid wetting the leaves."),
    _p("Boston fern", "fern", ["nephrolepis"]),
    _p("Bird's nest fern", "fern", ["asplenium", "birds nest fern"]),
    _p("Jade plant", "succulent", ["crassula", "money tree succulent"]),
    _p("Aloe vera", "succulent", ["aloe"]),
    _p("Cactus", "succulent", ["cacti"]),
    _p("Echeveria", "succulent"),
    _p("String of pearls", "succulent", ["senecio rowleyanus"]),
    _p("Christmas cactus", "succulent", ["schlumbergera", "holiday cactus"],
       "A forest cactus: it needs more water and less direct sun than desert cacti."),
    _p("Orchid", "orchid", ["phalaenopsis", "moth orchid"],
       "Most common orchids grow in bark, not soil; keep water out of the crown."),
    # --- Herbs ---
    _p("Basil", "herb_soft", ["sweet basil"],
       "Basil downy mildew causes yellowing leaves with dark, fuzzy growth underneath."),
    _p("Holy basil", "herb_soft", ["tulsi", "tulasi"], "Pinch off flower spikes to keep it bushy."),
    _p("Mint", "herb_soft", ["pudina", "peppermint", "spearmint"],
       "Spreads aggressively; rust shows as orange pustules under the leaves."),
    _p("Coriander", "herb_soft", ["cilantro", "dhania"], "Bolts (flowers) quickly in hot weather."),
    _p("Parsley", "herb_soft"),
    _p("Dill", "herb_soft"),
    _p("Chives", "herb_soft"),
    _p("Lemongrass", "herb_soft"),
    _p("Rosemary", "herb_mediterranean"),
    _p("Thyme", "herb_mediterranean"),
    _p("Oregano", "herb_mediterranean"),
    _p("Sage", "herb_mediterranean"),
    _p("Lavender", "herb_mediterranean", [], "Dislikes wet soil and humidity; root rot is the main risk."),
    _p("Curry leaf", "tree_shrub", ["curry leaves", "kadi patta", "murraya koenigii"],
       "Prone to scale insects and psyllids; water sparingly in cool weather."),
    _p("Fenugreek", "leafy_vegetable", ["methi"]),
    # --- Vegetables (disease-model species included) ---
    _p("Tomato", "fruiting_vegetable", [], "Irregular watering can cause blossom-end rot on the fruit."),
    _p("Potato", "root_vegetable", [], "Late blight can spread very fast in cool, wet weather."),
    _p("Bell pepper", "fruiting_vegetable"),
    _p("Chili pepper", "fruiting_vegetable", ["chilli", "chili", "green chilli", "hot pepper"],
       "Leaf curl (often virus spread by whiteflies or thrips) is a common problem."),
    _p("Eggplant", "fruiting_vegetable", ["brinjal", "aubergine", "baingan"],
       "Shoot and fruit borer caterpillars are a major pest in South Asia."),
    _p("Okra", "fruiting_vegetable", ["lady finger", "ladies finger", "bhindi"],
       "Yellow vein mosaic virus, spread by whiteflies, is common in warm regions."),
    _p("Cucumber", "cucurbit"),
    _p("Zucchini", "cucurbit", ["courgette"]),
    _p("Squash", "cucurbit"),
    _p("Pumpkin", "cucurbit"),
    _p("Watermelon", "cucurbit"),
    _p("Muskmelon", "cucurbit", ["cantaloupe", "melon"]),
    _p("Bitter gourd", "cucurbit", ["karela", "bitter melon"]),
    _p("Bottle gourd", "cucurbit", ["lauki", "calabash"]),
    _p("Lettuce", "leafy_vegetable"),
    _p("Spinach", "leafy_vegetable", ["palak"]),
    _p("Cabbage", "leafy_vegetable"),
    _p("Cauliflower", "leafy_vegetable"),
    _p("Broccoli", "leafy_vegetable"),
    _p("Kale", "leafy_vegetable"),
    _p("Carrot", "root_vegetable"),
    _p("Radish", "root_vegetable", ["mooli"]),
    _p("Beetroot", "root_vegetable", ["beet"]),
    _p("Onion", "root_vegetable"),
    _p("Garlic", "root_vegetable"),
    _p("Ginger", "root_vegetable", [], "Rhizome rot is common in waterlogged soil."),
    _p("Turmeric", "root_vegetable", ["haldi"]),
    _p("Sweet potato", "root_vegetable"),
    _p("Green beans", "legume", ["beans", "french beans", "string beans"]),
    _p("Peas", "legume", ["green peas", "garden pea"]),
    _p("Chickpea", "legume", ["chana", "garbanzo"]),
    _p("Soybean", "legume"),
    _p("Peanut", "legume", ["groundnut"]),
    # --- Fruits (disease-model species included) ---
    _p("Apple", "fruit_tree"),
    _p("Peach", "fruit_tree"),
    _p("Cherry (incl. sour cherry)", "fruit_tree"),
    _p("Pear", "fruit_tree"),
    _p("Plum", "fruit_tree"),
    _p("Apricot", "fruit_tree"),
    _p("Mango", "fruit_tree", ["aam"],
       "Powdery mildew and anthracnose commonly affect flowers and young leaves."),
    _p("Guava", "fruit_tree", ["amrood"], "Watch for fruit flies and mealybugs."),
    _p("Pomegranate", "fruit_tree", ["anar"]),
    _p("Fig", "fruit_tree", ["anjeer"]),
    _p("Avocado", "fruit_tree", [], "Very sensitive to waterlogged soil and root rot."),
    _p("Olive", "fruit_tree"),
    _p("Jackfruit", "tropical_fruit"),
    _p("Lychee", "tropical_fruit", ["litchi"]),
    _p("Orange", "citrus", [], "Citrus greening (HLB) is regulated in many regions - report suspected cases."),
    _p("Lemon", "citrus", ["nimbu"]),
    _p("Lime", "citrus", ["key lime"]),
    _p("Grapefruit", "citrus"),
    _p("Mandarin", "citrus", ["tangerine", "clementine", "kinnow"]),
    _p("Grape", "berry_vine"),
    _p("Strawberry", "berry_vine"),
    _p("Blueberry", "berry_vine", [], "Needs acidic soil; yellow leaves with green veins can indicate high soil pH."),
    _p("Raspberry", "berry_vine"),
    _p("Blackberry", "berry_vine"),
    _p("Kiwi", "berry_vine", ["kiwifruit"]),
    _p("Banana", "tropical_fruit", [], "Needs plenty of water and warmth; protect from strong wind."),
    _p("Papaya", "tropical_fruit", [], "Very sensitive to waterlogging; ringspot virus is spread by aphids."),
    _p("Pineapple", "tropical_fruit"),
    _p("Coconut", "tropical_fruit", ["coconut palm"]),
    _p("Dragon fruit", "succulent", ["pitaya"], "A climbing cactus: needs support and well-drained soil."),
    # --- Flowers ---
    _p("Rose", "rose", ["gulab"]),
    _p("Hibiscus", "garden_flower", ["gudhal", "china rose"],
       "Prone to aphids, whiteflies and mealybugs; bud drop is often linked to watering or temperature swings."),
    _p("Jasmine", "garden_flower", ["mogra", "chameli"]),
    _p("Marigold", "garden_flower", ["genda"]),
    _p("Sunflower", "garden_flower"),
    _p("Chrysanthemum", "garden_flower", ["mums", "guldaudi"]),
    _p("Dahlia", "garden_flower"),
    _p("Petunia", "garden_flower"),
    _p("Zinnia", "garden_flower", [], "Very prone to powdery mildew; water at the base."),
    _p("Geranium", "garden_flower", ["pelargonium"]),
    _p("Tulip", "garden_flower"),
    _p("Lily", "garden_flower"),
    _p("Daisy", "garden_flower"),
    _p("Gerbera", "garden_flower", ["gerbera daisy"]),
    _p("Carnation", "garden_flower"),
    _p("Daffodil", "garden_flower", ["narcissus"]),
    _p("Hydrangea", "garden_flower"),
    _p("Bougainvillea", "garden_flower", [], "Flowers best when kept on the dry side."),
    _p("Periwinkle", "garden_flower", ["vinca", "sadabahar"]),
    _p("Impatiens", "garden_flower"),
    # --- Trees and shrubs ---
    _p("Neem", "tree_shrub", ["azadirachta indica"], "Hardy and drought tolerant once established."),
    _p("Moringa", "tree_shrub", ["drumstick tree", "drumstick"], "Drought tolerant; avoid waterlogged soil."),
    _p("Bamboo", "tree_shrub"),
    _p("Maple", "tree_shrub"),
    _p("Oak", "tree_shrub"),
    _p("Boxwood", "tree_shrub", ["buxus"]),
    _p("Coffee", "tree_shrub", ["coffee plant"], "Coffee leaf rust shows as orange powdery spots under the leaves."),
    _p("Tea", "tree_shrub", ["tea plant", "camellia sinensis"]),
    # --- Field crops ---
    _p("Corn (maize)", "field_crop"),
    _p("Rice", "field_crop", ["paddy"], "Rice blast and planthoppers are major problems; consult local agronomists."),
    _p("Wheat", "field_crop", [], "Rusts (orange, brown or yellow pustules) are common; consult local agronomists."),
    _p("Barley", "field_crop"),
    _p("Millet", "field_crop", ["bajra", "ragi", "jowar", "sorghum"]),
    _p("Sugarcane", "field_crop"),
    _p("Cotton", "field_crop", [], "Whiteflies can spread leaf curl virus; bollworms damage the bolls."),
    _p("Mustard", "field_crop", ["sarson", "rapeseed"]),
]

_LOOKUP: dict[str, PlantProfile] = {}
for _profile in PROFILES:
    for _key in (_profile.name, *_profile.aliases):
        _LOOKUP.setdefault(normalize_name(_key), _profile)


def find_profile(name: str) -> PlantProfile | None:
    return _LOOKUP.get(normalize_name(name))


def profile_names() -> list[str]:
    return sorted(p.name for p in PROFILES)
