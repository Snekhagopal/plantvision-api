"""Reference information for the 38 PlantVillage classes.

Content is deliberately conservative: it describes typical symptoms and
low-risk cultural actions. It never names chemical products or doses; those
decisions require confirmation and local, label-compliant advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FUNGAL, BACTERIAL, VIRAL, OOMYCETE, PEST, HEALTHY = (
    "Fungal disease", "Bacterial disease", "Viral disease",
    "Water-mould (oomycete) disease", "Pest damage (mites)", "Healthy",
)

CATEGORY_SANITATION = {
    FUNGAL: [
        "Remove and bag clearly affected leaves; do not compost them.",
        "Water at the soil line instead of over the leaves, ideally in the morning.",
        "Improve air circulation by spacing or lightly pruning crowded growth.",
    ],
    BACTERIAL: [
        "Avoid handling or pruning plants while foliage is wet - bacteria spread in water films.",
        "Remove badly affected leaves and disinfect pruning tools between plants.",
        "Avoid overhead irrigation and splashing water.",
    ],
    VIRAL: [
        "There is no cure for plant viruses; focus on confirming the cause and limiting spread.",
        "Wash hands and disinfect tools after handling suspect plants.",
        "Isolate or mark the plant until the diagnosis is confirmed.",
    ],
    OOMYCETE: [
        "Act quickly: this type of disease can spread rapidly in cool, wet weather.",
        "Remove and bag affected foliage; do not compost it.",
        "Keep foliage as dry as possible and improve airflow.",
    ],
    PEST: [
        "Inspect leaf undersides with a magnifying glass for tiny moving mites and fine webbing.",
        "Rinse leaf undersides with a strong spray of plain water to dislodge mites.",
        "Isolate potted plants so mites do not move to neighbouring plants.",
    ],
    HEALTHY: [
        "No treatment is indicated from this photo. Keep up regular care and monitoring.",
    ],
}


@dataclass(frozen=True)
class ConditionInfo:
    name: str
    category: str
    cause: str = ""
    symptoms: list[str] = field(default_factory=list)
    inspect: list[str] = field(default_factory=list)
    prevention: list[str] = field(default_factory=list)
    escalate: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.category == HEALTHY


def _healthy(plant: str) -> ConditionInfo:
    return ConditionInfo(
        name=f"Healthy {plant.lower()} leaf",
        category=HEALTHY,
        symptoms=["Leaf resembles healthy training examples (uniform colour, no lesions)."],
        inspect=["Check other leaves, stems and fruit - one healthy leaf does not clear the whole plant."],
        prevention=["Keep monitoring weekly, especially after wet or hot spells."],
    )


CONDITIONS: dict[str, ConditionInfo] = {
    "Apple___Apple_scab": ConditionInfo(
        "Apple scab", FUNGAL, "Fungus Venturia inaequalis",
        ["Olive-green to dark, velvety spots on leaves", "Corky, scabby spots on fruit", "Early leaf drop"],
        ["Look for velvety olive spots on both leaf surfaces and on young fruit."],
        ["Rake and remove fallen leaves in autumn to reduce overwintering spores.",
         "Prune to open the canopy; consider scab-resistant varieties for new plantings."],
        "If many leaves or fruit are affected, ask a local extension service about a seasonal management plan.",
    ),
    "Apple___Black_rot": ConditionInfo(
        "Apple black rot", FUNGAL, "Fungus Botryosphaeria obtusa",
        ["Leaf spots with purple margins and tan centres ('frog-eye')", "Rotting fruit, often from the blossom end",
         "Sunken cankers on branches"],
        ["Check branches for cankers and dead wood", "Look for shrivelled, mummified fruit on the tree or ground."],
        ["Prune out dead or cankered wood during dry weather.", "Remove mummified fruit from the tree and ground."],
        "Branch cankers on mature trees warrant advice from an arborist or extension service.",
    ),
    "Apple___Cedar_apple_rust": ConditionInfo(
        "Cedar apple rust", FUNGAL, "Fungus Gymnosporangium juniperi-virginianae",
        ["Bright yellow-orange spots on upper leaf surface", "Small tube-like structures on leaf undersides later in season"],
        ["Look for nearby junipers / eastern red cedars with orange, gelatinous galls in spring - the alternate host."],
        ["Choose rust-resistant apple varieties.", "Where practical, remove galls from nearby junipers."],
        "Consult a local expert if infection recurs every year.",
    ),
    "Apple___healthy": _healthy("Apple"),
    "Blueberry___healthy": _healthy("Blueberry"),
    "Cherry_(including_sour)___Powdery_mildew": ConditionInfo(
        "Cherry powdery mildew", FUNGAL, "Fungus Podosphaera clandestina",
        ["White powdery patches on leaves", "Curling or distorted young leaves"],
        ["Rub a patch gently - powdery mildew wipes off as a white powder."],
        ["Prune for airflow and sunlight penetration.", "Avoid excess nitrogen that drives soft new growth."],
        "Seek advice if fruit is affected or the problem recurs every season.",
    ),
    "Cherry_(including_sour)___healthy": _healthy("Cherry"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ConditionInfo(
        "Gray leaf spot", FUNGAL, "Fungus Cercospora zeae-maydis",
        ["Narrow, rectangular tan-to-gray lesions running between leaf veins"],
        ["Check whether lesions are confined between veins and start on lower leaves."],
        ["Rotate crops and manage infected residue.", "Use tolerant hybrids where available."],
        "For field crops, consult an agronomist before making treatment decisions.",
    ),
    "Corn_(maize)___Common_rust_": ConditionInfo(
        "Common rust", FUNGAL, "Fungus Puccinia sorghi",
        ["Small, powdery, brick-red to brown pustules on both leaf surfaces"],
        ["Rub a pustule - rust spores leave a reddish-brown powder on your finger."],
        ["Plant resistant hybrids.", "Monitor during cool, humid weather."],
        "For field crops, consult an agronomist if rust reaches upper leaves before grain fill.",
    ),
    "Corn_(maize)___Northern_Leaf_Blight": ConditionInfo(
        "Northern leaf blight", FUNGAL, "Fungus Exserohilum turcicum",
        ["Long, cigar-shaped gray-green to tan lesions (several cm long)"],
        ["Measure lesions: northern leaf blight lesions are typically much longer than other leaf spots."],
        ["Rotate crops and manage residue.", "Use resistant hybrids."],
        "For field crops, consult an agronomist when lesions appear before tasselling.",
    ),
    "Corn_(maize)___healthy": _healthy("Corn"),
    "Grape___Black_rot": ConditionInfo(
        "Grape black rot", FUNGAL, "Fungus Guignardia bidwellii",
        ["Tan circular leaf spots with dark borders and tiny black dots", "Berries shrivel into hard black 'mummies'"],
        ["Check developing berries and look for mummified fruit from last season."],
        ["Remove mummified berries and infected canes.", "Train and prune vines for good airflow."],
        "Consult a viticulture advisor if fruit clusters are affected.",
    ),
    "Grape___Esca_(Black_Measles)": ConditionInfo(
        "Esca (black measles)", FUNGAL, "Complex of wood-infecting fungi",
        ["'Tiger-stripe' interveinal yellowing or browning on leaves", "Dark spotting on berries", "Sudden vine collapse in hot weather"],
        ["Look at the trunk and cordons for dead wood; a cross-section may show dark streaking."],
        ["Prune during dry weather and protect large pruning wounds.", "Remove dead wood from the vineyard."],
        "Esca is a long-term trunk disease; consult a viticulture specialist for confirmation and management.",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ConditionInfo(
        "Isariopsis leaf spot (leaf blight)", FUNGAL, "Fungus Pseudocercospora vitis",
        ["Irregular dark brown spots, sometimes with a yellow edge", "Premature leaf drop"],
        ["Check older leaves first; spots often begin there."],
        ["Remove fallen infected leaves.", "Improve canopy airflow."],
        "Consult a local advisor if defoliation is significant.",
    ),
    "Grape___healthy": _healthy("Grape"),
    "Orange___Haunglongbing_(Citrus_greening)": ConditionInfo(
        "Huanglongbing (citrus greening)", BACTERIAL, "Bacterium 'Candidatus Liberibacter' spread by the Asian citrus psyllid",
        ["Blotchy, asymmetrical yellow mottling across leaf veins", "Small, lopsided, bitter fruit that stays partly green"],
        ["Compare both halves of the leaf: HLB mottling is typically asymmetrical, unlike most nutrient deficiencies.",
         "Look for small psyllid insects on new shoots."],
        ["Buy certified disease-free nursery stock.", "Monitor new flushes for psyllids."],
        "HLB is incurable and regulated in many regions. Contact your local agricultural authority or extension "
        "service for lab confirmation - do not move plant material.",
    ),
    "Peach___Bacterial_spot": ConditionInfo(
        "Peach bacterial spot", BACTERIAL, "Bacterium Xanthomonas arboricola pv. pruni",
        ["Small angular water-soaked spots turning purple-brown", "Spot centres fall out ('shot-hole')", "Spots or cracks on fruit"],
        ["Check for shot-holes and small sunken spots on fruit."],
        ["Choose tolerant varieties.", "Avoid excessive nitrogen; maintain tree vigour."],
        "Consult an extension service for an orchard management plan if fruit is affected.",
    ),
    "Peach___healthy": _healthy("Peach"),
    "Pepper,_bell___Bacterial_spot": ConditionInfo(
        "Bell pepper bacterial spot", BACTERIAL, "Xanthomonas bacteria",
        ["Small water-soaked spots that turn brown with yellow halos", "Raised scabby spots on fruit"],
        ["Check several leaves and fruit; spots are often more numerous on lower leaves."],
        ["Use disease-free seed and transplants.", "Rotate away from peppers and tomatoes for 2-3 seasons."],
        "Seek advice if spread is rapid or many plants are affected.",
    ),
    "Pepper,_bell___healthy": _healthy("Bell pepper"),
    "Potato___Early_blight": ConditionInfo(
        "Potato early blight", FUNGAL, "Fungus Alternaria solani",
        ["Brown spots with concentric rings ('target' pattern), often with yellow halo", "Starts on older, lower leaves"],
        ["Check whether lesions show concentric rings and begin on the lowest leaves."],
        ["Rotate crops.", "Maintain adequate plant nutrition and remove infected debris."],
        "Consult a local advisor if the disease moves rapidly up the plant.",
    ),
    "Potato___Late_blight": ConditionInfo(
        "Potato late blight", OOMYCETE, "Water mould Phytophthora infestans",
        ["Dark, water-soaked lesions that expand quickly", "White fuzzy growth on leaf undersides in humid weather",
         "Brown rot in tubers"],
        ["Check leaf undersides early in the morning for white growth at lesion edges.",
         "Check nearby tomatoes, which are also susceptible."],
        ["Use certified seed potatoes and destroy volunteer plants.", "Hill soil over tubers."],
        "Late blight can destroy crops within days and is reportable in some regions. Contact a local extension "
        "service promptly for confirmation.",
    ),
    "Potato___healthy": _healthy("Potato"),
    "Raspberry___healthy": _healthy("Raspberry"),
    "Soybean___healthy": _healthy("Soybean"),
    "Squash___Powdery_mildew": ConditionInfo(
        "Squash powdery mildew", FUNGAL, "Fungi such as Podosphaera xanthii",
        ["White, powdery spots that spread to cover leaves", "Leaves yellow and dry out"],
        ["Rub a patch gently - powdery mildew wipes off as a white powder."],
        ["Grow resistant varieties.", "Space plants for airflow and full sun."],
        "Usually manageable at home; seek advice if it appears early in the season every year.",
    ),
    "Strawberry___Leaf_scorch": ConditionInfo(
        "Strawberry leaf scorch", FUNGAL, "Fungus Diplocarpon earlianum",
        ["Many small purple spots that merge, making leaves look scorched"],
        ["Distinguish from drought scorch: fungal scorch shows discrete purple spots, drought scorch browns leaf edges evenly."],
        ["Remove old infected leaves after harvest.", "Renew plantings regularly and avoid overhead watering."],
        "Consult a local advisor for commercial plantings.",
    ),
    "Strawberry___healthy": _healthy("Strawberry"),
    "Tomato___Bacterial_spot": ConditionInfo(
        "Tomato bacterial spot", BACTERIAL, "Xanthomonas bacteria",
        ["Small, dark, water-soaked spots, sometimes with yellow halos", "Raised scabby spots on fruit"],
        ["Spots are usually small (under ~3 mm) and lack the concentric rings of early blight."],
        ["Use disease-free seed and transplants.", "Rotate away from tomatoes and peppers."],
        "Seek advice if many plants are affected.",
    ),
    "Tomato___Early_blight": ConditionInfo(
        "Tomato early blight", FUNGAL, "Fungus Alternaria (A. solani / A. linariae)",
        ["Brown spots with concentric 'target' rings on older leaves", "Yellowing around lesions", "Lower leaves die first"],
        ["Check that lesions start on the oldest leaves and show concentric rings."],
        ["Mulch to reduce soil splash.", "Stake plants and remove lower leaves touching the soil.", "Rotate crops."],
        "Consult a local advisor if defoliation progresses rapidly.",
    ),
    "Tomato___Late_blight": ConditionInfo(
        "Tomato late blight", OOMYCETE, "Water mould Phytophthora infestans",
        ["Large, greasy, gray-green to brown lesions", "White fuzzy growth on leaf undersides in humid weather",
         "Firm brown patches on fruit"],
        ["Check leaf undersides for white growth and look at stems for dark lesions.", "Check nearby potatoes."],
        ["Choose resistant varieties.", "Avoid wetting foliage and space plants for airflow."],
        "Late blight spreads rapidly and can affect neighbouring gardens. Contact a local extension service promptly.",
    ),
    "Tomato___Leaf_Mold": ConditionInfo(
        "Tomato leaf mould", FUNGAL, "Fungus Passalora fulva",
        ["Pale yellow patches on the upper leaf surface", "Olive-green to gray velvety mould on the underside"],
        ["Turn the leaf over: velvety olive growth underneath supports leaf mould."],
        ["Reduce humidity (ventilate greenhouses/tunnels).", "Avoid wetting foliage; space plants."],
        "Seek advice if it persists despite lowering humidity.",
    ),
    "Tomato___Septoria_leaf_spot": ConditionInfo(
        "Septoria leaf spot", FUNGAL, "Fungus Septoria lycopersici",
        ["Many small circular spots with dark borders and gray centres", "Tiny black dots inside spots", "Starts on lower leaves"],
        ["Use a magnifying glass to look for tiny black dots in spot centres."],
        ["Remove infected lower leaves.", "Mulch, stake and rotate crops."],
        "Consult a local advisor if it spreads to upper leaves despite sanitation.",
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": ConditionInfo(
        "Two-spotted spider mite damage", PEST, "Mite Tetranychus urticae",
        ["Fine yellow/white stippling on leaves", "Fine webbing, especially on undersides", "Bronzed or dry leaves"],
        ["Hold white paper under a leaf and tap it - moving specks suggest mites.",
         "Note conditions: mites thrive in hot, dry, dusty conditions."],
        ["Keep plants adequately watered (drought-stressed plants are more susceptible).",
         "Encourage natural predators by avoiding broad-spectrum sprays."],
        "Seek advice if infestation is heavy or affects many plants.",
    ),
    "Tomato___Target_Spot": ConditionInfo(
        "Tomato target spot", FUNGAL, "Fungus Corynespora cassiicola",
        ["Brown lesions with light centres and concentric rings", "Can affect leaves, stems and fruit"],
        ["Target spot can look very similar to early blight; laboratory confirmation may be needed."],
        ["Improve airflow and avoid prolonged leaf wetness.", "Remove crop debris after harvest."],
        "Consult a local advisor or diagnostic lab to distinguish it from early blight.",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ConditionInfo(
        "Tomato yellow leaf curl virus (TYLCV)", VIRAL, "Begomovirus transmitted by whiteflies",
        ["Upward curling, cupped leaves with yellow margins", "Stunted growth and flower drop"],
        ["Shake the plant gently and look for tiny white flying insects (whiteflies).",
         "Compare with nutrient or herbicide damage, which can cause similar curling."],
        ["Use whitefly-resistant or TYLCV-tolerant varieties.", "Use fine insect netting for seedlings."],
        "Confirmation needs a lab test; consult an extension service before removing plants.",
    ),
    "Tomato___Tomato_mosaic_virus": ConditionInfo(
        "Tomato mosaic virus (ToMV)", VIRAL, "Tobamovirus spread by contact, tools and hands",
        ["Light- and dark-green mosaic mottling", "Distorted or fern-like leaves", "Stunted growth"],
        ["Check new growth for mottling; compare with nutrient-deficiency yellowing patterns."],
        ["Wash hands and disinfect tools, especially after handling tobacco products.", "Use resistant varieties."],
        "Confirmation needs a lab test; consult an extension service.",
    ),
    "Tomato___healthy": _healthy("Tomato"),
}


def get_condition(label: str) -> ConditionInfo | None:
    return CONDITIONS.get(label)
