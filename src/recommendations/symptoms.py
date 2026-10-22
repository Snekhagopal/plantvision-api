"""Conservative, plant-agnostic advice for each symptom pattern of the symptom screen.

The pattern says what the leaf looks like, not which disease causes it, so
advice is limited to low-risk inspection, hygiene and cultural steps.
"""

SYMPTOM_ADVICE = {
    "healthy": {
        "actions": ["No obvious disease pattern was found on this leaf. Keep up normal care and monitor weekly."],
        "inspect": ["Check other leaves, especially older ones and leaf undersides, for problems not visible in "
                    "this photo."],
        "prevention": [],
    },
    "spots": {
        "actions": ["Remove the worst-affected leaves and bin them (do not compost) to reduce spread.",
                    "Water at the base of the plant, in the morning, so leaves dry quickly."],
        "inspect": ["Are the spots dry and papery (often fungal) or dark and water-soaked with a yellow halo "
                    "(often bacterial)?",
                    "Do spots have rings, a dark border, or tiny black dots in the centre (fungal fruiting bodies)?",
                    "Are spots spreading to new leaves, or only on old, lower leaves?",
                    "Rule out non-disease causes: sun scorch, cold, fertiliser burn or spray damage also cause "
                    "brown patches."],
        "prevention": ["Improve air flow: space plants and thin dense growth.",
                       "Clean up fallen leaves; disinfect pruning tools between plants."],
    },
    "mildew": {
        "actions": ["Remove the most heavily coated leaves and bin them.",
                    "Improve air circulation and move the plant to brighter light if possible."],
        "inspect": ["Rub the coating gently: powdery mildew wipes off as a white powder; dust or residue does "
                    "not come back.",
                    "Check new growth and buds, where mildew often starts.",
                    "Downy mildew looks different: yellow patches on top with grey-purple fuzz underneath."],
        "prevention": ["Avoid crowding and excess nitrogen fertiliser, which produces soft, susceptible growth.",
                       "Water the soil, not the leaves."],
    },
    "rust": {
        "actions": ["Remove affected leaves and bin them; rust spores spread easily on hands and tools.",
                    "Keep foliage dry and improve air flow."],
        "inspect": ["Look at the leaf underside: rust usually forms raised orange, brown or yellow pustules that "
                    "leave coloured powder on a tissue.",
                    "Some rusts alternate between two host plants (e.g. juniper and apple); note nearby plants."],
        "prevention": ["Clear fallen leaves at the end of the season.",
                       "Choose rust-resistant varieties when replanting."],
    },
    "mottling": {
        "actions": ["Isolate the plant if it is near others of the same kind until the cause is known: some "
                    "viruses spread by insects, tools or hands.",
                    "Do not fertilise heavily as a fix; mottled yellowing has many causes."],
        "inspect": ["Mosaic (patchy light/dark green), crinkled or curled new leaves point towards a virus; check "
                    "for aphids, whiteflies or thrips, which spread them.",
                    "Yellowing between green veins on new leaves suggests iron or manganese issues (often high "
                    "soil pH); on old leaves it suggests magnesium or nitrogen.",
                    "Check roots and drainage: waterlogged or root-bound plants often yellow unevenly.",
                    "A soil test is the reliable way to confirm a nutrient problem."],
        "prevention": ["Control sap-sucking insects early, and disinfect tools after handling suspect plants.",
                       "Keep watering consistent and make sure pots drain freely."],
    },
}
