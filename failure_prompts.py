"""
failure_prompts.py
------------------
Pre-designed experiment inputs for the diffusion-editing failure survey.

Each method gets a list of experiments. Every experiment is a dict with:
    - id:            short unique string (used as output filename stem)
    - failure_type:  one of
                       "unnatural_prompt"     - physically impossible content
                       "attribute_conflict"   - multiple simultaneous edits
                       "spatial_complex"      - edit in crowded/cluttered scene
    - hypothesis:    what failure mode we expect to observe
    - (method-specific fields below)

These are the cases we analyze in the write-up, so keep the set small and
diverse rather than exhaustive.
"""

# ---------------------------------------------------------------------------
# MasaCtrl: consistent non-rigid editing via mutual self-attention control.
# Inputs are a source prompt (describes the generated source image) and a
# target prompt (the edited version). MasaCtrl is expected to preserve
# identity/appearance while changing pose/structure specified in the target.
# ---------------------------------------------------------------------------
MASACTRL_EXPERIMENTS = [
    {
        "id": "masactrl_three_eyes",
        "failure_type": "unnatural_prompt",
        "source_image_key": "portrait",
        "source_prompt": "a portrait of a person",
        "target_prompt": "a portrait of a person with three eyes on the face",
        "hypothesis": (
            "Mutual self-attention reuses the source's facial feature layout, "
            "so the model is expected to either drop the third eye entirely "
            "(identity wins) or paste a low-quality duplicate eye that "
            "disagrees with the preserved face geometry."
        ),
        "seed": 42,
    },
    {
        "id": "masactrl_floating_car",
        "failure_type": "unnatural_prompt",
        "source_image_key": "street",
        "source_prompt": "a busy city street with cars",
        "target_prompt": "a busy city street with cars floating above the buildings",
        "hypothesis": (
            "Structure-preserving attention anchors the cars to the ground plane "
            "of the source; the target's aerial position should collide with "
            "the preserved road layout and produce stretched or ghosted cars."
        ),
        "seed": 7,
    },
    {
        "id": "masactrl_hair_and_glasses",
        "failure_type": "attribute_conflict",
        "source_image_key": "portrait",
        "source_prompt": "a portrait of a person",
        "target_prompt": "a portrait of a person with long blonde hair wearing round glasses",
        "hypothesis": (
            "Two simultaneous edits (hair length+color AND glasses) compete for "
            "the same self-attention map; we expect partial success on one "
            "attribute and failure on the other, or 'ghost' glasses fused into "
            "the hair region."
        ),
        "seed": 123,
    },
    {
        "id": "masactrl_crowded_pose",
        "failure_type": "spatial_complex",
        "source_image_key": "street",
        "source_prompt": "a crowded city street with people walking",
        "target_prompt": "a crowded city street where a person is jumping in the middle",
        "hypothesis": (
            "Distractors in the crowded background create spurious self-attention "
            "matches. MasaCtrl is expected to leak the jumping pose into "
            "neighboring people or fail to change pose at all while preserving "
            "background identity."
        ),
        "seed": 2024,
    },
]

# ---------------------------------------------------------------------------
# PnP (Plug-and-Play Diffusion): text-driven translation by injecting spatial
# features / self-attention from an inverted source image. Needs a real input
# image to invert, then a target prompt.
# ---------------------------------------------------------------------------
PNP_EXPERIMENTS = [
    {
        "id": "pnp_portrait_to_alien",
        "failure_type": "unnatural_prompt",
        "source_image_key": "portrait",
        "source_prompt": "a photo of a person",
        "target_prompt": "a photo of an alien with four eyes and green skin",
        "hypothesis": (
            "Feature injection forces the alien output to follow the human "
            "facial topology; the extra eyes should appear as texture blobs "
            "on the forehead rather than as geometrically new features."
        ),
        "seed": 1,
    },
    {
        "id": "pnp_street_to_underwater",
        "failure_type": "unnatural_prompt",
        "source_image_key": "street",
        "source_prompt": "a photo of a city street",
        "target_prompt": "an underwater coral reef street with fish instead of cars",
        "hypothesis": (
            "Spatial feature maps encode 'street' layout. Replacing cars with "
            "fish under the same structure should produce fish-shaped cars or "
            "cars with coral texture rather than a clean semantic swap."
        ),
        "seed": 2,
    },
    {
        "id": "pnp_indoor_multi_edit",
        "failure_type": "attribute_conflict",
        "source_image_key": "indoor",
        "source_prompt": "a photo of a living room",
        "target_prompt": "a snowy outdoor forest with a wooden cabin at night",
        "hypothesis": (
            "The target changes scene type, illumination, and time-of-day "
            "simultaneously. Structure guidance from the interior should "
            "cause the forest to inherit wall/furniture boundaries, i.e. "
            "structure breakdown."
        ),
        "seed": 3,
    },
    {
        "id": "pnp_street_crowd_swap",
        "failure_type": "spatial_complex",
        "source_image_key": "street",
        "source_prompt": "a photo of a crowded city street",
        "target_prompt": "a crowded city street where every person is a robot",
        "hypothesis": (
            "Many small subjects share injected features; we expect per-person "
            "identity bleed, with some people only partially converted to "
            "robots."
        ),
        "seed": 4,
    },
]

# ---------------------------------------------------------------------------
# DragonDiffusion: drag-based editing via feature correspondence.
# Each experiment specifies handle points (source) and target points in
# normalized [0,1] image coordinates, plus an optional mask region.
# ---------------------------------------------------------------------------
DRAGON_EXPERIMENTS = [
    {
        "id": "dragon_face_stretch",
        "failure_type": "unnatural_prompt",
        "source_image_key": "portrait",
        "prompt": "a portrait of a person",
        "handle_points": [(0.50, 0.40)],   # tip of nose
        "target_points": [(0.50, 0.10)],   # way above the head
        "mask_box": (0.30, 0.20, 0.70, 0.60),
        "hypothesis": (
            "Pulling the nose far out of the face region forces the feature "
            "correspondence to extrapolate; we expect a distorted, "
            "smeared face or a duplicated nose artifact."
        ),
        "seed": 10,
    },
    {
        "id": "dragon_hair_and_shoulder",
        "failure_type": "attribute_conflict",
        "source_image_key": "portrait",
        "prompt": "a portrait of a person",
        "handle_points": [(0.40, 0.30), (0.70, 0.75)],   # hairline + shoulder
        "target_points": [(0.35, 0.20), (0.80, 0.70)],   # lift hair + widen shoulder
        "mask_box": (0.20, 0.10, 0.90, 0.90),
        "hypothesis": (
            "Two simultaneous drags in disjoint regions share the same feature "
            "optimization; background between them is expected to warp or "
            "corrupt even though it was not selected."
        ),
        "seed": 11,
    },
    {
        "id": "dragon_move_person_in_crowd",
        "failure_type": "spatial_complex",
        "source_image_key": "street",
        "prompt": "a crowded street scene",
        "handle_points": [(0.50, 0.60)],   # a person in the middle
        "target_points": [(0.25, 0.60)],   # move to the left, over other people
        "mask_box": (0.10, 0.40, 0.60, 0.90),
        "hypothesis": (
            "Dragging a subject across overlapping people should cause "
            "identity merging in the destination region and a duplicated / "
            "'hollow' figure in the origin region."
        ),
        "seed": 12,
    },
    {
        "id": "dragon_move_object_indoor",
        "failure_type": "spatial_complex",
        "source_image_key": "indoor",
        "prompt": "a living room",
        "handle_points": [(0.30, 0.70)],
        "target_points": [(0.70, 0.40)],
        "mask_box": (0.10, 0.30, 0.90, 0.90),
        "hypothesis": (
            "Moving an object through a cluttered indoor scene will likely "
            "leave a hole or a ghost of the original furniture and bleed "
            "texture into the destination region."
        ),
        "seed": 13,
    },
]


def all_experiments():
    """Flat iterator over every experiment, tagged with its method name."""
    for exp in MASACTRL_EXPERIMENTS:
        yield "masactrl", exp
    for exp in PNP_EXPERIMENTS:
        yield "pnp", exp
    for exp in DRAGON_EXPERIMENTS:
        yield "dragon", exp


if __name__ == "__main__":
    for method, exp in all_experiments():
        print(f"[{method:9s}] {exp['id']:35s}  {exp['failure_type']}")
