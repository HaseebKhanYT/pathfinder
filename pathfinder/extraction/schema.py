"""Fixed JSON schema for one pathology report.

The LLM fills `ReportExtraction` through strict structured output (Nebius's
OpenAI-compatible API), so every field is always present; a field that does not apply to the report's
organ is `null` (or `"not_reported"` for status enums). Fields are grouped by
the report section they come from: the header line, the SPECIMEN line, and the
FINAL DIAGNOSIS block.

Derived values that need arithmetic (tumor size in mm, age bounds) are NOT
asked of the model. `postprocess.derive` computes them in code from the raw
fields, so a unit conversion can never be hallucinated.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["present", "not_identified", "not_reported"]
Receptor = Literal["positive", "low_positive", "negative", "equivocal", "not_reported"]


class Header(BaseModel):
    """`Patient: F, age band 60-69 | Report year: 2024`."""

    sex: Optional[Literal["M", "F"]] = Field(description="Patient sex from the header line.")
    age_band: Optional[str] = Field(description="Age band exactly as written, e.g. '60-69' or '90+'.")
    report_year: Optional[int] = Field(description="Report year from the header line.")


class Specimen(BaseModel):
    """`SPECIMEN: Breast, left, total mastectomy`."""

    organ: Literal["breast", "colon", "lung", "prostate", "stomach", "kidney", "skin", "other"] = Field(
        description="Primary organ of the specimen."
    )
    site: Optional[str] = Field(
        description="Anatomic subsite as written (e.g. 'sigmoid colon', 'upper lobe', 'right cheek', 'antrum')."
    )
    laterality: Literal["left", "right", "bilateral", "not_applicable"] = Field(
        description="Side named on the SPECIMEN line; not_applicable when none is named."
    )
    procedure: Optional[str] = Field(description="Procedure as written (e.g. 'lumpectomy', 'shave biopsy').")


class TumorCore(BaseModel):
    """Fields shared by most organs."""

    histologic_type: Optional[str] = Field(description="Histologic type or skin diagnosis, verbatim.")
    histologic_category: Optional[
        Literal[
            # breast
            "invasive_ductal_nst",
            "invasive_lobular",
            # colon
            "adenocarcinoma",
            "mucinous_adenocarcinoma",
            # lung
            "squamous_cell_carcinoma",
            "lung_adenocarcinoma",
            "lung_other",
            # stomach
            "poorly_cohesive_diffuse",
            "mixed_glandular_poorly_cohesive",
            "intestinal_tubular_or_nos_adenocarcinoma",
            # kidney
            "clear_cell_rcc",
            "other_rcc",
            # prostate
            "acinar_adenocarcinoma",
            # anything else
            "other",
        ]
    ] = Field(description="Normalised histologic category; null for skin (use skin.group instead).")
    differentiation_grade: Optional[int] = Field(
        description="G1/G2/G3 differentiation grade as an integer (colon, stomach, skin SCC). Null if not stated."
    )
    tumor_size_value: Optional[float] = Field(
        description="Greatest tumor dimension from the 'Tumor size' field, number only, as written."
    )
    tumor_size_unit: Optional[Literal["mm", "cm"]] = Field(description="Unit of tumor_size_value as written.")
    lymphovascular_invasion: Status
    perineural_invasion: Status
    margin_status: Literal["negative", "positive", "not_reported"] = Field(
        description="positive when ANY margin is involved/positive by tumor (carcinoma, melanoma, nevus, etc.)."
    )
    closest_margin_mm: Optional[float] = Field(
        description=(
            "Breast: mm from invasive carcinoma to the closest margin; kidney: parenchymal margin "
            "closest distance. Only when margins are negative. Null for every other organ."
        )
    )
    involved_margins: list[str] = Field(
        description=(
            "Lower-case names of every involved/positive margin, e.g. ['deep'], ['bronchial'], "
            "['right posterolateral'], ['peripheral','deep']. Empty list when none is involved."
        )
    )
    lymph_nodes_positive: Optional[int] = Field(description="Number of positive lymph nodes. Null if none examined.")
    lymph_nodes_examined: Optional[int] = Field(description="Number of lymph nodes examined. Null if none examined.")
    pT: Optional[str] = Field(description="pT category exactly as written, e.g. 'pT1c(m)', 'pT4a', 'pTis'.")
    pN: Optional[str] = Field(description="pN category exactly as written, e.g. 'pN0(sn)', 'pN1a', 'pNX'.")


class Breast(BaseModel):
    nottingham_grade: Optional[int] = Field(description="Nottingham grade 1, 2 or 3.")
    er_status: Receptor
    pr_status: Receptor
    her2_status: Receptor = Field(description="HER2 overall result; IHC 2+ with ISH not amplified is negative.")
    tumor_focality: Optional[Literal["unifocal", "multifocal"]]
    dcis_present: Optional[bool]


class Lung(BaseModel):
    visceral_pleural_invasion: Status


class Colon(BaseModel):
    tumor_deposits: Status
    macroscopic_perforation: Status
    depth_of_invasion: Optional[str] = Field(description="Depth of invasion text, verbatim.")


class Stomach(BaseModel):
    her2_status: Receptor
    depth_of_invasion: Optional[str] = Field(description="Depth of invasion text, verbatim.")


class Prostate(BaseModel):
    gleason_primary: Optional[int]
    gleason_secondary: Optional[int]
    grade_group: Optional[int] = Field(description="Grade Group 1-5 as written, or derived from Gleason if unwritten.")
    extraprostatic_extension: Optional[Literal["not_identified", "focal", "established"]]
    seminal_vesicle_invasion: Status
    cribriform_glands: Status
    intraductal_carcinoma: Status


class Kidney(BaseModel):
    isup_grade: Optional[int]
    sarcomatoid_or_rhabdoid: Status
    extent_of_invasion: Optional[
        Literal[
            "confined_to_kidney",
            "perinephric_fat",
            "renal_sinus_fat",
            "renal_vein_segmental_branches",
            "main_renal_vein",
            "ivc_below_diaphragm",
            "ivc_above_diaphragm",
            "beyond_gerota_fascia",
            "ipsilateral_adrenal",
        ]
    ] = Field(description="Furthest extent of invasion from the 'Extent of invasion' field.")


class Skin(BaseModel):
    group: Optional[
        Literal[
            "bcc",
            "scc",
            "scc_in_situ",
            "melanoma",
            "melanoma_in_situ",
            "nevus",
            "seb_keratosis",
            "actinic_keratosis",
            "cyst",
            "wart",
            "dermatofibroma",
            "other_benign",
        ]
    ] = Field(description="Skin diagnosis group. 'scc' means INVASIVE squamous cell carcinoma only.")
    breslow_thickness_mm: Optional[float]
    ulceration: Status


class ReportExtraction(BaseModel):
    """One report, every section. Organ blocks that do not apply stay null."""

    header: Header
    specimen: Specimen
    tumor: TumorCore
    breast: Optional[Breast]
    lung: Optional[Lung]
    colon: Optional[Colon]
    stomach: Optional[Stomach]
    prostate: Optional[Prostate]
    kidney: Optional[Kidney]
    skin: Optional[Skin]
