 
"""
app/core/feature_engineering.py
All feature derivation / enrichment logic for a LeadRequest.
Design principles:
-----------------
• Every function in this module is a PURE function:
    - Takes only plain Python arguments (no model state).
    - Returns a value (never mutates arguments).
    - Is independently testable without Pydantic or DB connections.
• FeatureEngineer is the single public entry point for the service layer:
    from app.core.feature_engineering import FeatureEngineer
    from app.services.synapse_lookup import DISTRICT_LOOKUP_DICT, ...
    lead = FeatureEngineer(
        lead,
        district_lookup=DISTRICT_LOOKUP_DICT,
        lp_source_lookup=LP_SOURCE_SUBSOURCE_LOOKUP_DICT,
        question_mapping=QUESTION_MAPPING_DICT,
    ).enrich()
• Lookup dicts are injected — no module-level globals, easy to mock in tests.
• All regex is compiled at module level (paid once at import, not per request).
• No circular imports: this module imports from app.schemas.request only for
    the type hint (TYPE_CHECKING guard), keeping import cost zero at runtime.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import ziptimezone as zpt
from app.core.logger import setup_logger
if TYPE_CHECKING:
    from app.schemas.request import LeadRequest
    
logger = setup_logger(__name__)
# --- Pre-compiled regex (module-level — one-time cost) ---
_RE_VER_SCORE = re.compile(r"-ver(\d+)")
_RE_EXP_SCORE = re.compile(r"-exp:(\d+)")
_RE_DEBUG_ZIP  = re.compile(r"zp:(\d{5})")
_RE_DEBUG_CITY = re.compile(r"ct:([^:]+?)\s+zp:")
_RE_DEBUG_ADDR = re.compile(r"ad:([^:]+?)\s+st:")
_RE_DEBUG_ST   = re.compile(r"st:([A-Z]{2})")
_RE_DEBUG_PH   = re.compile(r"ph:(\d{10})")
_RE_DEBUG_EM   = re.compile(r"em:([^:\s]+)")
_RE_WWW        = re.compile(r"^www\.")
# --- Timezone constants ---
_US_CENTRAL  = "US/Central"
_US_EASTERN  = "US/Eastern"
_US_ALASKA   = "US/Alaska"
_US_PACIFIC  = "US/Pacific"
_US_MOUNTAIN = "US/Mountain"
# Maps lowercase 2-letter state code → canonical US/* timezone string
STATE_TIMEZONE_MAP: Dict[str, str] = {
    "al": _US_CENTRAL,  "ak": _US_ALASKA,   "az": "US/Arizona",
    "ar": _US_CENTRAL,  "ca": _US_PACIFIC,  "co": _US_MOUNTAIN,
    "ct": _US_EASTERN,  "de": _US_EASTERN,  "dc": _US_EASTERN,
    "fl": _US_EASTERN,  "ga": _US_EASTERN,  "hi": "US/Hawaii",
    "id": _US_MOUNTAIN, "il": _US_CENTRAL,  "in": _US_EASTERN,
    "ia": _US_CENTRAL,  "ks": _US_CENTRAL,  "ky": _US_EASTERN,
    "la": _US_CENTRAL,  "me": _US_EASTERN,  "md": _US_EASTERN,
    "ma": _US_EASTERN,  "mi": _US_EASTERN,  "mn": _US_CENTRAL,
    "ms": _US_CENTRAL,  "mo": _US_CENTRAL,  "mt": _US_MOUNTAIN,
    "ne": _US_CENTRAL,  "nv": _US_PACIFIC,  "nh": _US_EASTERN,
    "nj": _US_EASTERN,  "nm": _US_MOUNTAIN, "ny": _US_EASTERN,
    "nc": _US_EASTERN,  "nd": _US_CENTRAL,  "oh": _US_EASTERN,
    "ok": _US_CENTRAL,  "or": _US_PACIFIC,  "pa": _US_EASTERN,
    "ri": _US_EASTERN,  "sc": _US_EASTERN,  "sd": _US_CENTRAL,
    "tn": _US_CENTRAL,  "tx": _US_CENTRAL,  "ut": _US_MOUNTAIN,
    "vt": _US_EASTERN,  "va": _US_EASTERN,  "wa": _US_PACIFIC,
    "wv": _US_EASTERN,  "wi": _US_CENTRAL,  "wy": _US_MOUNTAIN,
}
_ZONE_CACHE: Dict[str, ZoneInfo] = {
    tz: ZoneInfo(tz) for tz in set(STATE_TIMEZONE_MAP.values())
}
# Sentinel comment text that counts as "no comment"
_NO_COMMENT_TEXT = (
    "Customer did not provide additional comments. "
    "Please contact the customer to discuss the details of this project."
)
# Credit rating thresholds — checked in descending score order
_CREDIT_THRESHOLDS: Tuple[Tuple[int, str], ...] = (
    (750, "A"),
    (700, "B"),
    (650, "C"),
    (500, "D"),
    (0,   "E"),
)

# --- Internal helper ---
def _lower(value: Optional[str]) -> Optional[str]:
    return value.strip().lower() if value else None
# --- Pure feature functions ---
def map_questions(
    search_terms: Optional[List[Dict[str, str]]],
    question_mapping: Dict[str, str],
) -> Dict[str, str]:
    """
    Map {question, answer} pairs to question_N field names.
    Args:
        search_terms:     Client-supplied [{question: ..., answer: ...}] list.
        question_mapping: Synapse mapping  {question_text: "question_N"}.
    Returns:
        Dict of {question_field_name: answer}, e.g. {"question_1": "Replace shower"}.
        Unknown question texts are logged and skipped.
        Duplicate question fields keep the first occurrence.
    """
    result: Dict[str, str] = {}
    if not search_terms:
        return result
    for item in search_terms:
        question_text = (item.get("question") or "").strip()
        answer        = (item.get("answer")   or "").strip()
        if not question_text:
            continue
        field_name = question_mapping.get(question_text)
        if not field_name:
            logger.debug("Question not in Synapse mapping: '%s'", question_text)
            continue
        if field_name not in result:          # first occurrence wins
            result[field_name] = answer
    return result

# === USER AGENT PARSING ==================================
def parse_user_agent(user_agent: Optional[str]) -> Dict[str, Any]:
    """
    Parse user agent exactly according to the feature engineering logic
    used during model training.
    """
    ua_lower = (user_agent or "").lower()
    # -----------------------------
    # Device type
    # -----------------------------
    device_type = "desktop"
    if re.search(r"mobile|android|iphone", ua_lower):
        device_type = "mobile"
    if re.search(r"tablet|ipad", ua_lower):
        device_type = "tablet"
    # -----------------------------
    # OS detection
    # -----------------------------
    os = "other"
    if "android" in ua_lower:
        os = "android"
    if re.search(r"iphone|ios", ua_lower):
        os = "ios"
    if "windows" in ua_lower:
        os = "windows"
    if "mac os" in ua_lower:
        os = "mac"
    # -----------------------------
    # Browser detection
    # Order must match DS code
    # -----------------------------
    browser = "other"
    if "chrome" in ua_lower:
        browser = "chrome"
    if "safari" in ua_lower:
        browser = "safari"
    if "firefox" in ua_lower:
        browser = "firefox"
    if "edge" in ua_lower:
        browser = "edge"
    # -----------------------------
    # App source
    # -----------------------------
    app_source = "web"
    if re.search(r"fbav|fb_iab", ua_lower):
        app_source = "facebook"
    if "instagram" in ua_lower:
        app_source = "instagram"
    # -----------------------------
    # In-app browser flag
    # -----------------------------
    is_in_app = app_source in ["facebook", "instagram"]
    return {
        "trustedform_device_type": device_type,
        "trustedform_os": os,
        "trustedform_browser": browser,
        "trustedform_app_source": app_source,
        "trustedform_is_in_app_browser": is_in_app,
    }

# === DISTANCE CALCULATION ================================
def compute_distance_km(
    tf_lat:  Optional[float],
    tf_lon:  Optional[float],
    zip_lat: Optional[float],
    zip_lon: Optional[float],
) -> Optional[float]:
    """
    Haversine distance between TrustedForm coords and ZIP-Codes.com coords.
    Returns km rounded to 4 dp, or None if any coordinate is missing.
    """
    if None in (tf_lat, tf_lon, zip_lat, zip_lon):
        return None
    try:
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [tf_lat, tf_lon, zip_lat, zip_lon])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        a = max(0.0, min(1.0, a))           # numerical stability clip
        return round(R * 2 * asin(sqrt(a)), 4)
    except Exception as exc:
        logger.warning("compute_distance_km failed: %s", exc)
        return None
# === TIMEZONE NORMALIZATION & MATCH ======================
def normalize_us_timezone(tz_raw: Optional[str]) -> Optional[str]:
    if tz_raw is None:
        return None
    tz_str = str(tz_raw)
    result = ""
    if re.search(r'New_York|Detroit|Indiana|Kentucky|Toronto', tz_str, re.IGNORECASE):
        result = 'America/New_York'
    if re.search(r'Chicago|Winnipeg', tz_str, re.IGNORECASE):
        result = 'America/Chicago'
    if re.search(r'Denver|Boise', tz_str, re.IGNORECASE):
        result = 'America/Denver'
    if re.search(r'Los_Angeles|Vancouver', tz_str, re.IGNORECASE):
        result = 'America/Los_Angeles'
    if re.search(r'Phoenix', tz_str, re.IGNORECASE):
        result = 'America/Phoenix'
    return result if result != "" else None
@lru_cache(maxsize=5000)
def get_official_timezone(zip_code: Optional[str]) -> Optional[str]:
    if not zip_code:
        return None
    try:
        zip_str = str(zip_code).split(".")[0].zfill(5)
        if zip_str.startswith(("85", "86")):
            return "America/Phoenix"
        # official_name = zpt.get_timezone_by_zip(zip_str)
        # tz_map = {
        #     "Eastern": "America/New_York",
        #     "Central": "America/Chicago",
        #     "Mountain": "America/Denver",
        #     "Pacific": "America/Los_Angeles",
        #     "Phoenix": "America/Phoenix",
        # }
        # return tz_map.get(official_name)
        return None
    except Exception:
        return None
def compute_timezone_match(
    tz_raw: Optional[str],
    postal_code: Optional[str],
) -> bool:
    normalized = normalize_us_timezone(tz_raw)
    official = get_official_timezone(postal_code)
    # Exactly mirrors: normalized_tz.fillna("").astype(str).str.strip() == df["official_timezone"].fillna("").astype(str)
    norm_str = "" if normalized is None else str(normalized).strip()
    off_str = "" if official is None else str(official)
    return norm_str == off_str
def compute_trestle_matches(
    addr_city:          Optional[str],
    email:              Optional[str],
    first_name:         Optional[str],
    trestle_city:       Optional[str],
    trestle_email:      Optional[str],
    trestle_first_name: Optional[str],
) -> Dict[str, Optional[bool]]:
    """
    Case-insensitive comparison of lead identity fields against Trestle's returned values.
    Returns:
        {"azure_city_match": bool|None, "azure_email_match": bool|None,
        "azure_first_name_match": bool|None}
    """
    def _eq(a: Optional[str], b: Optional[str]) -> Optional[bool]:
        if a and b:
            return a == b
            # return a.strip().lower() == b.strip().lower()
        # return None
        return False
    
    return {
        "azure_city_match":       _eq(trestle_city,       addr_city),
        "azure_email_match":      _eq(trestle_email,      email),
        "azure_first_name_match": _eq(trestle_first_name, first_name),
    }
# === AZURE CREDIT FILTER DEBUG EXTRACTION ================
def extract_debug_attributes(
    debug_str: Optional[str],
    addr_zip:  Optional[str],
) -> Dict[str, Any]:
    """
    Parse the AzureCreditFilterV20Debug string to extract:
        - experian_creditscore  (int | None)
        - versium_creditscore   (int | None)
        - azure_zip_match       (bool | None)
        - address1 fallback     (str | None)
        - addr_state fallback   (str | None)
        - addr_city fallback    (str | None)
        - phone fallback        (str | None)
        - email fallback        (str | None)
    Fallback fields are only returned when they have a value in the debug string
    so the caller can use them to fill gaps in the request without overwriting
    data that was already supplied.
    """
    out: Dict[str, Any] = {
        "experian_creditscore": None,
        "versium_creditscore":  None,
        "azure_zip_match":      None,
        "address1_fallback":    None,
        "addr_state_fallback":  None,
        "addr_city_fallback":   None,
        "phone_fallback":       None,
        "email_fallback":       None,
    }
    if not debug_str or not debug_str.strip():
        logger.info("extract_debug_attributes: no debug string supplied, skipping.")
        return out
    s = debug_str.strip()
    m = _RE_EXP_SCORE.search(s)
    out["experian_creditscore"] = int(m.group(1)) if m else None
    if not m:
        logger.debug("Experian credit score not found in debug string.")
    m = _RE_VER_SCORE.search(s)
    out["versium_creditscore"] = int(m.group(1)) if m else None
    if not m:
        logger.debug("Versium credit score not found in debug string.")
    m = _RE_DEBUG_ZIP.search(s)
    if m and addr_zip:
        out["azure_zip_match"] = m.group(1) == addr_zip
    m = _RE_DEBUG_ADDR.search(s)
    if m:
        out["address1_fallback"] = m.group(1).strip()
    m = _RE_DEBUG_ST.search(s)
    if m:
        out["addr_state_fallback"] = m.group(1).strip()
    m = _RE_DEBUG_CITY.search(s)
    if m:
        out["addr_city_fallback"] = m.group(1).strip()
    m = _RE_DEBUG_PH.search(s)
    if m:
        out["phone_fallback"] = m.group(1).strip()
    m = _RE_DEBUG_EM.search(s)
    if m:
        out["email_fallback"] = m.group(1).strip()
    return out
# === MONTHS SINCE LINKED =================================
def compute_months_since_linked(
    link_date_str: Optional[str],
    submission_timestamp_str: Optional[str],
) -> Optional[int]:
    """
    Replicates training logic without pandas.
    months_since_linked =
        (submission_year - link_year) * 12 +
        (submission_month - link_month)
    submission_timestamp format: "MM-DD-YYYY HH:MM"
    """
    if not link_date_str or not submission_timestamp_str:
        return None
    
    # Parse submission timestamp
    try:
        submission_ts = datetime.strptime(
            submission_timestamp_str.strip(),
            "%m-%d-%Y %H:%M",
        )
    except Exception:
        logger.warning(
            "compute_months_since_linked: invalid submission_timestamp '%s'",
            submission_timestamp_str,
        )
        return None
    # Parse link date
    formats = (
        "%a %b %d %Y %H:%M:%S GMT+0000",
        "%a %b %d %Y %H:%M:%S GMT%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    )
    link_date = None
    for fmt in formats:
        try:
            link_date = datetime.strptime(link_date_str.strip(), fmt)
            break
        except ValueError:
            continue
    if link_date is None:
        logger.warning(
            "compute_months_since_linked: cannot parse link_date '%s'",
            link_date_str,
        )
        return None
    # Compute months difference
    months = (
        (submission_ts.year - link_date.year) * 12
        + (submission_ts.month - link_date.month)
    )
    return max(months, 0)
    
# === ROOT DOMAIN EXTRACTION ==============================
def extract_root_domain(page_url: Optional[str]) -> Optional[str]:
    """
    Extract the eTLD+1 root domain from a TrustedForm page URL.
    "https://sub.bathremodelspecialists.com/path" → "bathremodelspecialists.com"
    """
    if not page_url or not page_url.strip():
        return None
    try:
        parsed = urlparse(page_url.strip())
        host   = parsed.netloc or parsed.path
        host   = _RE_WWW.sub("", host)         # strip leading www.
        parts  = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else (host or None)
    except Exception as exc:
        logger.warning("extract_root_domain failed for '%s': %s", page_url, exc)
        return None

# === SUBMISSION HOUR EXTRACTION ==========================
def compute_submission_hour(
    submission_timestamp: Optional[str],
    state: Optional[str],
) -> Optional[int]:
    """
    Convert the UTC submission timestamp to the lead's local hour (0–23).
    Args:
        submission_timestamp: "MM-DD-YYYY HH:MM" string or a datetime object.
        state:                addr_state (2-letter code).
    Returns:
        Local hour as int, or None if either input is missing / unparseable.
    """
    if not submission_timestamp or not state:
        return None
    try:
        if isinstance(submission_timestamp, str):
            dt = datetime.strptime(submission_timestamp.strip(), "%m-%d-%Y %H:%M")
            dt = dt.replace(tzinfo=timezone.utc)
        elif isinstance(submission_timestamp, datetime):
            dt = submission_timestamp
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            return None
        tz_str    = STATE_TIMEZONE_MAP.get(_lower(state), "UTC")
        local_dt = dt.astimezone(_ZONE_CACHE[tz_str])
        return local_dt.hour
    except Exception as exc:
        logger.warning("compute_submission_hour failed: %s", exc)
        return None
    
# --- DISTRICT LOOKUP ---
def assign_district(
    division:        Optional[str],
    addr_zip:        Optional[str],
    district_lookup: Dict[Any, str],
) -> Optional[str]:
    """
    Look up district name from the pre-loaded Synapse DISTRICT_LOOKUP_DICT.
    Key format: (lowercase_division, zip_string)
    """
    div = _lower(division)
    z   = (addr_zip or "").strip()
    if not div or not z:
        return None
    district = district_lookup.get((div, z))
    if not district:
        logger.warning("assign_district: no match for division='%s' zip='%s'", div, z)
    return district

def map_lp_source(
    srs_id:           Optional[int],
    division:         Optional[str],
    lp_source_lookup: Dict[Any, Tuple[str, str]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (lp_source, lp_sub_source) from the Synapse lookup table.
    Key format: (srs_id, lowercase_division)
    Returns (None, None) when the key is absent.
    """
    div = _lower(division)
    if not srs_id or not div:
        return None, None
    mapping = lp_source_lookup.get((srs_id, div))
    if mapping:
        return mapping
    logger.warning("map_lp_source: no mapping for srs_id=%s division='%s'", srs_id, div)
    return None, None
def compute_experian_credit_rating(score: Optional[int]) -> str:
    """
    Map an Experian credit score to a letter rating.
    Bands:  ≥750 → A | ≥700 → B | ≥650 → C | ≥500 → D | ≥0 → E | None → U
    """
    if score is None:
        return "U"
    for min_score, rating in _CREDIT_THRESHOLDS:
        if score >= min_score:
            return rating
    return "U"
def compute_comment_features(comments: Optional[str]) -> Dict[str, Any]:
    """
    Return comment length and a presence label.
    Returns:
        {"commentlength": int, "commentpresent": "Comments" | "No Comments"}
    """
    text = (comments or "").strip()
    present = "Comments" if text and text != _NO_COMMENT_TEXT else "No Comments"
    return {"commentlength": len(text), "commentpresent": present}

# ================================================
# FeatureEngineer  — the single public entry point
# ================================================
class FeatureEngineer:
    """
    Enriches a LeadRequest in-place by calling each pure feature function
    in dependency order.
    Usage (from score.py or any service layer):
        from app.services.synapse_lookup import (
            DISTRICT_LOOKUP_DICT,
            LP_SOURCE_SUBSOURCE_LOOKUP_DICT,
            QUESTION_MAPPING_DICT,
        )
        from app.core.feature_engineering import FeatureEngineer
        lead = FeatureEngineer(
            lead,
            district_lookup  = DISTRICT_LOOKUP_DICT,
            lp_source_lookup = LP_SOURCE_SUBSOURCE_LOOKUP_DICT,
            question_mapping = QUESTION_MAPPING_DICT,
        ).enrich()
    """
    def __init__(
        self,
        lead:             "LeadRequest",
        district_lookup:  Dict[Any, str],
        lp_source_lookup: Dict[Any, Tuple[str, str]],
        question_mapping: Dict[str, str],
    ) -> None:
        self._lead            = lead
        self._district_lookup  = district_lookup
        self._lp_source_lookup = lp_source_lookup
        self._question_mapping = question_mapping
    def enrich(self) -> "LeadRequest":
        """
        Run all enrichment steps and return the mutated lead.
        Steps are ordered so that fields produced in an early step can be
        consumed by a later step (e.g. experian_creditscore → experian_creditrating).
        """
        self._pipeline = [
                self._enrich_debug_attributes,   # fills credit scores + address gaps
                self._enrich_credit_rating,      # needs experian_creditscore
                self._enrich_trestle_matches,    # city / email / name comparison
                self._enrich_user_agent,         # device / os / browser
                self._enrich_distance,           # haversine
                self._enrich_timezone_match,     # TZ vs state
                self._enrich_submission_hour,    # local hour
                self._enrich_months_since_linked,
                self._enrich_root_domain,
                self._enrich_district,
                self._enrich_lp_source,
                self._enrich_comment_features,
                self._enrich_questions
        ]
        for step in self._pipeline:
            step()
        return self._lead
    # --- private enrichment steps ---
    def _enrich_debug_attributes(self) -> None:
        lead = self._lead
        attrs = extract_debug_attributes(
            lead.azure_credit_filter_v_20_debug,
            lead.addr_zip,
        )
        lead.experian_creditscore = attrs["experian_creditscore"]
        lead.versium_creditscore  = attrs["versium_creditscore"]
        lead.azure_zip_match      = attrs["azure_zip_match"]
        # Fill address gaps only when the client did not supply them
        if not lead.address1:
            lead.address1 = attrs["address1_fallback"] or "-"
        if not lead.addr_state:
            lead.addr_state = attrs["addr_state_fallback"]
        if not lead.addr_city:
            lead.addr_city = attrs["addr_city_fallback"]
        if not lead.phone:
            lead.phone = attrs["phone_fallback"]
        if not lead.email:
            lead.email = attrs["email_fallback"]
    def _enrich_credit_rating(self) -> None:
        self._lead.experian_creditrating = compute_experian_credit_rating(
            self._lead.experian_creditscore
        )
    def _enrich_trestle_matches(self) -> None:
        lead    = self._lead
        matches = compute_trestle_matches(
            addr_city          = lead.addr_city,
            email              = str(lead.email) if lead.email else None,
            first_name         = lead.first_name,
            trestle_city       = lead.azure_credit_filter_v_20_trestle_city,
            trestle_email      = lead.azure_credit_filter_v_20_trestle_email,
            trestle_first_name = lead.azure_credit_filter_v_20_trestle_first_name,
        )
        lead.azure_city_match       = matches["azure_city_match"]
        lead.azure_email_match      = matches["azure_email_match"]
        lead.azure_first_name_match = matches["azure_first_name_match"]
    def _enrich_user_agent(self) -> None:
        lead = self._lead
        ua   = parse_user_agent(lead.trustedform_user_agent)
        lead.trustedform_device_type       = ua["trustedform_device_type"]
        lead.trustedform_os                = ua["trustedform_os"]
        lead.trustedform_browser           = ua["trustedform_browser"]
        lead.trustedform_app_source        = ua["trustedform_app_source"]
        lead.trustedform_is_in_app_browser = ua["trustedform_is_in_app_browser"]
    def _enrich_distance(self) -> None:
        lead = self._lead
        lead.distance_km = compute_distance_km(
            lead.trustedform_latitude,
            lead.trustedform_longitude,
            lead.zip_codes_com_lookup_latitude,
            lead.zip_codes_com_lookup_longitude,
        )
    def _enrich_timezone_match(self) -> None:
        lead = self._lead
        lead.trustedform_timezone_match = compute_timezone_match(
            lead.trustedform_timezone,
            lead.addr_zip,
        )

    def _enrich_submission_hour(self) -> None:
        lead = self._lead
        lead.submission_hour = compute_submission_hour(
            lead.submission_timestamp,
            lead.addr_state,
        )
    def _enrich_months_since_linked(self) -> None:
        self._lead.months_since_linked = compute_months_since_linked(
            self._lead.azure_credit_filter_v_20_trestle_link_to_phone_start_date,
            self._lead.submission_timestamp,
        )

    def _enrich_root_domain(self) -> None:
        self._lead.root_domain = extract_root_domain(self._lead.trustedform_page_url)
    def _enrich_district(self) -> None:
        lead = self._lead
        lead.district = assign_district(
            lead.division,
            lead.addr_zip,
            self._district_lookup,
        )
    def _enrich_lp_source(self) -> None:
        lead = self._lead
        lead.lp_source, lead.lp_sub_source = map_lp_source(
            lead.srs_id,
            lead.division,
            self._lp_source_lookup,
        )
    def _enrich_comment_features(self) -> None:
        lead    = self._lead
        metrics = compute_comment_features(lead.comments)
        lead.commentlength = metrics["commentlength"]
        lead.commentpresent = metrics["commentpresent"]
    def _enrich_questions(self) -> None:
        """Map search_terms answers into question_N fields on the lead."""
        lead    = self._lead
        mapping = map_questions(lead.search_terms, self._question_mapping)
        for field_name, answer in mapping.items():
            if hasattr(lead, field_name):
                setattr(lead, field_name, answer)
            else:
                logger.warning(
                    "_enrich_questions: unknown field '%s' from question_mapping", field_name
                )
 
 