from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import tldextract
import numpy as np
import pandas as pd
import math
from collections import Counter
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import os

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# Load trained model (safe load with error handling)
# ──────────────────────────────────────────────
# Project root = parent of the backend/ folder (model/ and dataset/ live there)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"[WARNING] Model file not found at: {MODEL_PATH}. ML prediction will be skipped.")
except Exception as e:
    model = None
    print(f"[ERROR] Failed to load model: {e}. ML prediction will be skipped.")


# ──────────────────────────────────────────────
# Brand list: hardcoded fallback + optional data.txt merge
# ──────────────────────────────────────────────
_BUILTIN_BRANDS = [
    "paypal", "google", "facebook", "amazon", "apple", "microsoft",
    "netflix", "instagram", "twitter", "linkedin", "dropbox", "github",
    "yahoo", "ebay", "wellsfargo", "chase", "bankofamerica", "citibank"
]

_data_txt_path = os.path.join(BASE_DIR, "dataset", "data.txt")
if os.path.exists(_data_txt_path):
    try:
        with open(_data_txt_path, "r", encoding="utf-8") as f:
            _file_brands = [line.strip().lower() for line in f if line.strip()]
        BRAND_LIST = list(set(_BUILTIN_BRANDS + _file_brands))
        print(f"[INFO] Loaded {len(_file_brands)} brands from data.txt")
    except Exception as e:
        print(f"[WARNING] Could not read data.txt: {e}. Using built-in brands only.")
        BRAND_LIST = _BUILTIN_BRANDS
else:
    BRAND_LIST = _BUILTIN_BRANDS

# Whitelist of known-good exact domains — these will NEVER be flagged as phishing
WHITELIST = {
    "google.com", "www.google.com",
    "facebook.com", "www.facebook.com",
    "amazon.com", "www.amazon.com",
    "apple.com", "www.apple.com",
    "microsoft.com", "www.microsoft.com",
    "netflix.com", "www.netflix.com",
    "instagram.com", "www.instagram.com",
    "twitter.com", "www.twitter.com",
    "linkedin.com", "www.linkedin.com",
    "github.com", "www.github.com",
    "paypal.com", "www.paypal.com",
    "yahoo.com", "www.yahoo.com",
    "ebay.com", "www.ebay.com",
    "dropbox.com", "www.dropbox.com",
}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def is_whitelisted(url: str, full_domain: str) -> bool:
    """
    Returns True if the URL's full domain exactly matches a known-safe domain.
    Strips protocol and path before checking.
    """
    # Normalize: strip protocol, path, query, fragment
    stripped = re.sub(r"^https?://", "", url.lower().strip())
    stripped = stripped.split("/")[0].split("?")[0].split("#")[0]
    return stripped in WHITELIST or full_domain.lower() in WHITELIST


def detect_typosquat_similarity(domain: str) -> list:
    """
    Levenshtein-based fuzzy matching using rapidfuzz.

    Uses TWO checks:
      - Levenshtein distance: 1-2 character edits (catches 'paypa1', 'gooogle')
      - fuzz.partial_ratio >= 90: brand inside a longer domain (catches 'paypal-login')

    Exact matches are excluded.
    Returns: sorted list of (brand, score) tuples, highest score first.
    """
    domain = domain.lower().strip()
    if not domain:
        return []

    matches = []

    for brand in BRAND_LIST:
        brand_lower = brand.lower().strip()

        if domain == brand_lower:
            continue  # exact match is NOT a typosquat

        # 1. Character swaps / typos (paypa1, gooogle): small edit distance.
        #    Short brands allow 1 edit, longer brands allow 2.
        max_edits = 1 if len(brand_lower) <= 5 else 2
        distance  = Levenshtein.distance(domain, brand_lower)
        if distance <= max_edits and len(domain) >= 4:
            matches.append((brand_lower, round(fuzz.ratio(domain, brand_lower), 1)))
            continue

        # 2. Brand embedded in a longer domain (paypal-login, secure-google).
        #    Only checked when the domain is longer than the brand, so short
        #    domains like "bit" are not matched against "binance".
        if len(domain) > len(brand_lower):
            partial = fuzz.partial_ratio(domain, brand_lower)
            if partial >= 90:
                matches.append((brand_lower, round(partial, 1)))

    # Deduplicate and sort by score descending
    seen = set()
    unique_matches = []
    for brand, score in sorted(matches, key=lambda x: -x[1]):
        if brand not in seen:
            seen.add(brand)
            unique_matches.append((brand, score))

    return unique_matches


def check_typosquatting(domain: str) -> bool:
    """
    Substring check — catches 'paypal-secure.com' style domains.
    Only flags if the brand name appears as part of a LONGER string (not an exact match).
    """
    common_targets = ["paypal", "google", "facebook", "amazon", "apple"]
    domain = domain.lower().strip()
    for target in common_targets:
        # Must contain the brand AND not BE the brand exactly
        if target in domain and domain != target:
            return True
    return False


def check_homograph(domain: str) -> bool:
    """Detect Unicode homograph attacks (non-ASCII characters in domain)."""
    try:
        domain.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def count_subdomains(url: str) -> int:
    """Count the number of subdomains using tldextract."""
    try:
        extracted = tldextract.extract(url)
        if extracted.subdomain:
            return len(extracted.subdomain.split("."))
    except Exception:
        pass
    return 0


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    probabilities = [n_x / len(text) for _, n_x in Counter(text).items()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return round(entropy, 4)


def check_keywords(url: str) -> bool:
    """
    Check for phishing-related keywords using whole-word regex matching.
    Avoids false positives like matching 'secure' inside 'https'.
    """
    keywords = ["login", "verify", "secure", "account", "update", "bank", "confirm"]
    # Extract path + query only (after the domain) to reduce false positives on protocol
    path_part = re.sub(r"^https?://[^/]+", "", url.lower())
    full_check = url.lower()

    for word in keywords:
        # Check in path with word boundary OR as standalone param
        if re.search(r'[\W_]' + re.escape(word) + r'[\W_]|[\W_]' + re.escape(word) + r'$|^' + re.escape(word) + r'[\W_]', path_part):
            return True
        # Also check full URL but only with word boundaries
        if re.search(r'\b' + re.escape(word) + r'\b', full_check):
            return True
    return False


# URL shortening services (UCI feature "Shortining_Service")
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bitly.com", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc",
    "lnkd.in", "db.tt", "qr.ae", "bl.ink", "rebrand.ly", "s.id", "v.gd",
}

IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def extract_features(url: str) -> dict:
    """
    Extract URL-based features using the same encoding as the UCI Phishing
    Websites dataset: 1 = legitimate, 0 = suspicious, -1 = phishing.

    Only features that can be computed from the URL itself are used, so the
    model sees exactly the same kind of data it was trained on.
    """
    parsed_host = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    host_port   = parsed_host.split("/")[0].split("?")[0].split("#")[0]
    host_port   = host_port.split("@")[-1]          # ignore user-info before @
    host        = host_port.split(":")[0].lower()

    extracted = tldextract.extract(url)
    registered = ".".join(filter(None, [extracted.domain, extracted.suffix]))
    sub_parts = [p for p in extracted.subdomain.split(".") if p and p != "www"]

    # Non-standard port in the URL
    port_match = re.search(r":(\d+)$", host_port)
    port = int(port_match.group(1)) if port_match else None

    features = {
        # IP address used instead of a domain name
        "having_IP_Address": -1 if IP_PATTERN.search(host) else 1,

        # URL length: < 54 legit, 54–75 suspicious, > 75 phishing
        "URL_Length": 1 if len(url) < 54 else (0 if len(url) <= 75 else -1),

        # Known URL shortener
        "Shortining_Service": -1 if (host in SHORTENERS or registered in SHORTENERS) else 1,

        # "@" symbol (browser ignores everything before it)
        "having_At_Symbol": -1 if "@" in url else 1,

        # "//" appearing after the protocol (redirect trick)
        "double_slash_redirecting": -1 if url.rfind("//") > 7 else 1,

        # "-" in the domain name (e.g. paypal-secure.com)
        "Prefix_Suffix": -1 if "-" in host else 1,

        # Subdomain depth (www ignored): 0 legit, 1 suspicious, 2+ phishing
        "having_Sub_Domain": 1 if len(sub_parts) == 0 else (0 if len(sub_parts) == 1 else -1),

        # SSL: approximated from the scheme (the certificate itself is not checked)
        "SSLfinal_State": 1 if url.lower().startswith("https://") else -1,

        # Non-standard port
        "port": -1 if (port is not None and port not in (80, 443)) else 1,

        # "https" token inside the domain name (e.g. https-paypal.com)
        "HTTPS_token": -1 if "https" in host else 1,
    }
    return features


def get_ml_prediction(url: str) -> tuple:
    """
    Run ML model prediction safely.
    Returns (result_label, raw_value, phishing_probability)
    or ("Unknown", None, None) if the model is unavailable.
    """
    if model is None:
        return "Unknown", None, None

    try:
        features = extract_features(url)
        columns  = list(getattr(model, "feature_names_in_", features.keys()))
        X        = pd.DataFrame([[features[c] for c in columns]], columns=columns)

        prediction_raw = int(model.predict(X)[0])
        classes        = list(model.classes_)
        phishing_proba = float(model.predict_proba(X)[0][classes.index(-1)])

        result = "Phishing" if prediction_raw == -1 else "Legitimate"
        return result, prediction_raw, round(phishing_proba * 100, 1)
    except Exception as e:
        print(f"[ERROR] ML prediction failed: {e}")
        return "Unknown", None, None


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def home():
    return "Phishing Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field in request body"}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400

    # Basic URL format sanity check
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return jsonify({"error": "URL must start with http:// or https://"}), 400

    # Extract domain parts safely
    try:
        extracted   = tldextract.extract(url)
        domain      = extracted.domain
        suffix      = extracted.suffix
        subdomain   = extracted.subdomain
        full_domain = ".".join(filter(None, [subdomain, domain, suffix]))
    except Exception as e:
        return jsonify({"error": f"Failed to parse URL: {str(e)}"}), 400

    if not domain:
        return jsonify({"error": "Could not extract a valid domain from the URL"}), 400

    # ── Whitelist check — skip all analysis for known-safe domains ──
    if is_whitelisted(url, full_domain):
        return jsonify({
            "url":         url,
            "prediction":  "Legitimate",
            "risk_score":  0,
            "alerts":      [],
            "alert_count": 0,
            "whitelisted": True,
            "details": {
                "uses_https":              url.lower().startswith("https"),
                "has_ip_address":          False,
                "subdomain_count":         count_subdomains(url),
                "domain_entropy":          calculate_entropy(domain),
                "domain":                  domain,
                "full_domain":             full_domain,
                "url_length":              len(url),
                "has_suspicious_keywords": False,
                "is_typosquat":            False,
                "is_homograph":            False,
                "similar_brands":          [],
            }
        })

    alerts = []

    # 1. Substring typosquat check
    if check_typosquatting(domain):
        alerts.append("Possible typosquatting attack detected")

    # 2. Homograph / Unicode attack check
    if check_homograph(full_domain):
        alerts.append("Unicode homograph attack detected")

    # 3. Levenshtein fuzzy brand similarity check
    similar_brands = detect_typosquat_similarity(domain)
    for brand, score in similar_brands:
        alerts.append(f"Possible typosquatting of '{brand}' (similarity {score}%)")

    # 4. Too many subdomains
    subdomain_count = count_subdomains(url)
    if subdomain_count > 2:
        alerts.append(f"Too many subdomains ({subdomain_count} detected)")

    # 5. High entropy domain
    entropy = calculate_entropy(domain)
    if entropy > 3.5:
        alerts.append(f"High entropy domain detected (entropy: {entropy})")

    # 6. Phishing keywords in URL
    has_keywords = check_keywords(url)
    if has_keywords:
        alerts.append("Suspicious phishing keywords found in URL")

    # 7. IP address used instead of domain
    has_ip = bool(re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", url))
    if has_ip:
        alerts.append("URL contains an IP address instead of a domain name")

    # 8. No HTTPS
    uses_https = url.lower().startswith("https")
    if not uses_https:
        alerts.append("URL does not use HTTPS")

    # 9. URL shortener (hides the real destination)
    ml_features = extract_features(url)
    if ml_features["Shortining_Service"] == -1:
        alerts.append("URL shortener detected (real destination is hidden)")

    # 10. "@" in the address part (browsers ignore everything before it,
    #     e.g. https://www.paypal.com@evil.ru). "@" in a path like
    #     medium.com/@user is normal and is not flagged.
    authority = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).split("/")[0]
    has_at_in_host = "@" in authority
    if has_at_in_host:
        alerts.append("URL hides the real domain behind an '@' symbol")

    # 11. Hidden redirect with "//" after the protocol
    if ml_features["double_slash_redirecting"] == -1:
        alerts.append("Suspicious '//' redirect inside the URL")

    # 12. Non-standard port
    if ml_features["port"] == -1:
        alerts.append("URL uses a non-standard port")

    # ── ML prediction ──
    ml_result, prediction_raw, ml_phishing_probability = get_ml_prediction(url)

    # ── Smart final verdict ──
    # Heuristic alerts and the ML model are combined. Neither signal can
    # override the other on its own: a clean URL is not flagged just because
    # the model is unsure, and strong alerts are not ignored because the
    # model says "Legitimate".
    alert_count = len(alerts)
    ml_says_phishing = ml_phishing_probability is not None and ml_phishing_probability >= 50

    is_homograph = check_homograph(full_domain)

    if alert_count >= 3 or is_homograph or has_at_in_host:
        # Homographs and "@" tricks are almost never legitimate, so they are decisive
        final_result = "Phishing"
    elif alert_count == 2:
        if ml_result == "Unknown" or ml_says_phishing:
            final_result = "Phishing"
        else:
            final_result = "Suspicious"
    elif alert_count == 1:
        final_result = "Suspicious"
    else:
        # No alerts: only a very confident model raises a warning
        if ml_phishing_probability is not None and ml_phishing_probability >= 80:
            final_result = "Suspicious"
        else:
            final_result = "Legitimate"

    # ── Risk score (0–100) ──
    # Each alert adds 15 points and the ML probability adds up to 25 points.
    # The score is then kept inside the band that matches the verdict, so a
    # "Phishing" result never shows a low score (and vice versa).
    raw_score = alert_count * 15 + (ml_phishing_probability or 0) * 0.25
    bands = {"Legitimate": (0, 29), "Suspicious": (30, 69), "Phishing": (70, 100)}
    low, high  = bands[final_result]
    risk_score = int(round(min(high, max(low, raw_score))))

    return jsonify({
        "url":         url,
        "prediction":  final_result,
        "risk_score":  risk_score,
        "alerts":      alerts,
        "alert_count": len(alerts),
        "whitelisted": False,
        "details": {
            "uses_https":              uses_https,
            "has_ip_address":          has_ip,
            "subdomain_count":         subdomain_count,
            "domain_entropy":          entropy,
            "domain":                  domain,
            "full_domain":             full_domain,
            "url_length":              len(url),
            "has_suspicious_keywords": has_keywords,
            "is_typosquat":            check_typosquatting(domain),
            "is_homograph":            check_homograph(full_domain),
            "similar_brands": [
                {"brand": b, "similarity": s} for b, s in similar_brands
            ],
            "ml_raw_prediction": prediction_raw,
            "ml_phishing_probability": ml_phishing_probability,
        }
    })


if __name__ == "__main__":
    app.run(debug=True)