import streamlit as st
import pandas as pd
import dns.resolver
import io
import re
from typing import Any

st.set_page_config(page_title="PropStream CSV Builder", page_icon="📄", layout="wide")

st.title("PropStream CSV Builder")

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "aol.com",
    "outlook.com",
    "icloud.com",
    "comcast.net",
    "att.net",
    "me.com",
    "msn.com",
    "live.com",
    "mac.com",
    "verizon.net",
    "sbcglobal.net",
    "cox.net",
    "proton.me",
    "protonmail.com",
    "gmx.com",
}

# Disposable / throwaway email domains to reject
DISPOSABLE_DOMAINS = {
    "mailinator.com",
    "guerrillamail.com",
    "tempmail.com",
    "throwaway.email",
    "yopmail.com",
    "sharklasers.com",
    "guerrillamailblock.com",
    "grr.la",
    "dispostable.com",
    "trashmail.com",
    "trashmail.me",
    "fakeinbox.com",
    "temp-mail.org",
    "10minutemail.com",
    "getnada.com",
    "maildrop.cc",
    "harakirimail.com",
    "mailnesia.com",
    "binkmail.com",
    "safetymail.info",
}

OUTPUT_COLUMNS = [
    "email",
    "first_name",
    "last_name",
    "phone",
    "company_name",
    "property_address",
    "property_city",
    "property_state",
    "property_zip",
    "property_full_address",
    "property_value",
    "property_bedrooms",
    "property_bathrooms",
    "property_sqft",
    "mls_status",
    "mls_amount",
    "est_equity",
]


def _s(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _clean_number(value: object) -> str:
    """
    Keep as string, strip trailing .0, and remove obvious NaN-ish values.
    We intentionally don't add commas/currency symbols.
    """
    s = _s(value)
    if not s:
        return ""
    s_lower = s.lower()
    if s_lower in {"nan", "none", "null"}:
        return ""
    # "123.0" -> "123"
    if re.fullmatch(r"-?\d+\.0", s):
        return s.split(".", 1)[0]
    return s


def _zip(value: object) -> str:
    return _clean_number(value)


def _phone(value: object) -> str:
    # Keep digits only; streamlines reading + avoids "123-456-7890" variations.
    digits = re.sub(r"\D+", "", _s(value))
    return digits


def _normalize_street(address: str) -> str:
    """
    Normalize street address for matching between exports.
    We intentionally ignore city/state/zip per your requirement (match by street address).
    """
    a = _s(address).lower()
    if not a:
        return ""
    a = a.replace(",", " ")
    a = re.sub(r"\s+", " ", a).strip()

    # Strip common unit patterns to improve matching.
    # Examples: "123 Main St #4", "123 Main St Apt 4", "123 Main St Unit 4"
    a = re.sub(r"\s+(#|apt|apartment|unit|ste|suite)\s*\w+\s*$", "", a).strip()
    return a


# ---------------------------------------------------------------------------
# Email validation helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _is_valid_syntax(email: str) -> bool:
    """Basic RFC-ish syntax check."""
    return bool(_EMAIL_RE.match(email))


@st.cache_data(show_spinner=False, ttl=3600)
def _domain_has_mx(domain: str) -> bool:
    """
    Check if the domain has valid MX records (can receive email).
    Results are cached for the session so we only look up each domain once.
    """
    try:
        answers = dns.resolver.resolve(domain, "MX", lifetime=5)
        return len(answers) > 0
    except Exception:
        # NXDOMAIN, NoAnswer, Timeout, etc. → domain can't receive mail
        return False


def _is_disposable(domain: str) -> bool:
    return domain in DISPOSABLE_DOMAINS


def validate_email(email: str) -> tuple[bool, str]:
    """
    Free verification pipeline. Returns (is_valid, reason).
    Steps:
      1. Syntax check
      2. Disposable domain check
      3. MX record lookup (does the domain actually accept mail?)
    """
    email = email.strip().lower()
    if not email:
        return False, "empty"

    if not _is_valid_syntax(email):
        return False, "bad_syntax"

    domain = email.split("@")[-1]

    if _is_disposable(domain):
        return False, "disposable"

    if not _domain_has_mx(domain):
        return False, "no_mx"

    return True, "ok"


# ---------------------------------------------------------------------------
# Email extraction & ranking
# ---------------------------------------------------------------------------

def _extract_emails_ordered(contact_row: pd.Series) -> list[str]:
    """
    Extract emails from Email 1 → Email 4 columns in order (PropStream priority).
    Falls back to scanning any column with 'email' in the name.
    """
    emails: list[str] = []

    # Prefer explicit Email 1..4 columns in order
    explicit_cols = [f"Email {i}" for i in range(1, 10)]
    has_explicit = any(c in contact_row for c in explicit_cols)

    if has_explicit:
        for col in explicit_cols:
            if col not in contact_row:
                continue
            v = _s(contact_row[col]).lower()
            if v and "@" in v and not v.startswith("@") and not v.endswith("@"):
                emails.append(v)
    else:
        # Fallback: scan all email-like columns
        for col, value in contact_row.items():
            if "email" not in str(col).lower():
                continue
            v = _s(value).lower()
            if not v:
                continue
            parts = re.split(r"[,\s;]+", v)
            for p in parts:
                p = p.strip().lower()
                if p and "@" in p and not p.startswith("@") and not p.endswith("@"):
                    emails.append(p)

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for e in emails:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _score_email(email: str) -> int:
    """
    Higher score = better candidate.
    Scoring:
      +10  personal domain (gmail, yahoo, etc.)
      +5   valid MX records
      -100 bad syntax / disposable / no MX
    Position priority (Email 1 vs 2 vs 3) is handled by the caller
    preferring earlier emails at equal scores.
    """
    email = email.strip().lower()
    valid, reason = validate_email(email)
    if not valid:
        return -100

    score = 5  # has valid MX
    domain = email.split("@")[-1]
    if domain in PERSONAL_EMAIL_DOMAINS:
        score += 10
    return score


def _pick_best_email_validated(emails: list[str]) -> tuple[str, str]:
    """
    Pick the best email from the ordered list. Returns (email, reason).
    Strategy:
      1. Score each email (validation + personal domain bonus)
      2. Among tied scores, prefer the one that appeared earlier (PropStream priority)
      3. Skip emails that fail validation entirely
    """
    if not emails:
        return "", "no_emails"

    best_email = ""
    best_score = -999
    best_reason = "no_valid_emails"

    for email in emails:
        score = _score_email(email)
        if score > best_score:
            best_score = score
            best_email = email
            _, reason = validate_email(email)
            best_reason = reason

    if best_score <= -100:
        return "", best_reason

    return best_email, "ok"


# ---------------------------------------------------------------------------
# Phone helpers
# ---------------------------------------------------------------------------

def _extract_phones(contact_row: pd.Series) -> list[str]:
    phones: list[str] = []
    for col, value in contact_row.items():
        col_l = str(col).lower()
        if "phone" not in col_l and col_l not in {"mobile", "landline", "other"}:
            continue
        p = _phone(value)
        if not p:
            continue
        # Basic sanity for US numbers: 10+ digits
        if len(p) < 10:
            continue
        phones.append(p)

    # preserve order, remove duplicates
    seen: set[str] = set()
    out: list[str] = []
    for p in phones:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _is_dnc(value: object) -> bool:
    v = _s(value).strip().lower()
    if not v:
        return False
    return v in {"dnc", "yes", "true", "1", "y"}


def _pick_best_phone_from_row(contact_row: pd.Series) -> str:
    """
    Prefer Phone 1..5 that are NOT marked DNC (when those columns exist).
    Falls back to the first phone found in any phone-like column.
    """
    # Common PropStream contact export format
    phone_cols = [f"Phone {i}" for i in range(1, 6)]
    if any(c in contact_row for c in phone_cols):
        for i in range(1, 6):
            phone = _phone(contact_row.get(f"Phone {i}", ""))
            if not phone or len(phone) < 10:
                continue
            if _is_dnc(contact_row.get(f"Phone {i} DNC", "")):
                continue
            return phone
        # If everything is DNC (or invalid), return blank to respect compliance.
        return ""

    # Fallback: no explicit Phone 1..5 columns
    phones = _extract_phones(contact_row)
    return phones[0] if phones else ""


def _get_first_existing(row: pd.Series, candidates: list[str]) -> str:
    for c in candidates:
        if c in row:
            v = _s(row.get(c, ""))
            if v:
                return v
    return ""


# ---------------------------------------------------------------------------
# Core build: ONE lead per property
# ---------------------------------------------------------------------------

def build_output_csv(
    contacts_df: pd.DataFrame,
    properties_df: pd.DataFrame,
    *,
    run_verification: bool = True,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Merge PropStream contact/skip-trace export with property export.

    Key behavior:
      - ONLY properties in the property export get a row (contacts-only = dropped)
      - ONE lead per property address (best contact chosen)
      - Email validated: syntax + disposable check + MX lookup
      - Deduped by email at the end

    Returns (output_df, stats_dict).
    """
    # Build property lookup by normalized street address
    properties_df = properties_df.copy()
    properties_df["_match_street"] = properties_df.apply(
        lambda r: _normalize_street(
            _get_first_existing(r, ["Address", "Property Address", "Street Address"])
        ),
        axis=1,
    )
    prop_lookup: dict[str, pd.Series] = {}
    for _, prow in properties_df.iterrows():
        key = _s(prow.get("_match_street", ""))
        if not key:
            continue
        if key not in prop_lookup:
            prop_lookup[key] = prow

    # Group contacts by normalized street address
    contacts_df = contacts_df.copy()
    contacts_df["_match_street"] = contacts_df.apply(
        lambda r: _normalize_street(
            _get_first_existing(r, ["Street Address", "Property Address", "Address"])
        ),
        axis=1,
    )

    # Build {address: [list of contact rows]} only for addresses in the property file
    addr_contacts: dict[str, list[pd.Series]] = {}
    for _, crow in contacts_df.iterrows():
        key = _s(crow.get("_match_street", ""))
        if not key:
            continue
        if key not in prop_lookup:
            continue  # ← THIS is the key filter: skip contacts without a matching property
        addr_contacts.setdefault(key, []).append(crow)

    stats = {
        "total_properties": len(prop_lookup),
        "properties_with_contacts": 0,
        "properties_no_contact": 0,
        "contacts_evaluated": 0,
        "emails_bad_syntax": 0,
        "emails_disposable": 0,
        "emails_no_mx": 0,
        "emails_valid": 0,
        "final_rows": 0,
        "deduped_emails": 0,
    }

    rows: list[dict[str, str]] = []
    total_addresses = len(prop_lookup)
    processed = 0

    for addr_key, prow in prop_lookup.items():
        processed += 1
        if progress_callback and processed % 50 == 0:
            progress_callback(processed / total_addresses)

        contact_rows = addr_contacts.get(addr_key, [])
        if not contact_rows:
            stats["properties_no_contact"] += 1
            continue

        stats["properties_with_contacts"] += 1

        # Pick the best contact for this property:
        # Evaluate each contact row's best email, pick the highest-scoring one.
        best_candidate = None  # (email, phone, first, last, company, score)

        for crow in contact_rows:
            stats["contacts_evaluated"] += 1
            emails = _extract_emails_ordered(crow)

            if run_verification:
                email, reason = _pick_best_email_validated(emails)
                if reason == "bad_syntax":
                    stats["emails_bad_syntax"] += 1
                elif reason == "disposable":
                    stats["emails_disposable"] += 1
                elif reason == "no_mx":
                    stats["emails_no_mx"] += 1
            else:
                # Lightweight: just pick first personal email or first email
                email = ""
                for e in emails:
                    domain = e.split("@")[-1].lower()
                    if domain in PERSONAL_EMAIL_DOMAINS:
                        email = e
                        break
                if not email and emails:
                    email = emails[0]

            if not email:
                continue

            stats["emails_valid"] += 1
            score = _score_email(email) if run_verification else 0
            phone = _pick_best_phone_from_row(crow)
            first_name = _get_first_existing(
                crow, ["First Name", "Owner 1 First Name", "FirstName", "first_name"]
            )
            last_name = _get_first_existing(
                crow, ["Last Name", "Owner 1 Last Name", "LastName", "last_name"]
            )
            company_name = _get_first_existing(
                crow, ["Company Name", "Company", "company_name"]
            )

            if best_candidate is None or score > best_candidate[5]:
                best_candidate = (email, phone, first_name, last_name, company_name, score)

        if best_candidate is None:
            continue

        email, phone, first_name, last_name, company_name, _ = best_candidate

        # Property details from the property export
        prop_address = _s(
            _get_first_existing(prow, ["Address", "Property Address", "Street Address"])
        )
        prop_city = _get_first_existing(prow, ["City"])
        prop_state = _get_first_existing(prow, ["State"])
        prop_zip = _zip(_get_first_existing(prow, ["Zip"]))
        prop_value = _clean_number(
            _get_first_existing(prow, ["Est. Value", "Est Value", "Estimated Value"])
        )
        prop_beds = _clean_number(
            _get_first_existing(prow, ["Bedrooms", "Beds"])
        )
        prop_baths = _clean_number(
            _get_first_existing(prow, ["Total Bathrooms", "Bathrooms", "Baths"])
        )
        prop_sqft = _clean_number(
            _get_first_existing(prow, ["Building Sqft", "Sqft", "Square Feet"])
        )
        mls_status = _s(
            _get_first_existing(prow, ["MLS Status", "Mls Status"])
        )
        mls_amount = _clean_number(
            _get_first_existing(prow, ["MLS Amount", "Mls Amount"])
        )
        est_equity = _clean_number(
            _get_first_existing(prow, ["Est. Equity", "Est Equity", "Estimated Equity"])
        )
        prop_full_address = ", ".join(
            p for p in [prop_address, prop_city, prop_state, prop_zip] if p
        )

        rows.append(
            {
                "email": email.lower(),
                "first_name": _s(first_name),
                "last_name": _s(last_name),
                "phone": _s(phone),
                "company_name": _s(company_name),
                "property_address": prop_address,
                "property_city": _s(prop_city),
                "property_state": _s(prop_state),
                "property_zip": _s(prop_zip),
                "property_full_address": _s(prop_full_address),
                "property_value": _s(prop_value),
                "property_bedrooms": _s(prop_beds),
                "property_bathrooms": _s(prop_baths),
                "property_sqft": _s(prop_sqft),
                "mls_status": _s(mls_status),
                "mls_amount": _s(mls_amount),
                "est_equity": _s(est_equity),
            }
        )

    if progress_callback:
        progress_callback(1.0)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Deduplicate by email (same person might own multiple properties → one email)
    pre_dedup = len(out)
    out = out.drop_duplicates(subset=["email"], keep="first")
    stats["deduped_emails"] = pre_dedup - len(out)
    stats["final_rows"] = len(out)

    out = out.fillna("")
    return out, stats


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.caption(
    "Upload **two PropStream exports** (Contact/Skip Trace + Property Export). "
    "The app will auto-detect which is which, pick **one lead per property**, "
    "validate emails, and generate a clean lead CSV."
)

uploaded_files = st.file_uploader(
    "Upload both exports (CSV or Excel)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)


def _read_tabular_file(uploaded_file) -> pd.DataFrame:
    """
    Read either CSV or XLSX into a dataframe of strings.
    Streamlit's UploadedFile is file-like; pandas can read it directly.
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")
        return df.fillna("")
    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return df.fillna("")


def _score_contacts_columns(cols: list[str]) -> int:
    cols_l = [c.lower() for c in cols]
    score = 0
    strong_any = [
        "phone 1", "phone 2", "phone 3", "phone 4", "phone 5",
        "phone 1 dnc", "email 1", "email 2", "email 3", "email 4",
    ]
    score += sum(6 for s in strong_any if s in cols_l)
    weak_any = [
        "first name", "last name", "company name",
        "street address", "mail street address",
    ]
    score += sum(2 for s in weak_any if s in cols_l)
    return score


def _score_properties_columns(cols: list[str]) -> int:
    cols_l = [c.lower() for c in cols]
    score = 0
    strong_any = [
        "apn", "mls status", "mls amount", "est. value", "est. equity",
        "building sqft", "total bathrooms", "bedrooms", "owner occupied",
    ]
    score += sum(6 for s in strong_any if s in cols_l)
    weak_any = ["address", "city", "state", "zip", "mailing address"]
    score += sum(2 for s in weak_any if s in cols_l)
    return score


def _detect_roles(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    a_cols = list(df_a.columns)
    b_cols = list(df_b.columns)

    a_c = _score_contacts_columns(a_cols)
    a_p = _score_properties_columns(a_cols)
    b_c = _score_contacts_columns(b_cols)
    b_p = _score_properties_columns(b_cols)

    map1 = a_c + b_p
    map2 = a_p + b_c

    if map1 > map2:
        return df_a, df_b, f"Detected file A as contacts (score {a_c}), file B as properties (score {b_p})."
    if map2 > map1:
        return df_b, df_a, f"Detected file B as contacts (score {b_c}), file A as properties (score {a_p})."

    a_explicit = sum(1 for c in a_cols if "email" in c.lower() or "phone" in c.lower())
    b_explicit = sum(1 for c in b_cols if "email" in c.lower() or "phone" in c.lower())
    if a_explicit >= b_explicit:
        return df_a, df_b, "Detection was ambiguous; defaulted file A to contacts based on email/phone column count."
    return df_b, df_a, "Detection was ambiguous; defaulted file B to contacts based on email/phone column count."


if uploaded_files and len(uploaded_files) == 2:
    try:
        df1 = _read_tabular_file(uploaded_files[0])
        df2 = _read_tabular_file(uploaded_files[1])
    except Exception as e:
        st.error(
            "Couldn't read one of the uploaded files. "
            "This app currently supports **.csv** and **.xlsx** (Excel) exports."
        )
        st.code(str(e)[:2000])
        st.stop()

    contacts_df, properties_df, explain = _detect_roles(df1, df2)
    st.caption(f"Auto-detect: {explain}")

    run_verification = st.checkbox(
        "Verify emails (MX lookup — recommended)",
        value=True,
        help="Checks each email domain has a working mail server. Adds ~30-60 seconds for large files but removes dead emails.",
    )

    st.divider()

    progress_bar = st.progress(0, text="Building leads…")

    def _update_progress(pct: float):
        progress_bar.progress(min(pct, 1.0), text=f"Processing properties… {int(pct * 100)}%")

    output_df, stats = build_output_csv(
        contacts_df=contacts_df,
        properties_df=properties_df,
        run_verification=run_verification,
        progress_callback=_update_progress,
    )

    progress_bar.progress(1.0, text="Done!")

    # Stats display
    st.subheader("Build Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties in file", f"{stats['total_properties']:,}")
    c2.metric("With a contact match", f"{stats['properties_with_contacts']:,}")
    c3.metric("No contact found", f"{stats['properties_no_contact']:,}")
    c4.metric("Final leads", f"{stats['final_rows']:,}")

    if run_verification:
        st.caption("Email verification results:")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Valid emails", f"{stats['emails_valid']:,}")
        v2.metric("Bad syntax", f"{stats['emails_bad_syntax']:,}")
        v3.metric("Disposable", f"{stats['emails_disposable']:,}")
        v4.metric("Dead domain (no MX)", f"{stats['emails_no_mx']:,}")

    if stats["deduped_emails"] > 0:
        st.caption(
            f"Removed **{stats['deduped_emails']:,}** duplicate emails "
            f"(same person owns multiple properties — kept first)."
        )

    # Warn if very few contacts had emails (common when "All Contacts" is used
    # instead of the skip-trace export for the specific property list)
    if stats["properties_with_contacts"] > 0:
        email_rate = stats["emails_valid"] / stats["properties_with_contacts"]
        if email_rate < 0.10:
            st.warning(
                f"⚠️ Only **{stats['emails_valid']:,}** out of "
                f"**{stats['properties_with_contacts']:,}** matched contacts "
                f"had an email address ({email_rate:.1%}). "
                f"This usually means the contact export hasn't been skip-traced "
                f"for these specific properties.\n\n"
                f"**Tip:** In PropStream, run a skip-trace on this property list, "
                f"then export the contacts for *just that list* instead of \"All Contacts\"."
            )

    st.subheader("Output")
    st.dataframe(output_df, use_container_width=True, height=520)

    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    base_name = uploaded_files[0].name.rsplit(".", 1)[0]
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"{base_name}_leads.csv",
        mime="text/csv",
        use_container_width=True,
    )
elif uploaded_files and len(uploaded_files) != 2:
    st.info("Please upload exactly 2 files (CSV or XLSX).")
