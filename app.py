import streamlit as st
import pandas as pd
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


def _extract_emails(contact_row: pd.Series) -> list[str]:
    emails: list[str] = []
    for col, value in contact_row.items():
        if "email" not in str(col).lower():
            continue
        v = _s(value).lower()
        if not v:
            continue
        # Some exports can contain multiple emails in one cell.
        parts = re.split(r"[,\s;]+", v)
        for p in parts:
            p = p.strip().lower()
            if not p:
                continue
            if "@" not in p:
                continue
            if p.startswith("@") or p.endswith("@"):
                continue
            emails.append(p)
    # preserve order, remove duplicates
    seen = set()
    out = []
    for e in emails:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _pick_best_email(emails: list[str]) -> str:
    if not emails:
        return ""
    for e in emails:
        domain = e.split("@")[-1].lower()
        if domain in PERSONAL_EMAIL_DOMAINS:
            return e
    return emails[0]


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
    seen = set()
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
    return _pick_best_phone(_extract_phones(contact_row))


def _pick_best_phone(phones: list[str]) -> str:
    return phones[0] if phones else ""


def _get_first_existing(row: pd.Series, candidates: list[str]) -> str:
    for c in candidates:
        if c in row:
            v = _s(row.get(c, ""))
            if v:
                return v
    return ""


def build_output_csv(contacts_df: pd.DataFrame, properties_df: pd.DataFrame, *, dedupe_by_email: bool = True) -> pd.DataFrame:
    """
    Merge PropStream contact/skip-trace export with property export and produce a clean lead CSV.
    - Match by street address
    - Pick ONE prioritized email
    - Optionally deduplicate by email
    """
    # Build property lookup by normalized street address
    properties_df = properties_df.copy()
    properties_df["_match_street"] = properties_df.apply(
        lambda r: _normalize_street(_get_first_existing(r, ["Address", "Property Address", "Street Address"])),
        axis=1,
    )
    prop_lookup = {}
    for _, prow in properties_df.iterrows():
        key = _s(prow.get("_match_street", ""))
        if not key:
            continue
        # keep first occurrence
        if key not in prop_lookup:
            prop_lookup[key] = prow

    rows = []
    for _, crow in contacts_df.iterrows():
        contact_street = _get_first_existing(
            crow,
            [
                "Street Address",
                "Property Address",
                "Address",
            ],
        )
        key = _normalize_street(contact_street)

        emails = _extract_emails(crow)
        email = _pick_best_email(emails).lower()
        if not email:
            # Email is required; skip rows without an email
            continue
        phone = _pick_best_phone_from_row(crow)

        first_name = _get_first_existing(crow, ["First Name", "Owner 1 First Name", "FirstName", "first_name"])
        last_name = _get_first_existing(crow, ["Last Name", "Owner 1 Last Name", "LastName", "last_name"])
        company_name = _get_first_existing(crow, ["Company Name", "Company", "company_name"])

        # Pull property details if we have a match
        prow = prop_lookup.get(key)
        prop_city = _get_first_existing(crow, ["City"])  # default to contact row
        prop_state = _get_first_existing(crow, ["State"])
        prop_zip = _zip(_get_first_existing(crow, ["Zip"]))

        prop_value = ""
        prop_beds = ""
        prop_baths = ""
        prop_sqft = ""
        mls_status = ""
        mls_amount = ""
        est_equity = ""

        if prow is not None:
            prop_city = _get_first_existing(prow, ["City"]) or prop_city
            prop_state = _get_first_existing(prow, ["State"]) or prop_state
            prop_zip = _zip(_get_first_existing(prow, ["Zip"])) or prop_zip

            prop_value = _clean_number(_get_first_existing(prow, ["Est. Value", "Est Value", "Estimated Value"]))
            prop_beds = _clean_number(_get_first_existing(prow, ["Bedrooms", "Beds"]))
            prop_baths = _clean_number(_get_first_existing(prow, ["Total Bathrooms", "Bathrooms", "Baths"]))
            prop_sqft = _clean_number(_get_first_existing(prow, ["Building Sqft", "Sqft", "Square Feet"]))
            mls_status = _s(_get_first_existing(prow, ["MLS Status", "Mls Status"]))
            mls_amount = _clean_number(_get_first_existing(prow, ["MLS Amount", "Mls Amount"]))
            est_equity = _clean_number(_get_first_existing(prow, ["Est. Equity", "Est Equity", "Estimated Equity"]))

        prop_address = _s(contact_street)
        prop_full_address = ", ".join(p for p in [prop_address, prop_city, prop_state, prop_zip] if p)

        rows.append(
            {
                "email": email,
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

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # Deduplicate by email address (keep first)
    if dedupe_by_email:
        out = out.drop_duplicates(subset=["email"], keep="first")
    # Final cleaning: replace any NaN remnants
    out = out.fillna("")
    return out


def summarize_build(contacts_df: pd.DataFrame, properties_df: pd.DataFrame, *, dedupe_by_email: bool) -> dict[str, Any]:
    """
    Explain how many rows we keep/drop:
    - Rows without any email are excluded
    - Optionally dedupe by chosen email
    - Match-rate to property export by street address
    """
    total_contacts = int(len(contacts_df))

    # Property keys
    prop_keys: set[str] = set()
    for _, prow in properties_df.iterrows():
        p_addr = _get_first_existing(prow, ["Address", "Property Address", "Street Address"])
        k = _normalize_street(p_addr)
        if k:
            prop_keys.add(k)

    selected_emails: list[str] = []
    matched_property_count = 0

    for _, crow in contacts_df.iterrows():
        emails = _extract_emails(crow)
        chosen = _pick_best_email(emails).lower().strip()
        if not chosen:
            continue

        selected_emails.append(chosen)

        c_addr = _get_first_existing(crow, ["Street Address", "Property Address", "Address"])
        c_key = _normalize_street(c_addr)
        if c_key and c_key in prop_keys:
            matched_property_count += 1

    with_any_email = len(selected_emails)
    unique_selected = len(set(selected_emails))
    duplicates_by_email = with_any_email - unique_selected
    final_rows = unique_selected if dedupe_by_email else with_any_email

    return {
        "total_contacts": total_contacts,
        "with_any_email": with_any_email,
        "missing_email": total_contacts - with_any_email,
        "duplicates_by_email": duplicates_by_email,
        "final_rows": final_rows,
        "matched_property_count": matched_property_count if dedupe_by_email else matched_property_count,
    }


st.caption(
    "Upload **two PropStream exports** (Contact/Skip Trace + Property Export). "
    "The app will auto-detect which is which and generate a clean lead CSV."
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
    # Ensure the underlying buffer is at the start (Streamlit sometimes reuses the object)
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".xlsx"):
        # Read first sheet by default
        df = pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")
        return df.fillna("")
    # Default: CSV
    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return df.fillna("")


def _score_contacts_columns(cols: list[str]) -> int:
    cols_l = [c.lower() for c in cols]
    score = 0
    # Strong signals for contact/skip-trace export
    strong_any = [
        "phone 1",
        "phone 2",
        "phone 3",
        "phone 4",
        "phone 5",
        "phone 1 dnc",
        "email 1",
        "email 2",
        "email 3",
        "email 4",
    ]
    score += sum(6 for s in strong_any if s in cols_l)

    # Weaker/general signals
    weak_any = [
        "first name",
        "last name",
        "company name",
        "street address",
        "mail street address",
    ]
    score += sum(2 for s in weak_any if s in cols_l)
    return score


def _score_properties_columns(cols: list[str]) -> int:
    cols_l = [c.lower() for c in cols]
    score = 0
    # Strong signals for property export
    strong_any = [
        "apn",
        "mls status",
        "mls amount",
        "est. value",
        "est. equity",
        "building sqft",
        "total bathrooms",
        "bedrooms",
        "owner occupied",
    ]
    score += sum(6 for s in strong_any if s in cols_l)

    # Common property address fields
    weak_any = [
        "address",
        "city",
        "state",
        "zip",
        "mailing address",
    ]
    score += sum(2 for s in weak_any if s in cols_l)
    return score


def _detect_roles(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Returns (contacts_df, properties_df, explanation).
    Deterministic: picks the mapping with the higher combined score.
    """
    a_cols = list(df_a.columns)
    b_cols = list(df_b.columns)

    a_c = _score_contacts_columns(a_cols)
    a_p = _score_properties_columns(a_cols)
    b_c = _score_contacts_columns(b_cols)
    b_p = _score_properties_columns(b_cols)

    map1 = a_c + b_p  # A=contacts, B=properties
    map2 = a_p + b_c  # A=properties, B=contacts

    if map1 > map2:
        return df_a, df_b, f"Detected file A as contacts (score {a_c}), file B as properties (score {b_p})."
    if map2 > map1:
        return df_b, df_a, f"Detected file B as contacts (score {b_c}), file A as properties (score {a_p})."

    # Tie-breaker: prefer the one with more explicit Email/Phone columns as contacts
    a_explicit = sum(1 for c in a_cols if "email" in c.lower() or "phone" in c.lower())
    b_explicit = sum(1 for c in b_cols if "email" in c.lower() or "phone" in c.lower())
    if a_explicit >= b_explicit:
        return df_a, df_b, "Detection was ambiguous; defaulted file A to contacts based on email/phone column count."
    return df_b, df_a, "Detection was ambiguous; defaulted file B to contacts based on email/phone column count."


if uploaded_files and len(uploaded_files) == 2:
    # Read with dtype=str to prevent numeric coercion (.0 issues)
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

    dedupe_by_email = st.checkbox(
        "Deduplicate by email (recommended)",
        value=True,
        help="This reduces duplicates and matches your spec.",
    )
    summary = summarize_build(contacts_df=contacts_df, properties_df=properties_df, dedupe_by_email=dedupe_by_email)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Contacts in file", f"{summary['total_contacts']:,}")
    m2.metric("With an email", f"{summary['with_any_email']:,}")
    m3.metric("Missing email (skipped)", f"{summary['missing_email']:,}")
    m4.metric("Duplicate emails", f"{summary['duplicates_by_email']:,}")

    st.caption(
        f"Final leads to upload/download: **{summary['final_rows']:,}** "
        + ("(deduped by email)." if dedupe_by_email else "(duplicates kept).")
    )

    output_df = build_output_csv(contacts_df=contacts_df, properties_df=properties_df, dedupe_by_email=dedupe_by_email)

    st.subheader("Output")
    st.dataframe(output_df, width="stretch", height=520)

    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    base_name = uploaded_files[0].name.rsplit(".", 1)[0]
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"{base_name}_leads.csv",
        mime="text/csv",
        width="stretch",
    )
elif uploaded_files and len(uploaded_files) != 2:
    st.info("Please upload exactly 2 files (CSV or XLSX).")
