"""
OHI Toolkit - combined Streamlit app
-------------------------------------
One app, two tools, with a sidebar navigation to switch between them:

  1. OHI Query API Builder  - builds `q` strings for the OHI Query API DSL and
                              the equivalent Groovy SearchBuilder code.
  2. JSON API Runner        - batch-runs an API by substituting {{variables}}
                              into a JSON template, one call per data row.

Run locally with:
    pip install streamlit requests pandas
    python -m streamlit run ohi_toolkit.py
"""

import json
import re
import time
from io import StringIO
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="OHI Toolkit", page_icon="\U0001f9f0", layout="wide")


# ===========================================================================
# SHARED: keep widget values when the user navigates between tools
# ===========================================================================
# Streamlit discards the value of any widget that isn't rendered in a run, so
# switching pages would normally wipe what the user typed. Re-assigning each
# key to itself (before the page renders) keeps it alive. File uploaders
# can't be handled this way, so the API Runner caches parsed rows instead.

PERSISTED_KEYS = [
    # Query builder
    "qb_resource_path", "qb_class_name", "qb_metadata_url",
    "qb_offset", "qb_limit", "qb_order_by", "qb_total_results",
    "qb_group_by", "qb_aggregate",
    "qb_auth_type", "qb_token", "qb_username", "qb_password",
    "qb_header_name", "qb_header_value", "qb_extra_headers",
    # API runner
    "ar_template", "ar_url", "ar_method", "ar_timeout", "ar_auth_type",
    "ar_token", "ar_username", "ar_password", "ar_header_name",
    "ar_header_value", "ar_extra_headers", "ar_delay", "ar_max_chars",
    "ar_data_mode",
]


WIDGET_DEFAULTS = {
    "qb_resource_path": "/generic/claims/search",
    "qb_class_name": "Claim",
    "qb_offset": 0,
    "qb_limit": 50,
    "qb_total_results": False,
    "ar_timeout": 30,
    "ar_delay": 0.0,
    "ar_max_chars": 0,
    "ar_data_mode": "Upload CSV / JSON",
}


def keep_widget_state():
    for k, v in WIDGET_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for k in PERSISTED_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


# ===========================================================================
# TOOL 1: OHI QUERY API BUILDER
# ===========================================================================
# Builds a query string for the Oracle Health Insurance (OHI) Query API DSL,
# per the official grammar:
#
#     q := expression
#     expression := simple_expression | complex_expression
#     simple_expression := field_name.operator(value, value, ...)
#     complex_expression := expression.complex_operator.expression
#     complex_operator := and | or
#
# Operators: eq, eqic, neq, lt, lte, gt, gte, in, bet, like, likeic, nin
#
# Also generates the equivalent Groovy SearchBuilder code (Dynamic Logic
# Search Library).

# Operator metadata: how many values each operator takes.
# arity: "one", "two" (bet), "many" (in/nin), "like" (1 required + 1 optional escape)
OPERATORS = {
    "eq (equals / isNull if value is null)": ("eq", "one"),
    "eqic (equals, ignore case)": ("eqic", "one"),
    "neq (not equal / isNotNull if value is null)": ("neq", "one"),
    "lt (less than)": ("lt", "one"),
    "lte (less than or equal)": ("lte", "one"),
    "gt (greater than)": ("gt", "one"),
    "gte (greater than or equal)": ("gte", "one"),
    "in (matches one of several values)": ("in", "many"),
    "nin (matches none of several values)": ("nin", "many"),
    "bet (between two values)": ("bet", "two"),
    "like (pattern match, % and _ wildcards)": ("like", "like"),
    "likeic (pattern match, ignore case)": ("likeic", "like"),
}

# Maps the REST Query API operator token to the Groovy SearchBuilder method name.
GROOVY_OPERATOR_MAP = {
    "eq": "eq",
    "eqic": "eqIgnoreCase",
    "neq": "neq",
    "lt": "lt",
    "lte": "lte",
    "gt": "gt",
    "gte": "gte",
    "in": "in",
    "nin": "notIn",
    "bet": "between",
    "like": "like",
    "likeic": "likeIgnoreCase",
}

VALUE_TYPES = ["String", "Number", "Date", "null"]

# Fallback presets used only when no metadata has been fetched yet.
DEFAULT_FIELD_PRESETS = [
    "(custom field)",
    "code",
    "status",
    "brand.code",
    "claimNumber",
    "servicedMember.code",
    "providerId",
    "flexCodeSystem.code",
    "startDate",
    "endDate",
]


def init_query_builder_state():
    # Each group = {"connector": "and"/"or" (how this group joins the PREVIOUS group),
    #               "internal_connector": "and"/"or" (how conditions inside this group join),
    #               "conditions": [ {field, operator, arity, values:[...], escape} ]}
    st.session_state.setdefault("groups", [])
    st.session_state.setdefault("dynamic_fields", [])   # populated from fetched metadata
    st.session_state.setdefault("metadata_source", "")  # last URL fetched, shown for reference
    st.session_state.setdefault("metadata_error", "")


def escape_string(raw: str) -> str:
    """OHI escapes a literal apostrophe by doubling it, e.g. D'Souza -> D''Souza."""
    return raw.replace("'", "''")


def format_single_value(raw: str, value_type: str) -> str:
    raw = raw.strip()
    if value_type == "null":
        return "null"
    if value_type == "Number":
        return raw
    # String or Date both get single-quoted, with apostrophe escaping
    return f"'{escape_string(raw)}'"


def build_condition_string(cond: dict) -> str:
    field = cond["field"]
    operator = cond["operator"]
    arity = cond["arity"]
    value_type = cond["value_type"]

    if arity == "one":
        val = format_single_value(cond["values"][0], value_type)
        return f"{field}.{operator}({val})"

    if arity == "two":
        v1 = format_single_value(cond["values"][0], value_type)
        v2 = format_single_value(cond["values"][1], value_type)
        return f"{field}.{operator}({v1},{v2})"

    if arity == "many":
        formatted = [format_single_value(v, value_type) for v in cond["values"] if v.strip() != ""]
        return f"{field}.{operator}({','.join(formatted)})"

    if arity == "like":
        pattern = format_single_value(cond["values"][0], "String")
        if cond.get("escape"):
            esc = format_single_value(cond["escape"], "String")
            return f"{field}.{operator}({pattern},{esc})"
        return f"{field}.{operator}({pattern})"

    return f"{field}.{operator}()"


def build_group_string(group: dict) -> str:
    parts = [build_condition_string(c) for c in group["conditions"]]
    joined = f".{group['internal_connector']}.".join(parts)
    # Wrap in parentheses only if there's more than one condition.
    if len(parts) > 1:
        return f"({joined})"
    return joined


def build_full_query() -> str:
    rendered = []
    for i, group in enumerate(st.session_state.groups):
        group_str = build_group_string(group)
        if i == 0:
            rendered.append(group_str)
        else:
            rendered.append(f".{group['connector']}.{group_str}")
    return "".join(rendered)


# --- Groovy SearchBuilder code generation ----------------------------------
# Every dotted path segment before the last one becomes a .join("...") call,
# and the final segment is the attribute passed to .by("...").
# .group()/.end() wrap a clause when it has more than one condition.

def escape_groovy_string(raw: str) -> str:
    """Groovy double-quoted strings: escape backslashes then double quotes."""
    return raw.replace("\\", "\\\\").replace('"', '\\"')


def format_groovy_value(raw: str, value_type: str) -> str:
    raw = raw.strip()
    if value_type == "Number":
        return raw
    # String and Date literals are both double-quoted in the doc's examples.
    return f'"{escape_groovy_string(raw)}"'


def build_condition_groovy_segments(cond: dict) -> list:
    field = cond["field"]
    operator = cond["operator"]
    arity = cond["arity"]
    value_type = cond["value_type"]
    groovy_op = GROOVY_OPERATOR_MAP.get(operator, operator)

    # "claimLines.procedureCode" -> join("claimLines"), by("procedureCode").
    # An outer-join marker like "brand(+)" is passed straight through to join().
    path_parts = [p for p in field.split(".") if p != ""]
    if not path_parts:
        path_parts = [field]
    join_parts = path_parts[:-1]
    attribute = path_parts[-1]

    segments = [f'.join("{j}")' for j in join_parts]

    if arity == "one":
        if value_type == "null":
            args = "null"  # eq(null)/neq(null) checks isNull/isNotNull per the Search Library docs
        else:
            args = format_groovy_value(cond["values"][0], value_type)
        segments.append(f'.by("{attribute}").{groovy_op}({args})')

    elif arity == "two":
        v1 = format_groovy_value(cond["values"][0], value_type)
        v2 = format_groovy_value(cond["values"][1], value_type)
        segments.append(f'.by("{attribute}").{groovy_op}({v1}, {v2})')

    elif arity == "many":
        formatted = [format_groovy_value(v, value_type) for v in cond["values"] if v.strip() != ""]
        segments.append(f'.by("{attribute}").{groovy_op}({", ".join(formatted)})')

    elif arity == "like":
        pattern = format_groovy_value(cond["values"][0], "String")
        if cond.get("escape"):
            esc = format_groovy_value(cond["escape"], "String")
            segments.append(f'.by("{attribute}").{groovy_op}({pattern}, {esc})')
        else:
            segments.append(f'.by("{attribute}").{groovy_op}({pattern})')

    else:
        segments.append(f'.by("{attribute}").{groovy_op}()')

    return segments


def build_group_groovy_segments(group: dict) -> list:
    conditions = group["conditions"]
    segments = []
    wrap = len(conditions) > 1
    if wrap:
        segments.append(".group()")
    for i, cond in enumerate(conditions):
        if i > 0:
            segments.append(f'.{group["internal_connector"]}()')
        segments.extend(build_condition_groovy_segments(cond))
    if wrap:
        segments.append(".end()")
    return segments


def parse_order_by_to_groovy(order_by_raw: str) -> list:
    """Best-effort parse of a REST-style orderBy value (e.g. 'code:asc',
    'startDate:desc nulls last') into SearchBuilder .orderBy(...) segments.
    Only the first field is converted."""
    raw = order_by_raw.strip()
    if not raw:
        return []
    first_field = raw.split(",")[0].strip()
    field_part = first_field
    direction = ""
    if ":" in first_field:
        field_part, _, rest = first_field.partition(":")
        rest_lower = rest.strip().lower()
        if rest_lower.startswith("asc"):
            direction = "ascending"
        elif rest_lower.startswith("desc"):
            direction = "descending"

    field_part = field_part.strip()
    nulls_suffix = ""
    lowered = first_field.lower()
    if "nulls first" in lowered:
        nulls_suffix = " NULLS FIRST"
    elif "nulls last" in lowered:
        nulls_suffix = " NULLS LAST"

    segments = [f'.orderBy("{field_part}{nulls_suffix}")']
    if direction:
        segments.append(f".{direction}()")
    return segments


def build_full_groovy(class_name: str, use_pagination: bool = False, page_size: int = None,
                      order_by_raw: str = "") -> str:
    class_name = class_name.strip() or "Claim"
    segments = []
    for i, group in enumerate(st.session_state.groups):
        if i > 0:
            segments.append(f'.{group["connector"]}()')
        segments.extend(build_group_groovy_segments(group))

    segments.extend(parse_order_by_to_groovy(order_by_raw))
    if use_pagination and page_size:
        segments.append(f'.setPageSize({int(page_size)})')
    segments.append(".execute()")

    lines = [f"List<{class_name}> results = new SearchBuilder({class_name}.class)"]
    for seg in segments:
        lines.append(f"  {seg}")
    return "\n".join(lines)


# --- Metadata fetching / field discovery -----------------------------------
# Tolerant flattener for three common shapes:
#   1. JSON-Schema style: {"properties": {"code": {...}, "brand": {"properties": {...}}}}
#   2. A list of field-definition objects with nested "fields"/"children"/"attributes"
#   3. A flat list of plain strings: ["code", "status", "brand.code"]

NAME_KEYS = ("name", "code", "fieldName", "field", "id")
NESTED_LIST_KEYS = ("fields", "children", "attributes", "properties")


def _flatten_schema_properties(props: dict, prefix: str, out: set):
    for key, val in props.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and isinstance(val.get("properties"), dict):
            _flatten_schema_properties(val["properties"], path, out)
        elif isinstance(val, dict) and val.get("type") == "array" and isinstance(val.get("items"), dict) \
                and isinstance(val["items"].get("properties"), dict):
            _flatten_schema_properties(val["items"]["properties"], path, out)
        else:
            out.add(path)


def _flatten_field_list(items, prefix: str, out: set):
    for item in items:
        if isinstance(item, str):
            out.add(f"{prefix}.{item}" if prefix else item)
            continue
        if not isinstance(item, dict):
            continue
        name = None
        for k in NAME_KEYS:
            if item.get(k):
                name = str(item[k])
                break
        if not name:
            continue
        path = f"{prefix}.{name}" if prefix else name

        nested = None
        for k in NESTED_LIST_KEYS:
            if isinstance(item.get(k), list):
                nested = item[k]
                break
        if nested:
            _flatten_field_list(nested, path, out)
        else:
            out.add(path)


def extract_fields_from_metadata(data) -> list:
    out = set()
    if isinstance(data, dict):
        if isinstance(data.get("properties"), dict):
            _flatten_schema_properties(data["properties"], "", out)
        else:
            for k in NESTED_LIST_KEYS + ("data", "result", "results", "items"):
                if isinstance(data.get(k), list):
                    _flatten_field_list(data[k], "", out)
            if not out:
                _flatten_schema_properties(data, "", out)
    elif isinstance(data, list):
        _flatten_field_list(data, "", out)
    return sorted(out)


def fetch_metadata_fields(url: str, headers: dict = None, auth=None, timeout: int = 15):
    """Returns (fields_list, error_message)."""
    try:
        req_headers = {"Accept": "application/json"}
        req_headers.update(headers or {})
        resp = requests.get(url, timeout=timeout, headers=req_headers, auth=auth)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return [], f"Request failed: {e}"
    except ValueError as e:
        return [], f"Response was not valid JSON: {e}"

    fields = extract_fields_from_metadata(data)
    if not fields:
        return [], "Fetched the response but couldn't find any recognizable field names in it."
    return fields, ""


def add_condition_callback():
    """Runs before the rerun, so it can read the entry widgets, add the condition
    and clear the value inputs."""
    ss = st.session_state
    operator, arity = OPERATORS[ss["qb_operator"]]
    field = ss.get("qb_custom_field", "").strip() if ss["qb_field"] == "(custom field)" else ss["qb_field"]

    value_type, values, escape_char = "String", [], ""
    if arity == "one":
        value_type = ss["qb_vtype_one"]
        values = [ss.get("qb_value", "")]
    elif arity == "two":
        value_type = ss["qb_vtype_two"]
        values = [ss.get("qb_from", ""), ss.get("qb_to", "")]
    elif arity == "many":
        value_type = ss["qb_vtype_many"]
        raw_list = ss.get("qb_list", "")
        values = [v.strip() for v in raw_list.split(",")] if raw_list else []
    elif arity == "like":
        values = [ss.get("qb_pattern", "")]
        escape_char = ss.get("qb_escape", "")

    if not field:
        ss["qb_msg"] = "Please provide a field name."
        return
    if arity == "one":
        if value_type != "null" and not values[0].strip():
            ss["qb_msg"] = "Please provide a value, or switch value type to 'null'."
            return
    elif not values or any(v.strip() == "" for v in values):
        ss["qb_msg"] = "Please fill in all required value fields."
        return

    new_cond = {
        "field": field,
        "operator": operator,
        "arity": arity,
        "values": values,
        "value_type": value_type,
        "escape": escape_char.strip() if escape_char else "",
    }

    target = ss.get("qb_target", "New group")
    if target == "New group" or not ss["groups"]:
        ss["groups"].append({
            "connector": ss.get("qb_new_connector", "and"),
            "internal_connector": "and",
            "conditions": [new_cond],
        })
    else:
        group_idx = int(target.split("#")[1]) - 1
        ss["groups"][group_idx]["conditions"].append(new_cond)
        ss["groups"][group_idx]["internal_connector"] = ss.get("qb_internal_connector", "and")

    ss["qb_msg"] = ""
    for k in ("qb_value", "qb_from", "qb_to", "qb_list", "qb_pattern", "qb_escape", "qb_custom_field"):
        ss[k] = ""


def page_query_builder():
    init_query_builder_state()

    # Centered column keeps this page readable under the wide layout.
    _, main, _ = st.columns([1, 3, 1])

    with main:
        st.title("\U0001f50e OHI Query API Builder")
        st.caption(
            "Builds `q` search strings for the OHI Query API DSL "
            "(POST `/generic/{resource}/search`) - and the equivalent Groovy "
            "`SearchBuilder` code for Dynamic Logic."
        )

        st.markdown(
            "**How grouping works below:** add conditions into a group when they belong "
            "together inside parentheses, e.g. `(endDate.eq(null).or.endDate.lt('2015-12-15'))`. "
            "Start a new group when you want a separate bracketed clause joined by `and`/`or` "
            "to the rest, e.g. `...and.code.eq('ABC')`."
        )

        st.divider()

        # --- Resource + metadata-driven field discovery ---------------------
        st.subheader("Resource & available fields")

        resource_path = st.text_input(
            "Resource endpoint",
            key="qb_resource_path",
            help="The POST path this q string will be sent to, e.g. /generic/claims/search",
        )

        groovy_class_name = st.text_input(
            "Groovy object class (for SearchBuilder)",
            key="qb_class_name",
            help="The class passed to `new SearchBuilder(...)`, e.g. Claim, Authorization, ReferenceSheetLine.",
        )

        metadata_url = st.text_input(
            "Metadata URL for this resource (optional)",
            placeholder="e.g. https://your-ohi-host/generic/claims/metadata",
            key="qb_metadata_url",
            help="A link that returns this resource's field metadata as JSON. "
                 "Fetched fields will populate the Field dropdown below.",
        )

        # Environment credentials for the metadata request (same options as the API Runner)
        md_auth_type = st.radio(
            "Environment auth type",
            ["None", "Bearer Token", "Basic Auth", "Custom Header"],
            horizontal=True,
            key="qb_auth_type",
        )

        md_username = md_password = md_token = md_header_name = md_header_value = ""
        if md_auth_type == "Bearer Token":
            md_token = st.text_input("Bearer token", type="password", key="qb_token")
        elif md_auth_type == "Basic Auth":
            mc1, mc2 = st.columns(2)
            md_username = mc1.text_input("Username", key="qb_username")
            md_password = mc2.text_input("Password", type="password", key="qb_password")
        elif md_auth_type == "Custom Header":
            mh1, mh2 = st.columns(2)
            md_header_name = mh1.text_input("Header name", placeholder="X-API-Key", key="qb_header_name")
            md_header_value = mh2.text_input("Header value", type="password", key="qb_header_value")

        md_extra_headers = st.text_area(
            "Extra headers (optional JSON)", height=70,
            placeholder='{"X-Correlation-Id": "builder-01"}', key="qb_extra_headers",
        )

        mcol1, mcol2 = st.columns([1, 1])
        with mcol1:
            if st.button("\U0001f504 Fetch fields from metadata URL", disabled=not metadata_url.strip()):
                md_headers = build_headers(md_auth_type, md_token, md_username, md_password,
                                           md_header_name, md_header_value, md_extra_headers)
                md_headers.pop("Content-Type", None)  # not needed on a GET
                md_auth = (md_username, md_password) if md_auth_type == "Basic Auth" else None
                fields, error = fetch_metadata_fields(metadata_url.strip(), md_headers, md_auth)
                st.session_state.dynamic_fields = fields
                st.session_state.metadata_error = error
                st.session_state.metadata_source = metadata_url.strip() if not error else ""
        with mcol2:
            if st.session_state.dynamic_fields and st.button("\U0001f9f9 Clear fetched fields"):
                st.session_state.dynamic_fields = []
                st.session_state.metadata_source = ""
                st.session_state.metadata_error = ""

        if st.session_state.metadata_error:
            st.warning(st.session_state.metadata_error)
        elif st.session_state.dynamic_fields:
            st.success(
                f"Loaded {len(st.session_state.dynamic_fields)} field(s) from "
                f"`{st.session_state.metadata_source}`."
            )
            with st.expander("Show fetched fields"):
                st.code("\n".join(st.session_state.dynamic_fields), language="text")
        else:
            st.caption("No metadata fetched yet - using the built-in default field list below.")

        # "(custom field)" first, then fetched metadata if present, else the static defaults.
        active_fields = (
            st.session_state.dynamic_fields if st.session_state.dynamic_fields else DEFAULT_FIELD_PRESETS[1:]
        )
        field_presets = ["(custom field)"] + active_fields

        st.divider()

        # --- Add a condition to a group -------------------------------------
        st.subheader("Add a condition")

        group_options = ["New group"] + [f"Existing group #{i + 1}" for i in range(len(st.session_state.groups))]
        if st.session_state.get("qb_target") not in group_options:
            st.session_state["qb_target"] = "New group"
        if st.session_state.groups:
            st.radio("Add this condition to:", group_options, horizontal=True, key="qb_target")
        target = st.session_state.get("qb_target", "New group")

        col1, col2 = st.columns([1, 1])
        with col1:
            field_choice = st.selectbox("Field", field_presets, key="qb_field")
        with col2:
            operator_label = st.selectbox("Operator", list(OPERATORS.keys()), key="qb_operator")

        if field_choice == "(custom field)":
            st.text_input(
                "Custom field name",
                placeholder="e.g. person.bloodGroup(+).code  or  claimLine.procedureCode",
                key="qb_custom_field",
            )

        operator, arity = OPERATORS[operator_label]

        if arity == "one":
            vcol1, vcol2 = st.columns([1, 1])
            with vcol1:
                vt = st.selectbox("Value type", VALUE_TYPES, key="qb_vtype_one")
            with vcol2:
                st.text_input("Value", placeholder="e.g. ABC or 123", disabled=(vt == "null"), key="qb_value")

        elif arity == "two":
            st.selectbox("Value type", [t for t in VALUE_TYPES if t != "null"], key="qb_vtype_two")
            vcol1, vcol2 = st.columns([1, 1])
            with vcol1:
                st.text_input("From value", placeholder="e.g. 6", key="qb_from")
            with vcol2:
                st.text_input("To value", placeholder="e.g. 10", key="qb_to")

        elif arity == "many":
            st.selectbox("Value type", [t for t in VALUE_TYPES if t != "null"], key="qb_vtype_many")
            st.text_input("Values (comma separated)", placeholder="e.g. AB, CD, XY", key="qb_list")

        elif arity == "like":
            st.text_input("Pattern", placeholder="e.g. J% or b_g", key="qb_pattern")
            st.text_input(
                "Escape character (optional)",
                placeholder="e.g. / - only needed if your pattern has a literal % or _",
                key="qb_escape",
            )

        if target == "New group" and st.session_state.groups:
            st.radio("Join this new group to the rest of the query using:", ["and", "or"],
                     horizontal=True, key="qb_new_connector")
        elif target != "New group":
            group_idx = int(target.split("#")[1]) - 1
            if st.session_state.groups[group_idx]["conditions"]:
                st.radio("Join this condition to the others already in that group using:",
                         ["and", "or"], horizontal=True, key="qb_internal_connector")

        st.button("\u2795 Add condition", on_click=add_condition_callback)

        msg = st.session_state.get("qb_msg")
        if msg:
            st.warning(msg)

        st.divider()

        # --- Show current groups/conditions ---------------------------------
        st.subheader("Current query structure")

        if not st.session_state.groups:
            st.info("No conditions added yet. Add one above to get started.")
        else:
            for gi, group in enumerate(st.session_state.groups):
                label = f"Group #{gi + 1}" if gi == 0 else f"**.{group['connector']}.** Group #{gi + 1}"
                st.markdown(label)
                for ci, cond in enumerate(group["conditions"]):
                    c1, c2 = st.columns([6, 1])
                    with c1:
                        prefix = "" if ci == 0 else f"  \u21b3 .{group['internal_connector']}. "
                        st.markdown(f"{prefix}`{build_condition_string(cond)}`")
                    with c2:
                        if st.button("\U0001f5d1\ufe0f", key=f"remove_{gi}_{ci}"):
                            group["conditions"].pop(ci)
                            if not group["conditions"]:
                                st.session_state.groups.pop(gi)
                            st.rerun()

            if st.button("Clear all"):
                st.session_state.groups = []
                st.rerun()

        st.divider()

        # --- Extra search parameters ----------------------------------------
        st.subheader("Other search parameters (optional)")

        col1, col2 = st.columns(2)
        with col1:
            offset = st.number_input("offset", min_value=0, key="qb_offset")
            order_by = st.text_input("orderBy", placeholder="e.g. code:asc, startDate:desc nulls last",
                                     key="qb_order_by")
        with col2:
            limit = st.number_input("limit", min_value=1, max_value=200, key="qb_limit")
            total_results = st.checkbox("totalResults (return total count)", key="qb_total_results")

        group_by = st.text_input("groupBy", placeholder="e.g. servicedMember.code", key="qb_group_by")
        aggregate = st.text_input("aggregate", placeholder="e.g. sum(totalCoveredAmount_amount):coveredAmount",
                                  key="qb_aggregate")

        st.divider()

        # --- Final output ---------------------------------------------------
        st.subheader("Generated q string Copy")

        query_string = build_full_query()
        st.code(query_string or "(no conditions yet)", language="text")

        if query_string:
            payload = {"resource": {"q": query_string}}
            if offset:
                payload["resource"]["offset"] = str(offset)
            if limit != 50:
                payload["resource"]["limit"] = str(limit)
            if order_by.strip():
                payload["resource"]["orderBy"] = order_by.strip()
            if total_results:
                payload["resource"]["totalResults"] = True
            if group_by.strip():
                payload["resource"]["groupBy"] = group_by.strip()
            if aggregate.strip():
                payload["resource"]["aggregate"] = aggregate.strip()

            st.subheader("Full POST payload Generated")
            st.code(json.dumps(payload, indent=2), language="json")

            st.caption(f"POST this to: `{resource_path or '/generic/{resource}/search'}`")

            st.subheader("Equivalent Groovy SearchBuilder code")
            st.caption(
                "Generated from the same conditions, per the Dynamic Logic Search Library "
                "(`.join()`, `.by()`, `.group()/.end()`, `.and()/.or()`, `.orderBy()`, `.execute()`)."
            )
            groovy_code = build_full_groovy(
                groovy_class_name,
                use_pagination=(limit != 50),
                page_size=limit if limit != 50 else None,
                order_by_raw=order_by,
            )
            st.code(groovy_code, language="groovy")
            st.caption(
                "Note: `eq(null)`/`neq(null)` test for null/not-null, matching the "
                "REST DSL's `eq(null)`/`neq(null)`. Only the first `orderBy` field is converted "
                "to `.orderBy(\"field\").ascending()/.descending()` - add more `.orderBy()` calls "
                "by hand for multi-field sorts. `groupBy`/`aggregate` have no direct SearchBuilder "
                "equivalent and are left out of the Groovy code."
            )


# ===========================================================================
# TOOL 2: JSON API RUNNER
# ===========================================================================
# Paste a JSON template containing {{variable}} placeholders, upload a CSV or
# JSON file that supplies values (one row/object = one API call), set the
# target URL + credentials, and run the batch.

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


def strip_json_comments(text: str) -> str:
    """Remove // line comments and /* */ block comments from a JSONC-style
    string, while leaving comment-like sequences inside actual string
    literals untouched. Also removes now-dangling trailing commas."""
    out = []
    i, n = 0, len(text)
    in_string = False
    escape = False

    while i < n:
        ch = text[i]

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\n\r":
                i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        out.append(ch)
        i += 1

    cleaned = "".join(out)
    # remove trailing commas left behind before a closing } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def preview_with_placeholders(template_str: str) -> str:
    """Build a parseable JSON preview by swapping {{var}} for a stand-in,
    string-boundary aware so embedded placeholders don't inject stray quotes."""
    out = []
    i, n = 0, len(template_str)

    while i < n:
        ch = template_str[i]

        if ch == '"':
            j = i + 1
            while j < n:
                if template_str[j] == "\\":
                    j += 2
                    continue
                if template_str[j] == '"':
                    break
                j += 1
            content = template_str[i + 1:j]
            stripped = content.strip()

            if PLACEHOLDER_RE.fullmatch(stripped):
                out.append('"__x__"')
            else:
                out.append('"' + PLACEHOLDER_RE.sub("x", content) + '"')

            i = j + 1
            continue

        m = PLACEHOLDER_RE.match(template_str, i)
        if m:
            out.append('"__x__"')
            i = m.end()
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def extract_variables(template_str: str):
    """Return the ordered, de-duplicated list of {{var}} names in the template."""
    seen = []
    for m in PLACEHOLDER_RE.finditer(template_str):
        name = m.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def _value_to_json_literal(var: str, row: dict) -> str:
    """Turn a row value into the right-hand-side JSON literal for a variable:
    numbers/booleans stay bare, nested JSON stays as-is, everything else is
    a quoted JSON string. Never adds quotes on top of quotes."""
    if var not in row or pd.isna(row[var]):
        return "null"
    val = row[var]
    if isinstance(val, (int, float, bool)):
        return json.dumps(val)
    s = str(val)
    try:
        # allow the CSV/JSON cell itself to already be valid JSON (e.g. nested object)
        json.loads(s)
        return s
    except (ValueError, TypeError):
        return json.dumps(s)


def substitute(template_str: str, row: dict) -> str:
    """Replace {{var}} with the row's value.
    - Placeholder is the ENTIRE contents of a string (e.g. "code": "{{Code}}")
      -> typed JSON literal (number/bool/nested-json/string), quotes handled for you.
    - Placeholder is bare, outside any quotes (e.g. "amount": {{amount}})
      -> typed JSON literal too.
    - Placeholder sits inside a larger string alongside other text
      (e.g. "descr": "Role for {{Name}}") -> "string concatenation":
      substituted as plain, JSON-escaped text, keeping the surrounding quotes intact.
    """
    out = []
    i, n = 0, len(template_str)

    while i < n:
        ch = template_str[i]

        if ch == '"':
            j = i + 1
            while j < n:
                if template_str[j] == "\\":
                    j += 2
                    continue
                if template_str[j] == '"':
                    break
                j += 1
            string_content = template_str[i + 1:j]
            stripped = string_content.strip()

            m = PLACEHOLDER_RE.fullmatch(stripped)
            if m:
                out.append(_value_to_json_literal(m.group(1), row))
            else:
                def concat_repl(match):
                    var = match.group(1)
                    val_str = "" if (var not in row or pd.isna(row[var])) else str(row[var])
                    return json.dumps(val_str)[1:-1]  # escaped, no surrounding quotes

                out.append('"' + PLACEHOLDER_RE.sub(concat_repl, string_content) + '"')

            i = j + 1
            continue

        bare_match = PLACEHOLDER_RE.match(template_str, i)
        if bare_match:
            out.append(_value_to_json_literal(bare_match.group(1), row))
            i = bare_match.end()
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def substitute_url(url_str: str, row: dict) -> str:
    """Replace {{var}} in a URL (path segments or query string) with the
    row's value, URL-encoded. Plain text substitution -- no JSON typing.
    e.g. https://api.example.com/v1/claims/{{claim_id}}?code={{Code}}
    """
    def repl(match):
        var = match.group(1)
        if var not in row or pd.isna(row[var]):
            return ""
        return quote(str(row[var]), safe="")

    return PLACEHOLDER_RE.sub(repl, url_str)


def load_data_file(uploaded_file):
    """Return a list of dict rows from an uploaded CSV or JSON file."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue().decode("utf-8-sig")

    if name.endswith(".json"):
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return data

    df = pd.read_csv(StringIO(raw))
    return df.to_dict(orient="records")


def build_headers(auth_type, token, username, password, header_name, header_value, extra_headers_raw):
    headers = {"Content-Type": "application/json"}

    if auth_type == "Bearer Token" and token:
        headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "Custom Header" and header_name:
        headers[header_name] = header_value

    if extra_headers_raw.strip():
        try:
            extra = json.loads(extra_headers_raw)
            headers.update(extra)
        except json.JSONDecodeError:
            st.warning("Extra headers isn't valid JSON - ignoring it.")

    return headers


def page_api_runner():
    st.title("\U0001f680 JSON API Runner")
    st.caption(
        "Send a single request like Postman, or batch-run it: add a JSON template and/or a data file "
        "(both optional) to substitute variables, one call per row."
    )

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("1. JSON Template (optional)")
        template_str = st.text_area(
            "Request body / query params - use {{variable_name}} for placeholders",
            height=260,
            key="ar_template",
            placeholder='{\n  "claimId": "{{claim_id}}",\n  "amount": {{amount}},\n  "provider": "{{provider_name}}"\n}',
        )

        detected_vars = extract_variables(template_str) if template_str else []
        if template_str:
            try:
                clean_template = strip_json_comments(template_str)
                json.loads(preview_with_placeholders(clean_template))
                st.success(
                    f"Valid JSON shape. Variables detected: {', '.join(detected_vars) if detected_vars else 'none'}"
                )
            except json.JSONDecodeError as e:
                st.error(f"Template isn't valid JSON once placeholders are filled in: {e}")

    with col2:
        st.subheader("2. Data (optional)")
        data_mode = st.radio(
            "Where do the variable values come from?",
            ["Upload CSV / JSON", "Enter manually"],
            horizontal=True,
            key="ar_data_mode",
            help="Skip both for a single request. Each row of data = one API call.",
        )

        rows = []
        data_file = None

        if data_mode == "Upload CSV / JSON":
            data_file = st.file_uploader(
                "Upload CSV or JSON to run one call per row (skip for a single request)",
                type=["csv", "json"],
            )

            # Uploader state is lost when navigating away, so parsed rows are cached.
            if data_file:
                try:
                    st.session_state["ar_rows"] = load_data_file(data_file)
                    st.session_state["ar_rows_name"] = data_file.name
                except Exception as e:
                    st.session_state["ar_rows"] = []
                    st.session_state["ar_rows_name"] = ""
                    st.error(f"Couldn't parse file: {e}")

            rows = st.session_state.get("ar_rows", [])
            if rows:
                st.success(f"Loaded {len(rows)} row(s) from `{st.session_state.get('ar_rows_name', 'file')}`.")
                st.dataframe(pd.DataFrame(rows).head(10), use_container_width=True, height=220)
                if not data_file:
                    st.caption("Using the file you loaded earlier (kept while you switch tools).")
                    if st.button("\U0001f9f9 Clear loaded data"):
                        st.session_state["ar_rows"] = []
                        st.session_state["ar_rows_name"] = ""
                        st.rerun()

        else:
            # Manual table: one column per {{variable}} found in the template / URL.
            cols = list(dict.fromkeys(
                detected_vars + extract_variables(st.session_state.get("ar_url", ""))
            ))
            if not cols:
                st.info("Add {{variable}} placeholders to the template or URL and a column "
                        "for each one will appear here.")
            else:
                sig = "|".join(cols)
                latest = st.session_state.get("ar_manual_latest")

                def blank_df():
                    return pd.DataFrame([{c: None for c in cols}], dtype="object")

                # (Re)build the starting table when the columns change or the page is re-entered,
                # carrying over anything already typed into matching columns.
                if (st.session_state.get("ar_manual_sig") != sig
                        or (st.session_state.get("_entered") and latest is not None)):
                    if latest is not None:
                        base = latest.reindex(columns=cols).astype("object")
                        if base.empty:
                            base = blank_df()
                    else:
                        base = blank_df()
                    st.session_state["ar_manual_base"] = base
                    st.session_state["ar_manual_sig"] = sig
                    st.session_state["ar_manual_ver"] = st.session_state.get("ar_manual_ver", 0) + 1

                edited = st.data_editor(
                    st.session_state["ar_manual_base"],
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"ar_manual_editor_{st.session_state['ar_manual_ver']}",
                    column_config={c: st.column_config.TextColumn(c) for c in cols},
                )
                st.session_state["ar_manual_latest"] = edited
                st.caption("Add rows with the + at the bottom of the table. Numbers and true/false "
                           "are sent as such; other text is sent as a string. Empty cells are sent as null.")

                cleaned = edited.dropna(how="all")
                rows = [
                    {k: (None if pd.isna(v) else v) for k, v in rec.items()}
                    for rec in cleaned.to_dict(orient="records")
                ]
                if rows:
                    st.success(f"{len(rows)} row(s) entered.")

    st.subheader("3. Endpoint & Credentials")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        url = st.text_input(
            "Target URL",
            key="ar_url",
            placeholder="https://api.example.com/v1/claims/{{claim_id}}?code={{Code}}",
            help="Supports {{variable_name}} placeholders, substituted per row (URL-encoded), "
                 "for path segments and/or query string values.",
        )
    with c2:
        method = st.selectbox("Method", ["POST", "PUT", "PATCH", "GET", "DELETE"], key="ar_method")
    with c3:
        timeout = st.number_input("Timeout (s)", min_value=1, max_value=120, key="ar_timeout")

    detected_url_vars = extract_variables(url) if url else []
    all_detected_vars = list(dict.fromkeys(detected_vars + detected_url_vars))

    if not rows and all_detected_vars:
        st.warning(
            f"No data file loaded, so these variables will be sent empty/null: {', '.join(all_detected_vars)}"
        )
    if rows and all_detected_vars:
        missing = [v for v in all_detected_vars if not all(v in r for r in rows)]
        if missing:
            st.warning(f"These template/URL variables are missing from some/all rows: {', '.join(missing)}")

    if url and detected_url_vars:
        st.caption(f"URL variables detected: {', '.join(detected_url_vars)}")

    auth_type = st.radio("Auth type", ["None", "Bearer Token", "Basic Auth", "Custom Header"],
                         horizontal=True, key="ar_auth_type")

    username = password = token = header_name = header_value = ""
    if auth_type == "Bearer Token":
        token = st.text_input("Bearer token", type="password", key="ar_token")
    elif auth_type == "Basic Auth":
        bc1, bc2 = st.columns(2)
        username = bc1.text_input("Username", key="ar_username")
        password = bc2.text_input("Password", type="password", key="ar_password")
    elif auth_type == "Custom Header":
        hc1, hc2 = st.columns(2)
        header_name = hc1.text_input("Header name", placeholder="X-API-Key", key="ar_header_name")
        header_value = hc2.text_input("Header value", type="password", key="ar_header_value")

    extra_headers_raw = st.text_area(
        "Extra headers (optional JSON)", height=70, placeholder='{"X-Correlation-Id": "batch-01"}',
        key="ar_extra_headers",
    )

    delay_between_calls = st.slider("Delay between calls (seconds)", 0.0, 5.0, step=0.1, key="ar_delay")

    max_chars = st.number_input(
        "Max response characters to keep (0 = keep full response)",
        min_value=0, step=1000, key="ar_max_chars",
        help="Applies to the results table and downloads. 0 keeps the complete response.",
    )

    st.divider()

    run_rows = rows if rows else [{}]  # no data file -> one single call
    run_label = f"\u25b6 Run batch ({len(run_rows)} rows)" if rows else "\u25b6 Send request"
    run = st.button(run_label, type="primary", disabled=not url)

    if run:
        headers = build_headers(auth_type, token, username, password, header_name, header_value,
                                extra_headers_raw)
        auth = (username, password) if auth_type == "Basic Auth" else None

        results = []
        progress = st.progress(0, text="Starting...")
        clean_template_str = strip_json_comments(template_str) if template_str.strip() else ""

        for i, row in enumerate(run_rows):
            payload = None
            if clean_template_str:
                try:
                    body_str = substitute(clean_template_str, row)
                    payload = json.loads(body_str)
                except json.JSONDecodeError as e:
                    results.append({"row": i, "url": "", "status": "template_error", "detail": str(e),
                                    "response": ""})
                    progress.progress((i + 1) / len(run_rows))
                    continue

            row_url = substitute_url(url, row)

            start = time.time()
            try:
                resp = requests.request(
                    method,
                    row_url,
                    json=payload if (method != "GET" and payload is not None) else None,
                    params=payload if (method == "GET" and isinstance(payload, dict)) else None,
                    headers=headers,
                    auth=auth,
                    timeout=timeout,
                )
                elapsed = round(time.time() - start, 3)
                try:
                    resp_body = resp.json()
                    resp_body_str = json.dumps(resp_body)
                except ValueError:
                    resp_body_str = resp.text
                if max_chars:
                    resp_body_str = resp_body_str[:int(max_chars)]

                results.append(
                    {
                        "row": i,
                        "url": row_url,
                        "status": resp.status_code,
                        "detail": "ok" if resp.ok else "http_error",
                        "elapsed_s": elapsed,
                        "response": resp_body_str,
                    }
                )
            except requests.RequestException as e:
                results.append({"row": i, "url": row_url, "status": "request_error", "detail": str(e),
                                "response": ""})

            progress.progress((i + 1) / len(run_rows), text=f"Row {i + 1}/{len(run_rows)}")
            if delay_between_calls:
                time.sleep(delay_between_calls)

        progress.empty()
        st.session_state["results"] = results

    if "results" in st.session_state:
        st.subheader("Results")
        results_df = pd.DataFrame(st.session_state["results"])
        st.dataframe(results_df, use_container_width=True, height=350)

        ok_count = sum(1 for r in st.session_state["results"] if str(r["status"]) in ("200", "201"))
        st.caption(f"{ok_count}/{len(st.session_state['results'])} calls returned 2xx.")

        with st.expander("View full response for a row"):
            row_ids = [r["row"] for r in st.session_state["results"]]
            pick = st.selectbox("Row", row_ids, key="ar_view_row")
            picked = next(r for r in st.session_state["results"] if r["row"] == pick)
            st.caption(f"{picked.get('url', '')}  |  status: {picked.get('status')}  |  "
                       f"{len(str(picked.get('response', '')))} characters")
            try:
                st.json(json.loads(picked["response"]))
            except (ValueError, TypeError):
                st.code(str(picked.get("response", "")), language="text")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download results (CSV)",
                results_df.to_csv(index=False).encode("utf-8"),
                file_name="api_run_results.csv",
                mime="text/csv",
            )
        with dl2:
            st.download_button(
                "Download results (JSON)",
                json.dumps(st.session_state["results"], indent=2).encode("utf-8"),
                file_name="api_run_results.json",
                mime="application/json",
            )


# ===========================================================================
# NAVIGATION
# ===========================================================================
PAGES = {
    "\U0001f50e Query API Builder": page_query_builder,
    "\U0001f680 JSON API Runner": page_api_runner,
}

keep_widget_state()

with st.sidebar:
    st.title("\U0001f9f0 OHI Toolkit")
    selected = st.radio("Navigate", list(PAGES.keys()), key="nav_page")
    st.divider()
    st.caption("Switching tools keeps what you've entered.")

# Flag when a page has just been navigated to (used to restore the manual data table).
st.session_state["_entered"] = st.session_state.get("_prev_page") != selected
st.session_state["_prev_page"] = selected

PAGES[selected]()