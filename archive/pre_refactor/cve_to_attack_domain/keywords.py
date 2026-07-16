"""Domain keyword definitions for CVE to ATT&CK domain mapping."""

from typing import Dict, Set

ICS_KEYWORDS: Set[str] = {
    "plc",
    "scada",
    "rtu",
    "modbus",
    "dnp3",
    "hmi",
    "siemens",
    "rockwell",
    "schneider",
    "abb",
    "honeywell",
}

MOBILE_KEYWORDS: Set[str] = {
    "android",
    "ios",
    "iphone",
    "ipad",
    "watchos",
    "tvos",
}

ENTERPRISE_KEYWORDS: Set[str] = {
    "windows",
    "linux",
    "macos",
    "apache",
    "nginx",
    "tomcat",
    "confluence",
}

DOMAIN_ORDER = ["ICS", "Mobile", "Enterprise"]

DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "ICS": ICS_KEYWORDS,
    "Mobile": MOBILE_KEYWORDS,
    "Enterprise": ENTERPRISE_KEYWORDS,
}
