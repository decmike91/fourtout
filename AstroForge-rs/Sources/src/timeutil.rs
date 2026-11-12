//! Dates ISO 8601 (équivalent de datetime.fromisoformat) et jour julien.

use anyhow::{anyhow, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime};

/// Analyse une date ISO (avec ou sans fuseau) -> instant UTC naïf.
pub fn parse_iso_utc(s: &str) -> Result<NaiveDateTime> {
    let t = s.trim();
    let t_z = if let Some(stripped) = t.strip_suffix('Z') { format!("{}+00:00", stripped) } else { t.to_string() };
    if let Ok(d) = DateTime::parse_from_rfc3339(&t_z) {
        return Ok(d.naive_utc());
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f%:z", "%Y-%m-%d %H:%M:%S%.f%:z", "%Y-%m-%dT%H:%M%:z"] {
        if let Ok(d) = DateTime::parse_from_str(&t_z, fmt) {
            return Ok(d.naive_utc());
        }
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"] {
        if let Ok(d) = NaiveDateTime::parse_from_str(t, fmt) {
            return Ok(d);
        }
    }
    if let Ok(d) = NaiveDate::parse_from_str(t, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap());
    }
    Err(anyhow!("Date ISO invalide : « {} »", s))
}

/// Secondes (flottantes) depuis l'époque Unix.
pub fn timestamp(d: &NaiveDateTime) -> f64 {
    d.and_utc().timestamp() as f64 + d.and_utc().timestamp_subsec_micros() as f64 / 1e6
}

/// Jour julien (UTC) d'une date ISO, décalée de `offset_s` secondes.
pub fn julian_date(iso: &str, offset_s: f64) -> Option<f64> {
    let d = parse_iso_utc(iso).ok()?;
    Some(timestamp(&d) / 86400.0 + 2440587.5 + offset_s / 86400.0)
}

pub fn iso_ms(d: &NaiveDateTime) -> String {
    d.format("%Y-%m-%dT%H:%M:%S%.3f").to_string()
}

pub fn iso_s(d: &NaiveDateTime) -> String {
    d.format("%Y-%m-%dT%H:%M:%S").to_string()
}
