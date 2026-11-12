//! En-tête de type FITS : liste ordonnée de mots-clés -> valeurs typées.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum HVal {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl HVal {
    /// Conversion numérique façon `float(v)` Python (chaînes acceptées).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            HVal::Int(i) => Some(*i as f64),
            HVal::Float(f) => Some(*f),
            HVal::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            HVal::Str(s) => {
                let t = s.trim();
                let low = t.to_ascii_lowercase();
                match low.as_str() {
                    "nan" => Some(f64::NAN),
                    "inf" | "infinity" | "+inf" => Some(f64::INFINITY),
                    "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
                    _ => t.replace('_', "").parse::<f64>().ok(),
                }
            }
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            HVal::Int(i) => Some(*i),
            HVal::Float(f) if f.is_finite() => Some(*f as i64),
            HVal::Bool(b) => Some(*b as i64),
            HVal::Str(s) => s.trim().parse::<i64>().ok(),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            HVal::Bool(b) => *b,
            HVal::Int(i) => *i != 0,
            HVal::Float(f) => *f != 0.0,
            HVal::Str(s) => !s.is_empty(),
        }
    }
}

/// Représentation façon `str(v)` Python (valeur astropy).
impl fmt::Display for HVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HVal::Str(s) => write!(f, "{}", s),
            HVal::Int(i) => write!(f, "{}", i),
            HVal::Float(v) => write!(f, "{}", py_float_repr(*v)),
            HVal::Bool(b) => write!(f, "{}", if *b { "True" } else { "False" }),
        }
    }
}

impl From<&str> for HVal {
    fn from(s: &str) -> Self {
        HVal::Str(s.to_string())
    }
}
impl From<String> for HVal {
    fn from(s: String) -> Self {
        HVal::Str(s)
    }
}
impl From<f64> for HVal {
    fn from(v: f64) -> Self {
        HVal::Float(v)
    }
}
impl From<i64> for HVal {
    fn from(v: i64) -> Self {
        HVal::Int(v)
    }
}
impl From<usize> for HVal {
    fn from(v: usize) -> Self {
        HVal::Int(v as i64)
    }
}
impl From<bool> for HVal {
    fn from(v: bool) -> Self {
        HVal::Bool(v)
    }
}

/// repr() d'un flottant Python (« 300.0 », « 1e-07 »).
pub fn py_float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{:?}", v);
    // Rust : « 1e-7 » ; Python : « 1e-07 »
    if let Some((m, e)) = s.split_once('e') {
        let (sign, digits) = if let Some(d) = e.strip_prefix('-') { ("-", d) } else { ("+", e) };
        return format!("{}e{}{:0>2}", m, sign, digits);
    }
    s
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Header {
    pub cards: Vec<(String, HVal, String)>,
}

impl Header {
    pub fn new() -> Self {
        Header { cards: Vec::new() }
    }

    pub fn get(&self, key: &str) -> Option<&HVal> {
        self.cards.iter().find(|c| c.0 == key).map(|c| &c.1)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    pub fn get_str(&self, key: &str) -> Option<String> {
        self.get(key).map(|v| v.to_string())
    }

    pub fn set(&mut self, key: &str, val: impl Into<HVal>) {
        self.set_with_comment(key, val, "");
    }

    pub fn set_with_comment(&mut self, key: &str, val: impl Into<HVal>, comment: &str) {
        let v = val.into();
        if let Some(c) = self.cards.iter_mut().find(|c| c.0 == key) {
            c.1 = v;
            if !comment.is_empty() {
                c.2 = comment.to_string();
            }
        } else {
            self.cards.push((key.to_string(), v, comment.to_string()));
        }
    }

    pub fn remove(&mut self, key: &str) {
        self.cards.retain(|c| c.0 != key);
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.cards.iter().map(|c| c.0.as_str())
    }

    pub fn extend(&mut self, other: &Header) {
        for (k, v, c) in &other.cards {
            self.set_with_comment(k, v.clone(), c);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }
}
