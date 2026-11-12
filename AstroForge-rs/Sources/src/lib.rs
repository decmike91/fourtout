//! AstroForge — traitement d'images astronomiques (ciel profond & planétaire).
//! Portage Rust du moteur Python : calibration, alignement, empilement,
//! drizzle, astrométrie, photométrie, finition, planétaire, séquences batch.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const APP_NAME: &str = "AstroForge";

pub mod fft;
pub mod frames;
pub mod geom;
pub mod image;
pub mod io;
pub mod ndimage;
pub mod stats;
pub mod timeutil;

pub use image::Image;
pub mod display;
pub mod features;
pub mod paths;
pub mod registration;
pub mod stacking;
pub mod starmatch;
pub mod stars;
pub mod calibration;
pub mod defects;
pub mod masters;
pub mod narrowband;
pub mod session;
pub mod drizzle;
pub mod livestack;
pub mod multisession;
pub mod normalization;
pub mod subframes;
pub mod annotate;
pub mod astrometry;
pub mod draw;
pub mod equipment;
pub mod mosaic;
pub mod platesolve;
pub mod wcs;
pub mod animation;
pub mod colorcal;
pub mod comet;
pub mod derotation;
pub mod effects;
pub mod finishing;
pub mod gradient;
pub mod lightcurve;
pub mod masks;
pub mod planetary;
pub mod planetary_auto;
pub mod timelapse;
pub mod restoration;
pub mod solar;
pub mod stretch;
pub mod wavelets;
pub mod ai;
pub mod pipeline;
pub mod profiles;
pub mod sequence;
pub mod updates;
