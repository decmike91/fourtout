//! Séquences d'images à accès différé (vidéo SER projetée en mémoire, vue
//! « tonale » ajustée à la demande, liste en mémoire).

use crate::image::Image;

pub trait Frames: Sync {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> Image;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Frames for Vec<Image> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn get(&self, i: usize) -> Image {
        self[i].clone()
    }
}

impl Frames for [Image] {
    fn len(&self) -> usize {
        <[Image]>::len(self)
    }
    fn get(&self, i: usize) -> Image {
        self[i].clone()
    }
}

impl<T: Frames + ?Sized> Frames for &T {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn get(&self, i: usize) -> Image {
        (**self).get(i)
    }
}

/// Vue par indices sur une autre séquence.
pub struct Subset<'a, F: Frames + ?Sized> {
    pub src: &'a F,
    pub idx: Vec<usize>,
}

impl<F: Frames + ?Sized> Frames for Subset<'_, F> {
    fn len(&self) -> usize {
        self.idx.len()
    }
    fn get(&self, i: usize) -> Image {
        self.src.get(self.idx[i])
    }
}
