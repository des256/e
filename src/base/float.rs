pub trait Float: Sized {
    fn invsqrt(self) -> Self;
    fn modf(self) -> (Self,Self);
    fn mix(self,other: Self,a: Self) -> Self;
    fn mixb(self,other: Self,a: bool) -> Self;
    fn step(self,edge: Self) -> Self;
    fn smoothstep(self,edge0: Self,edge1: Self) -> Self;
    fn fma(self,b: Self,c: Self) -> Self;
}

macro_rules! impl_float {
    ($($t:ty)+) => {
        $(
            impl Float for $t {
                fn invsqrt(self) -> Self {
                    1.0 / self.sqrt()
                }

                fn modf(self) -> (Self,Self) {
                    (self.floor(),self.fract())
                }

                fn mix(self,other: Self,a: Self) -> Self {
                    self * (1.0 - a) + other * a
                }

                fn mixb(self,other: Self,a: bool) -> Self {
                    if a { other } else { self }
                }

                fn step(self,edge: Self) -> Self {
                    if self < edge { 0.0 } else { 1.0 }
                }

                fn smoothstep(self,edge0: Self,edge1: Self) -> Self {
                    let f = (self - edge0) / (edge1 - edge0);
                    let t = f.clamp(0.0,1.0);
                    t * t * (3.0 - 2.0 * t)
                }

                fn fma(self,b: Self,c: Self) -> Self {
                    self * b + c
                }
            }
        )+
    }
}

impl_float!(f32 f64);