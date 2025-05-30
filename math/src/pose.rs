use {
    crate::*,
    std::ops::{
        Add,
        AddAssign,
        Mul,
        MulAssign,
    },
};

/// 3D pose (position and orientation).
#[derive(Copy,Clone,Debug,PartialEq)]
pub struct Pose<T> {
    pub p: Vec3<T>,
    pub o: Quat<T>,
}

impl<T> One for Pose<T> where Vec3<T>: Zero, Quat<T>: One {
    const ONE: Self = Pose { p: Vec3::ZERO,o: Quat::ONE, };
}

/// Pose * vector.
impl<T> Mul<Vec3<T>> for Pose<T> where Vec3<T>: Add<Output=Vec3<T>>,Quat<T>: Mul<Vec3<T>,Output=Vec3<T>> {
    type Output = Vec3<T>;
    fn mul(self,other: Vec3<T>) -> Vec3<T> {
        self.o * other + self.p
    }
}

/// Pose * pose.
impl<T> Mul<Pose<T>> for Pose<T> where Vec3<T>: Add<Output=Vec3<T>>,Quat<T>: Copy + Mul<Quat<T>,Output=Quat<T>> + Mul<Vec3<T>,Output=Vec3<T>> {
    type Output = Pose<T>;
    fn mul(self,other: Pose<T>) -> Pose<T> {
        Pose {
            p: self.p + self.o * other.p,
            o: self.o * other.o,
        }
    }
}

/// Pose *= pose.
impl<T> MulAssign<Pose<T>> for Pose<T> where Vec3<T>: AddAssign<Vec3<T>>,Quat<T>: Copy + Mul<Vec3<T>,Output=Vec3<T>> + MulAssign<Quat<T>> {
    fn mul_assign(&mut self,other: Pose<T>) {
        self.p += self.o * other.p;
        self.o *= other.o;
    }
}

macro_rules! pose_real_impl {
    ($($t:ty)+) => {
        $(
            impl Pose<$t> {
                /// Invert the pose. 
                pub fn inv(self) -> Self {
                    let o = self.o.inv();
                    Pose {
                        p: o * -self.p,
                        o: o,
                    }
                }
            }
        )+
    }
}

pose_real_impl! { f32 f64 }

// if `T as U` exists, `Pose<U>::from(Pose<T>)` should also exist
impl From<Pose<f32>> for Pose<f64> { fn from(value: Pose<f32>) -> Self { Pose { p: value.p.into(),o: value.o.into(), } } }
impl From<Pose<f64>> for Pose<f32> { fn from(value: Pose<f64>) -> Self { Pose { p: value.p.into(),o: value.o.into(), } } }