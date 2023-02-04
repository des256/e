use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Formatter,
            Result,
        },
        ops::{
            Add,
            Sub,
            Mul,
            Div,
            AddAssign,
            SubAssign,
            MulAssign,
            DivAssign,
            Neg,
        },
    },
};

#[derive(Copy,Clone,Debug)]
pub struct Mat4x4<T> {
    pub x: Vec4<T>,
    pub y: Vec4<T>,
    pub z: Vec4<T>,
    pub w: Vec4<T>,
}

macro_rules! mat4x4_impl {
    ($($t:ty)+) => {
        $(
            impl Mat4x4<$t> {
                pub fn transpose(self) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 {
                            x: self.x.x,
                            y: self.y.x,
                            z: self.z.x,
                            w: self.w.x,
                        },
                        y: Vec4 {
                            x: self.x.y,
                            y: self.y.y,
                            z: self.z.y,
                            w: self.w.y,
                        },
                        z: Vec4 {
                            x: self.x.z,
                            y: self.y.z,
                            z: self.z.z,
                            w: self.w.z,
                        },
                        w: Vec4 {
                            x: self.x.w,
                            y: self.y.w,
                            z: self.z.w,
                            w: self.w.w,
                        },
                    }
                }

                pub fn det(self) -> $t {
                    let xx = self.x.x;
                    let xy = self.x.y;
                    let xz = self.x.z;
                    let xw = self.x.w;
                    let yx = self.y.x;
                    let yy = self.y.y;
                    let yz = self.y.z;
                    let yw = self.y.w;
                    let zx = self.z.x;
                    let zy = self.z.y;
                    let zz = self.z.z;
                    let zw = self.z.w;
                    let wx = self.w.x;
                    let wy = self.w.y;
                    let wz = self.w.z;
                    let ww = self.w.w;
                    let axx = yy * (zz * ww - wz * zw) - yz * (zw * wy - ww * zy) + yw * (zy * wz - wy * zz);
                    let axy = yw * (zx * wz - wx * zz) + yx * (zz * ww - wz * zw) - yz * (zw * wx - ww * zx);
                    let axz = yw * (zx * wy - wx * zy) - yx * (zy * ww - wy * zw) + yy * (zw * wx - ww * zx);
                    let axw = yy * (zz * wx - wz * wx) + yz * (wx * wy - wx * zy) - yx * (zy * wz - wy * zz);
                    xx * axx + xy * axy + xz * axz + xw * axw
                }
            }

            impl PartialEq for Mat4x4<$t> {
                fn eq(&self,other: &Self) -> bool {
                    (self.x == other.x) && (self.y == other.y) && (self.z == other.z) && (self.w == other.w)
                }
            }

            impl Display for Mat4x4<$t> {
                fn fmt(&self,f: &mut Formatter) -> Result {
                    write!(f,"[{},{},{},{}]",self.x,self.y,self.z,self.w)
                }
            }

            // matrix + matrix
            impl Add<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Self;
                fn add(self,other: Self) -> Self::Output {
                    Mat4x4 {
                        x: self.x + other.x,
                        y: self.y + other.y,
                        z: self.z + other.z,
                        w: self.w + other.w,
                    }
                }
            }

            // matrix += matrix
            impl AddAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn add_assign(&mut self,other: Self) {
                    self.x += other.x;
                    self.y += other.y;
                    self.z += other.z;
                    self.w += other.w;
                }
            }

            // matrix - matrix
            impl Sub<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Self;
                fn sub(self,other: Self) -> Self::Output {
                    Mat4x4 {
                        x: self.x - other.x,
                        y: self.y - other.y,
                        z: self.z - other.z,
                        w: self.w - other.w,
                    }
                }
            }

            // matrix -= matrix
            impl SubAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn sub_assign(&mut self,other: Self) {
                    self.x -= other.x;
                    self.y -= other.y;
                    self.z -= other.z;
                    self.w -= other.w;
                }
            }

            // scalar * matrix
            impl Mul<Mat4x4<$t>> for $t {
                type Output = Mat4x4<$t>;
                fn mul(self,other: Mat4x4<$t>) -> Self::Output {
                    Mat4x4 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                        w: self * other.w,
                    }
                }
            }

            // matrix * scalar
            impl Mul<$t> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn mul(self,other: $t) -> Self::Output {
                    Mat4x4 {
                        x: self.x * other,
                        y: self.y * other,
                        z: self.z * other,
                        w: self.w * other,
                    }
                }
            }

            // matrix * vector
            impl Mul<Vec4<$t>> for Mat4x4<$t> {
                type Output = Vec4<$t>;
                fn mul(self,other: Vec4<$t>) -> Self::Output {
                    Vec4 {
                        x: self.x.x * other.x + self.x.y * other.y + self.x.z * other.z + self.x.w * other.w,
                        y: self.y.x * other.x + self.y.y * other.y + self.y.z * other.z + self.y.w * other.w,
                        z: self.z.x * other.x + self.z.y * other.y + self.z.z * other.z + self.z.w * other.w,
                        w: self.w.x * other.x + self.w.y * other.y + self.w.z * other.z + self.w.w * other.w,
                    }
                }
            }

            // matrix * matrix
            impl Mul<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn mul(self,other: Mat4x4<$t>) -> Self::Output {
                    Mat4x4 {
                        x: Vec4 {
                            x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x + self.x.w * other.w.x,
                            y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y + self.x.w * other.w.y,
                            z: self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z + self.x.w * other.w.z,
                            w: self.x.x * other.x.w + self.x.y * other.y.w + self.x.z * other.z.w + self.x.w * other.w.w,
                        },
                        y: Vec4 {
                            x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x + self.y.w * other.w.x,
                            y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y + self.y.w * other.w.y,
                            z: self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z + self.y.w * other.w.z,
                            w: self.y.x * other.x.w + self.y.y * other.y.w + self.y.z * other.z.w + self.y.w * other.w.w,
                        },
                        z: Vec4 {
                            x: self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x + self.z.w * other.w.x,
                            y: self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y + self.z.w * other.w.y,
                            z: self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z + self.z.w * other.w.z,
                            w: self.z.x * other.x.w + self.z.y * other.y.w + self.z.z * other.z.w + self.z.w * other.w.w,
                        },
                        w: Vec4 {
                            x: self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x + self.w.w * other.w.x,
                            y: self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y + self.w.w * other.w.y,
                            z: self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z + self.w.w * other.w.z,
                            w: self.w.x * other.x.w + self.w.y * other.y.w + self.w.z * other.z.w + self.w.w * other.w.w,
                        }
                    }
                }
            }

            // matrix *= scalar
            impl MulAssign<$t> for Mat4x4<$t> {
                fn mul_assign(&mut self,other: $t) {
                    self.x.x *= other;
                    self.x.y *= other;
                    self.x.z *= other;
                    self.x.w *= other;
                    self.y.x *= other;
                    self.y.y *= other;
                    self.y.z *= other;
                    self.y.w *= other;
                    self.z.x *= other;
                    self.z.y *= other;
                    self.z.z *= other;
                    self.z.w *= other;
                    self.w.x *= other;
                    self.w.y *= other;
                    self.w.z *= other;
                    self.w.w *= other;
                }
            }

            // matrix *= matrix
            impl MulAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn mul_assign(&mut self,other: Mat4x4<$t>) {
                    let xx = self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x + self.x.w * other.w.x;
                    let xy = self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y + self.x.w * other.w.y;
                    let xz = self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z + self.x.w * other.w.z;
                    let xw = self.x.x * other.x.w + self.x.y * other.y.w + self.x.z * other.z.w + self.x.w * other.w.w;
                    let yx = self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x + self.y.w * other.w.x;
                    let yy = self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y + self.y.w * other.w.y;
                    let yz = self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z + self.y.w * other.w.z;
                    let yw = self.y.x * other.x.w + self.y.y * other.y.w + self.y.z * other.z.w + self.y.w * other.w.w;
                    let zx = self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x + self.z.w * other.w.x;
                    let zy = self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y + self.z.w * other.w.y;
                    let zz = self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z + self.z.w * other.w.z;
                    let zw = self.z.x * other.x.w + self.z.y * other.y.w + self.z.z * other.z.w + self.z.w * other.w.w;
                    let wx = self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x + self.w.w * other.w.x;
                    let wy = self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y + self.w.w * other.w.y;
                    let wz = self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z + self.w.w * other.w.z;
                    let ww = self.w.x * other.x.w + self.w.y * other.y.w + self.w.z * other.z.w + self.w.w * other.w.w;
                    Mat4x4 {
                        x: Vec4 { x: xx,y: xy,z: xz,w: xw, },
                        y: Vec4 { x: yx,y: yy,z: yz,w: yw, },
                        z: Vec4 { x: zx,y: zy,z: zz,w: zw, },
                        w: Vec4 { x: wx,y: wy,z: wz,w: ww, },
                    }
                }
            }

            // scalar / matrix
            impl Div<Mat4x4<$t>> for $t {
                type Output = Mat4x4<$t>;
                fn div(self,other: Mat4x4<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            // matrix / scalar
            impl Div<$t> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn div(self,other: $t) -> Self::Output {
                    self.x /= other;
                    self.y /= other;
                    self.z /= other;
                    self.w /= other;
                }
            }

            // matrix / matrix
            impl Div<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn div(self,other: Mat4x4<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            // matrix /= scalar
            impl DivAssign<$t> for Mat4x4<$t> {
                fn div_assign(&mut self,other: $t) {
                    self.x /= other;
                    self.y /= other;
                    self.z /= other;
                    self.w /= other;
                }
            }

            // matrix /= matrix
            impl DivAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn div_assign(&mut self,other: Mat4x4<$t>) {
                    self *= other.inv()
                }
            }

            // -matrix
            impl Neg for Mat4x4<$t> {
                fn neg(self) -> Self {
                    Mat4x4 {
                        x: -self.x,
                        y: -self.y,
                        z: -self.z,
                        w: -self.w,
                    }
                }
            }
        )+
    }
}

mat4x4_impl! { isize i8 i16 i32 i64 i128 f32 f64 }

macro_rules! mat4x4_real_impl {
    ($($t:ty)+) => {
        $(
            impl Mat4x4<$t> {
                pub fn inv(self) -> Self {
                    let xx = self.x.x;
                    let xy = self.x.y;
                    let xz = self.x.z;
                    let xw = self.x.w;
                    let yx = self.y.x;
                    let yy = self.y.y;
                    let yz = self.y.z;
                    let yw = self.y.w;
                    let zx = self.z.x;
                    let zy = self.z.y;
                    let zz = self.z.z;
                    let zw = self.z.w;
                    let wx = self.w.x;
                    let wy = self.w.y;
                    let wz = self.w.z;
                    let ww = self.w.w;
                    let axx = yy * (zz * ww - wz * zw) - yz * (zw * wy - ww * zy) + yw * (zy * wz - wy * zz);
                    let axy = yw * (zx * wz - wx * zz) + yx * (zz * ww - wz * zw) - yz * (zw * wx - ww * zx);
                    let axz = yw * (zx * wy - wx * zy) - yx * (zy * ww - wy * zw) + yy * (zw * wx - ww * zx);
                    let axw = yy * (zz * wx - wz * wx) + yz * (wx * wy - wx * zy) - yx * (zy * wz - wy * zz);
                    let det = xx * axx + xy * axy + xz * axz + xw * axw;
                    if det == 0.0 {
                        return self;
                    }
                    let ayx = zz * (ww * xy - xw * wy) + zw * (wy * xz - xy * wz) - zy * (wz * xw - xz * ww);
                    let ayy = zz * (ww * xx - xw * wx) - zw * (wx * xz - xx * wz) + zx * (wz * xw - xz * ww);
                    let ayz = zx * (wy * xw - xy * ww) + zy * (ww * xx - xw * wx) - zw * (wx * xy - xx * wy);
                    let ayw = zx * (wy * xz - xy * wz) - zy * (wz * xx - xz * wx) + zz * (wx * xy - xx * wy);
                    let azx = wy * (xz * yw - yz * xw) - wz * (xw * yy - yz * xy) + ww * (xy * yz - yy * xz);
                    let azy = ww * (xx * yz - yx * xz) + wx * (xz * yw - yz * xw) - wz * (xw * yx - yw * xx);
                    let azz = ww * (xx * yy - yx * xy) - wx * (xy * yw - yy * xw) + wy * (xw * yx - yw * xx);
                    let azw = wy * (xz * yx - yz * xx) + wz * (xx * yy - yx * xy) - wx * (xy * yz - yy * xz);
                    let awx = xz * (yw * zy - zw * yy) + xw * (yy * zz - zy * yz) - xy * (yz * zw - zz * yw);
                    let awy = xz * (yw * zx - zw * yx) - xw * (yx * zz - zx * yz) + xx * (yz * zw - zz * yw);
                    let awz = xx * (yy * zw - zy * yw) + xy * (yw * zx - zw * yx) - xw * (yx * zy - zx * yy);
                    let aww = xx * (yy * zz - zy * yz) - xy * (yz * zx - zz * yx) + xz * (yx * zy - zx * yy);
                    Mat4x4::new(
                        axx,ayx,azx,awx,
                        axy,ayy,azy,awy,
                        axz,ayz,azz,awz,
                        axw,ayw,azw,aww,
                    ) / det
                }
            }
        )+
    }
}

mat4x4_real_impl! { f32 f64 }
