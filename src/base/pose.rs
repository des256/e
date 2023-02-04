use {
    crate::*,
};

#[derive(Copy,Clone,Debug)]
struct Pose<T> {
    p: Vec3<T>,
    o: Quaternion<T>,
}
