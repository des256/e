pub trait One {
    const ONE: Self;
}

impl One for u8 {
    const ONE: Self = 1;
}

impl One for u16 {
    const ONE: Self = 1;
}

impl One for u32 {
    const ONE: Self = 1;
}

impl One for u64 {
    const ONE: Self = 1;
}

impl One for u128 {
    const ONE: Self = 1;
}

impl One for usize {
    const ONE: Self = 1;
}

impl One for i8 {
    const ONE: Self = 1;
}

impl One for i16 {
    const ONE: Self = 1;
}

impl One for i32 {
    const ONE: Self = 1;
}

impl One for i64 {
    const ONE: Self = 1;
}

impl One for i128 {
    const ONE: Self = 1;
}

impl One for isize {
    const ONE: Self = 1;
}

impl One for f32 {
    const ONE: Self = 1.0;
}

impl One for f64 {
    const ONE: Self = 1.0;
}
