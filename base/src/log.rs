#[cfg(debug_assertions)]
#[macro_export]
macro_rules! logi {
    ($($arg:tt)*) => { println!("INFO {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! logd {
    ($($arg:tt)*) => { println!("DEBUG {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! loge {
    ($($arg:tt)*) => { println!("ERROR {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! logw {
    ($($arg:tt)*) => { println!("WARNING {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! logi {
    ($($arg:tt)*) => {};
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! logd {
    ($($arg:tt)*) => {};
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! loge {
    ($($arg:tt)*) => {};
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! logw {
    ($($arg:tt)*) => {};
}
