/// Logs a message at the **info** level.
///
/// In debug builds, prints to stdout with the format:
///
/// ```text
/// INFO <file>:<line>:<column>: <message>
/// ```
///
/// In release builds, this macro is a no-op.
///
/// # Examples
///
/// ```no_run
/// use base::info;
///
/// info!("server started on port {}", 8080);
/// // DEBUG: INFO src/main.rs:3:1: server started on port 8080
/// ```
#[cfg(debug_assertions)]
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => { println!("INFO {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

/// Logs a message at the **debug** level.
///
/// In debug builds, prints to stdout with the format:
///
/// ```text
/// DEBUG <file>:<line>:<column>: <message>
/// ```
///
/// In release builds, this macro is a no-op.
///
/// # Examples
///
/// ```no_run
/// use base::debug;
///
/// let x = 42;
/// debug!("value of x = {x}");
/// // DEBUG: DEBUG src/main.rs:4:1: value of x = 42
/// ```
#[cfg(debug_assertions)]
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => { println!("DEBUG {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

/// Logs a message at the **error** level.
///
/// In debug builds, prints to stdout with the format:
///
/// ```text
/// ERROR <file>:<line>:<column>: <message>
/// ```
///
/// In release builds, this macro is a no-op.
///
/// # Examples
///
/// ```no_run
/// use base::error;
///
/// error!("connection failed: {}", "timeout");
/// // DEBUG: ERROR src/main.rs:3:1: connection failed: timeout
/// ```
#[cfg(debug_assertions)]
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => { println!("ERROR {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

/// Logs a message at the **warning** level.
///
/// In debug builds, prints to stdout with the format:
///
/// ```text
/// WARNING <file>:<line>:<column>: <message>
/// ```
///
/// In release builds, this macro is a no-op.
///
/// # Examples
///
/// ```no_run
/// use base::warn;
///
/// warn!("disk usage at {}%", 92);
/// // DEBUG: WARNING src/main.rs:3:1: disk usage at 92%
/// ```
#[cfg(debug_assertions)]
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => { println!("WARNING {}:{}:{}: {}",file!(),line!(),column!(),format_args!($($arg)*)) };
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {};
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {};
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {};
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {};
}
