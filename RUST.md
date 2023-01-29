# RUST TIPS

## Use `&str`

Pass `&str` instead of `String` or `&String`.

## Stop Using Slice Indexing

Instead of:

```
let points: Vec<Coordinate> = ...;
let mut differences = Vec::new();

for i in 1..points.len() {
    let current = points[i];
    let previous = points[i - 1];
    differences.push(current - previous);
}
```

Use this:

```
let points: Vec<Coordinate> = ...;
let mut differences = Vec::new();

for [previous,current] in points.array_windows().copied() {
    differences.push(current - previous);
}
```

Or even:

```
let points: Vec<Coordinate> = ...;
let differences: Vec<_> = points
    .array_windows()
    .copied()
    .map(|[previous,current]| current - previous)
    .collect();
```

## Don't Use Sentinel Values

Use `Option<>` and `Result<>` where appropriate.

## Use Enums

Use enums and pattern matching where applicable.

## Proper Error Handling

Use the `?` operator and implement Error trait. Possibly use `thiserror` library.

## Implement the `Default` Trait

This allows for quick and clear initialization.

## Implement `From` and `TryFrom` Trait

Using this with errors allows you to do this:

```
enum CliError {
    IoError(io::Error),
    ParseError(num::ParseIntError),
}

impl From<io::Error> for CliError {
    fn from(error: io::Error) -> Self {
        CliError::IoError(error)
    }
}

impl From<num::ParseIntError> for CliError {
    fn from(error: num::ParseIntError) -> Self {
        CliError::ParseError(error)
    }
}

fn open_and_parse_file(file_name: &str) -> Result<i32,CliError> {
    let mut contents = fs::read_to_string(&file_name)?;
    let num: i32 = contents.trim().parse()?;
    Ok(num)
}
```

## Implement 'FromStr' Trait

This helps immensely when interacting with text inputs and text files and such.

## Use `todo!` Macro

Stuff that isn't implemented yet can be marked as `todo!();` so the compiler won't complain about it being empty.

## Use `concat!` Macro

This allows you to concatenate anything into a static string slice.

## Use `format!` Macro

This allows you to use interpolation into a static string slice.

## Use `cargo fmt`

This ensures your formatting is consistent and easy to read. This can be done automatically in vscode.

## Use `cargo clippy`

This catches common mistakes and improves yout code.

## Reduce the Need for Smart Pointers

When designing a struct with methods to modify fields (like original OOP), the need for `Rc<RefCell<T>>` or similar constructions quickly appears. Instead, design the code such that you don't hold long-lived references to other objects. Refer to other structs via method parameters, rather than smart pointers in the struct.

## And Also

- async trait methods in nightly
- generic associated types
- let-else statements (much cleaner for init/done existing APIs)
- debug info on Linux
- `IntoFuture` Trait
- 