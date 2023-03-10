# TODO

- migrate to Result<> and ?, only panic at top level
- turn all .to_string() into &str references, make the utility AST in /src/gpu/sc to accept &'static str, because the strings will be hardcoded at compile time
